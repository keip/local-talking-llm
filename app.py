import io
import re
import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import soundfile as sf
import requests
import argparse
from queue import Queue
from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI

console = Console()
stt = whisper.load_model("base.en")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Local Voice Assistant with ChatterBox TTS")
parser.add_argument("--model", type=str, default="gemma3", help="LLM model to use")
parser.add_argument("--tts-url", type=str, default="http://192.168.0.226:8000", help="TTS server URL")
args = parser.parse_args()

# Modern prompt template using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize LLM (LM Studio serves an OpenAI-compatible API)
llm = ChatOpenAI(
    model=args.model,
    base_url="http://192.168.0.226:1234/v1",
    api_key="lm-studio",
)

# Create the chain with modern LCEL syntax
chain = prompt_template | llm

# Chat history storage
chat_sessions = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]

# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    # Use a default session ID for this simple voice assistant
    session_id = "voice_assistant_session"

    # Invoke the chain with history
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )

    # ChatOpenAI returns an AIMessage object, extract the text content
    if hasattr(response, "content"):
        text = response.content.strip()
    else:
        text = str(response).strip()

    # Strip Qwen-style <think>...</think> reasoning tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def synthesize_remote(text: str, tts_url: str) -> tuple[int, np.ndarray]:
    """Sends text to the remote TTS server and returns audio."""
    resp = requests.post(
        f"{tts_url}/v1/audio/speech",
        json={"text": text},
        timeout=120,
    )
    resp.raise_for_status()
    buffer = io.BytesIO(resp.content)
    audio, sr = sf.read(buffer)
    return sr, audio


def play_audio(sample_rate, audio_array):
    """Plays the given audio data using the sounddevice library."""
    sd.play(audio_array, sample_rate)
    sd.wait()


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant with ChatterBox TTS")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"[blue]LLM model: {args.model}")
    console.print(f"[blue]TTS server: {args.tts_url}")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    try:
        while True:
            console.input(
                "🎤 Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="dots"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating LLM response...", spinner="dots"):
                    response = get_llm_response(text)
                console.print(f"[cyan]Assistant: {response}")

                with console.status("Synthesizing speech (remote)...", spinner="dots"):
                    sample_rate, audio_array = synthesize_remote(
                        response, args.tts_url
                    )

                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")

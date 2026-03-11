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
parser.add_argument(
    "--mode", type=str, default="manual", choices=["manual", "always-on"],
    help="Interaction mode: 'manual' (press Enter) or 'always-on' (wake word)",
)
parser.add_argument("--wake-phrase", type=str, default="hey morgan", help="Wake phrase to activate the assistant")
parser.add_argument("--silence-timeout", type=float, default=1.5, help="Seconds of silence for end-of-speech")
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


def run_manual_mode(args):
    """Original press-Enter-to-record interaction loop."""
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


def _drain_queue(q):
    """Discard all pending items from a queue."""
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break


def run_always_on_mode(args):
    """Always-on wake word mode with automatic end-of-speech detection.

    Uses Silero VAD to detect speech, then Whisper to check for the wake
    phrase. Once activated, records the user's command until silence timeout,
    then processes through the LLM + TTS pipeline.
    """
    from wake_word import WakeWordDetector
    from vad import VoiceActivityDetector
    from beep import beep_start, beep_end

    detector = WakeWordDetector(stt_model=stt, wake_phrase=args.wake_phrase)
    vad = VoiceActivityDetector(silence_timeout=args.silence_timeout)

    # Short silence timeout for wake word detection (just detect end of phrase)
    WAKE_SILENCE = 0.8
    NO_SPEECH_TIMEOUT = 3.0
    CHUNK_SIZE = 1024  # 64ms at 16kHz (multiple of 512 for Silero VAD)

    audio_queue = Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            console.print(f"[red]Audio: {status}")
        audio_queue.put(bytes(indata))

    state = "LISTENING"
    wake_audio = []     # audio chunks while detecting wake phrase
    command_audio = []  # audio chunks for actual command

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1,
        blocksize=CHUNK_SIZE, callback=audio_callback,
    ):
        console.print(f"[green]Listening for '{args.wake_phrase}'...")

        while True:
            raw_data = audio_queue.get()
            chunk_np = np.frombuffer(raw_data, dtype=np.int16)

            if state == "LISTENING":
                result = vad.process_chunk(chunk_np)

                if result["is_speech"]:
                    # Speech detected — start collecting for wake word check
                    state = "WAKE_DETECT"
                    wake_audio = [chunk_np.copy()]
                    vad.reset()
                    vad.silence_timeout = WAKE_SILENCE
                    # Re-process this chunk so VAD tracks it
                    vad.process_chunk(chunk_np)
                continue

            elif state == "WAKE_DETECT":
                wake_audio.append(chunk_np.copy())
                result = vad.process_chunk(chunk_np)

                if result["speech_ended"] or (
                    not result["speech_detected_ever"]
                    and result["silence_duration"] > NO_SPEECH_TIMEOUT
                ):
                    # Speech segment ended — check for wake phrase
                    audio_float = (
                        np.concatenate(wake_audio).astype(np.float32) / 32768.0
                    )
                    detected, text = detector.check(audio_float)

                    if detected:
                        console.print(f"[cyan]Wake phrase detected! (\"{text}\")")
                        beep_start()
                        state = "RECORDING"
                        command_audio = []
                        vad.reset()
                        vad.silence_timeout = args.silence_timeout
                        # Drain audio accumulated during beep
                        _drain_queue(audio_queue)
                    else:
                        # Not a wake phrase, go back to listening
                        state = "LISTENING"
                        vad.reset()
                        vad.silence_timeout = WAKE_SILENCE
                    wake_audio = []
                continue

            elif state == "RECORDING":
                command_audio.append(chunk_np.copy())
                result = vad.process_chunk(chunk_np)

                if result["speech_ended"]:
                    beep_end()
                    state = "PROCESSING"
                elif (
                    not result["speech_detected_ever"]
                    and result["silence_duration"] > NO_SPEECH_TIMEOUT
                ):
                    console.print("[yellow]No command detected, resuming listening.")
                    state = "LISTENING"
                    vad.reset()
                    vad.silence_timeout = WAKE_SILENCE
                    console.print(f"[green]Listening for '{args.wake_phrase}'...")
                    continue
                else:
                    continue

            if state == "PROCESSING":
                audio_np = (
                    np.concatenate(command_audio).astype(np.float32) / 32768.0
                )

                if audio_np.size == 0:
                    state = "LISTENING"
                    vad.reset()
                    vad.silence_timeout = WAKE_SILENCE
                    console.print(f"[green]Listening for '{args.wake_phrase}'...")
                    continue

                with console.status("Transcribing...", spinner="dots"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                # Check for stop phrase
                if text.strip().lower().rstrip(".!") in ("thank you", "thanks"):
                    console.print("[cyan]Morgan: You're welcome!")
                    state = "LISTENING"
                    vad.reset()
                    vad.silence_timeout = WAKE_SILENCE
                    console.print(f"[green]Listening for '{args.wake_phrase}'...")
                    continue

                # Skip empty/noise transcriptions
                if not text or text.startswith("("):
                    console.print("[yellow]No speech recognized, resuming listening.")
                    state = "LISTENING"
                    vad.reset()
                    vad.silence_timeout = WAKE_SILENCE
                    console.print(f"[green]Listening for '{args.wake_phrase}'...")
                    continue

                with console.status("Generating LLM response...", spinner="dots"):
                    response = get_llm_response(text)
                console.print(f"[cyan]Morgan: {response}")

                with console.status("Synthesizing speech...", spinner="dots"):
                    sample_rate, audio_array = synthesize_remote(
                        response, args.tts_url
                    )

                play_audio(sample_rate, audio_array)

                # Drain audio queue to discard self-echo from TTS playback
                _drain_queue(audio_queue)
                state = "LISTENING"
                vad.reset()
                vad.silence_timeout = WAKE_SILENCE
                console.print(f"[green]Listening for '{args.wake_phrase}'...")


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant with ChatterBox TTS")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"[blue]Mode: {args.mode}")
    console.print(f"[blue]LLM model: {args.model}")
    console.print(f"[blue]TTS server: {args.tts_url}")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    try:
        if args.mode == "always-on":
            run_always_on_mode(args)
        else:
            run_manual_mode(args)
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")

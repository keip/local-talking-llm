import io
import re
import time
import struct
import threading
from queue import Queue

import numpy as np
import whisper
import sounddevice as sd
import soundfile as sf
import requests
from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI

from . import web_ui
from .audio import record_audio, beep_start, beep_end
from .wake_word import WakeWordDetector
from .vad import VoiceActivityDetector


class VoiceAssistant:
    """Orchestrates the full voice assistant pipeline: STT -> LLM -> TTS."""

    def __init__(self, args):
        self.args = args
        self.console = Console()

        # Load Whisper STT model
        self.stt = whisper.load_model("base.en")

        # Build LLM chain
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a pragmatic AI assistant with the voice of Morgan Freeman. You are polite, respectful, and aim to provide concise responses of 1 to 2 sentences. Respond with plain text only — no emojis, no markdown, no special characters or symbols."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        llm = ChatOpenAI(
            model=args.model,
            base_url="http://192.168.0.226:1234/v1",
            api_key="lm-studio",
        )

        chain = prompt_template | llm

        self._chat_sessions: dict[str, InMemoryChatMessageHistory] = {}
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._chat_sessions:
            self._chat_sessions[session_id] = InMemoryChatMessageHistory()
        return self._chat_sessions[session_id]

    def transcribe(self, audio_np: np.ndarray) -> str:
        """Transcribes audio data using the Whisper model."""
        result = self.stt.transcribe(audio_np, fp16=False)
        text = result["text"].strip()
        return text

    def get_llm_response(self, text: str) -> str:
        """Generates a response using the language model."""
        session_id = "voice_assistant_session"

        response = self.chain_with_history.invoke(
            {"input": text},
            config={"session_id": session_id}
        )

        if hasattr(response, "content"):
            text = response.content.strip()
        else:
            text = str(response).strip()

        # Strip Qwen-style <think>...</think> reasoning tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Strip markdown formatting
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)       # italic
        text = re.sub(r"`(.+?)`", r"\1", text)         # inline code
        text = re.sub(r"#+\s*", "", text)               # headings

        # Remove emojis and other non-ASCII symbols
        text = re.sub(r"[^\x00-\x7F]+", "", text)

        # Collapse extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def synthesize_remote(self, text: str) -> tuple[int, np.ndarray]:
        """Sends text to the remote TTS server and returns audio."""
        resp = requests.post(
            f"{self.args.tts_url}/v1/audio/speech",
            json={"text": text},
            timeout=120,
        )
        resp.raise_for_status()
        buffer = io.BytesIO(resp.content)
        audio, sr = sf.read(buffer)
        return sr, audio

    def stream_and_play_remote(self, text: str):
        """Streams audio from the TTS server and plays chunks as they arrive."""
        resp = requests.post(
            f"{self.args.tts_url}/v1/audio/speech/stream",
            json={"text": text},
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()

        raw = resp.raw

        # Read 8-byte header: sample_rate (int32) + channels (int32)
        header = raw.read(8)
        sample_rate, channels = struct.unpack("<ii", header)

        stream = sd.OutputStream(samplerate=sample_rate, channels=channels, dtype="float32")
        stream.start()

        try:
            BLOCK_SIZE = 4096  # bytes
            remainder = b""
            while True:
                try:
                    data = raw.read(BLOCK_SIZE)
                except Exception:
                    break
                if not data:
                    break
                data = remainder + data
                usable = len(data) - (len(data) % 4)
                remainder = data[usable:]
                if usable > 0:
                    audio = np.frombuffer(data[:usable], dtype=np.float32)
                    stream.write(audio.reshape(-1, channels))

            # Drain remaining buffered audio
            time.sleep(stream.latency + 0.1)
        finally:
            stream.stop()
            stream.close()

    def run_manual_mode(self):
        """Press-Enter-to-record interaction loop."""
        web_ui.emit({"type": "state", "state": "listening"})
        while True:
            self.console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            web_ui.emit({"type": "state", "state": "recording"})
            data_queue: Queue = Queue()
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
                web_ui.emit({"type": "state", "state": "processing"})
                with self.console.status("Transcribing...", spinner="dots"):
                    text = self.transcribe(audio_np)
                self.console.print(f"[yellow]You: {text}")
                web_ui.emit({"type": "message", "role": "user", "text": text})

                with self.console.status("Generating LLM response...", spinner="dots"):
                    response = self.get_llm_response(text)
                self.console.print(f"[cyan]Assistant: {response}")
                web_ui.emit({"type": "message", "role": "assistant", "text": response})

                self.console.print("[green]Speaking...")
                web_ui.emit({"type": "state", "state": "speaking"})
                self.stream_and_play_remote(response)
                web_ui.emit({"type": "state", "state": "listening"})
            else:
                self.console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )
                web_ui.emit({"type": "state", "state": "listening"})

    def run_always_on_mode(self):
        """Always-on wake word mode with automatic end-of-speech detection."""
        detector = WakeWordDetector(stt_model=self.stt, wake_phrase=self.args.wake_phrase)
        vad = VoiceActivityDetector(silence_timeout=self.args.silence_timeout)

        WAKE_SILENCE = 0.8
        NO_SPEECH_TIMEOUT = 3.0
        CHUNK_SIZE = 1024

        audio_queue: Queue = Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                self.console.print(f"[red]Audio: {status}")
            audio_queue.put(bytes(indata))

        state = "LISTENING"
        wake_audio: list = []
        command_audio: list = []

        with sd.RawInputStream(
            samplerate=16000, dtype="int16", channels=1,
            blocksize=CHUNK_SIZE, callback=audio_callback,
        ):
            self.console.print(f"[green]Listening for '{self.args.wake_phrase}'...")
            web_ui.emit({"type": "state", "state": "listening"})

            while True:
                raw_data = audio_queue.get()
                chunk_np = np.frombuffer(raw_data, dtype=np.int16)

                if state == "LISTENING":
                    result = vad.process_chunk(chunk_np)

                    if result["is_speech"]:
                        state = "WAKE_DETECT"
                        wake_audio = [chunk_np.copy()]
                        vad.reset()
                        vad.silence_timeout = WAKE_SILENCE
                        vad.process_chunk(chunk_np)
                    continue

                elif state == "WAKE_DETECT":
                    wake_audio.append(chunk_np.copy())
                    result = vad.process_chunk(chunk_np)

                    if result["speech_ended"] or (
                        not result["speech_detected_ever"]
                        and result["silence_duration"] > NO_SPEECH_TIMEOUT
                    ):
                        audio_float = (
                            np.concatenate(wake_audio).astype(np.float32) / 32768.0
                        )
                        detected, text = detector.check(audio_float)

                        if detected:
                            self.console.print(f'[cyan]Wake phrase detected! ("{text}")')
                            web_ui.emit({"type": "state", "state": "wake_detected"})
                            beep_start()
                            state = "RECORDING"
                            web_ui.emit({"type": "state", "state": "recording"})
                            command_audio = []
                            vad.reset()
                            vad.silence_timeout = self.args.silence_timeout
                            _drain_queue(audio_queue)
                        else:
                            state = "LISTENING"
                            web_ui.emit({"type": "state", "state": "listening"})
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
                        self.console.print("[yellow]No command detected, resuming listening.")
                        state = "LISTENING"
                        web_ui.emit({"type": "state", "state": "listening"})
                        vad.reset()
                        vad.silence_timeout = WAKE_SILENCE
                        self.console.print(f"[green]Listening for '{self.args.wake_phrase}'...")
                        continue
                    else:
                        continue

                elif state == "CONVERSING":
                    command_audio.append(chunk_np.copy())
                    result = vad.process_chunk(chunk_np)

                    if result["speech_ended"]:
                        beep_end()
                        state = "PROCESSING"
                    elif (
                        not result["speech_detected_ever"]
                        and result["silence_duration"] > self.args.idle_timeout
                    ):
                        self.console.print("[yellow]Idle timeout, returning to wake phrase mode.")
                        state = "LISTENING"
                        web_ui.emit({"type": "state", "state": "listening"})
                        vad.reset()
                        vad.silence_timeout = WAKE_SILENCE
                        self.console.print(f"[green]Listening for '{self.args.wake_phrase}'...")
                        continue
                    else:
                        continue

                if state == "PROCESSING":
                    web_ui.emit({"type": "state", "state": "processing"})
                    audio_np = (
                        np.concatenate(command_audio).astype(np.float32) / 32768.0
                    )

                    if audio_np.size == 0:
                        state = "LISTENING"
                        web_ui.emit({"type": "state", "state": "listening"})
                        vad.reset()
                        vad.silence_timeout = WAKE_SILENCE
                        self.console.print(f"[green]Listening for '{self.args.wake_phrase}'...")
                        continue

                    with self.console.status("Transcribing...", spinner="dots"):
                        text = self.transcribe(audio_np)
                    self.console.print(f"[yellow]You: {text}")
                    web_ui.emit({"type": "message", "role": "user", "text": text})

                    # Check for stop phrase
                    if text.strip().lower().rstrip(".!") in ("thank you", "thanks"):
                        self.console.print("[cyan]Morgan: You're welcome!")
                        web_ui.emit({"type": "message", "role": "assistant", "text": "You're welcome!"})
                        state = "LISTENING"
                        web_ui.emit({"type": "state", "state": "listening"})
                        vad.reset()
                        vad.silence_timeout = WAKE_SILENCE
                        self.console.print(f"[green]Listening for '{self.args.wake_phrase}'...")
                        continue

                    # Skip empty/noise transcriptions
                    if not text or text.startswith("("):
                        self.console.print("[yellow]No speech recognized, resuming listening.")
                        state = "LISTENING"
                        web_ui.emit({"type": "state", "state": "listening"})
                        vad.reset()
                        vad.silence_timeout = WAKE_SILENCE
                        self.console.print(f"[green]Listening for '{self.args.wake_phrase}'...")
                        continue

                    with self.console.status("Generating LLM response...", spinner="dots"):
                        response = self.get_llm_response(text)
                    self.console.print(f"[cyan]Morgan: {response}")
                    web_ui.emit({"type": "message", "role": "assistant", "text": response})

                    self.console.print("[green]Speaking...")
                    web_ui.emit({"type": "state", "state": "speaking"})
                    self.stream_and_play_remote(response)

                    # Drain audio queue to discard self-echo from TTS playback
                    _drain_queue(audio_queue)
                    beep_start()
                    state = "CONVERSING"
                    web_ui.emit({"type": "state", "state": "recording"})
                    command_audio = []
                    vad.reset()
                    vad.silence_timeout = self.args.silence_timeout
                    self.console.print("[green]Continuing conversation — speak or wait to return to wake phrase mode...")


def _drain_queue(q: Queue):
    """Discard all pending items from a queue."""
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break

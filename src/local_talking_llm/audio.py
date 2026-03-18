import time
import threading

import numpy as np
import sounddevice as sd
from queue import Queue


def record_audio(stop_event: threading.Event, data_queue: Queue):
    """Captures audio data from the user's microphone and adds it to a queue.

    Args:
        stop_event: An event that, when set, signals the function to stop recording.
        data_queue: A queue to which the recorded audio data will be added.
    """
    def callback(indata, frames, time, status):
        if status:
            print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def play_audio(sample_rate: int, audio_array: np.ndarray):
    """Plays the given audio data using the sounddevice library."""
    sd.play(audio_array, sample_rate)
    sd.wait()


def play_beep(frequency: float = 880, duration_ms: int = 150, volume: float = 0.3):
    """Play a short sine-wave beep tone.

    Args:
        frequency: Tone frequency in Hz.
        duration_ms: Duration in milliseconds.
        volume: Volume multiplier (0.0-1.0).
    """
    sample_rate = 24000
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    tone = (volume * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    sd.play(tone, sample_rate)
    sd.wait()


def beep_start():
    """High-pitched beep indicating 'I'm listening'."""
    play_beep(frequency=880, duration_ms=150)


def beep_end():
    """Lower-pitched beep indicating 'processing your request'."""
    play_beep(frequency=440, duration_ms=150)

import numpy as np
import sounddevice as sd


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

import time

import numpy as np
import torch


class VoiceActivityDetector:
    """Detects end-of-speech using Silero VAD.

    Tracks consecutive silence duration and signals end-of-speech
    when silence exceeds the configured timeout.
    """

    def __init__(
        self,
        silence_timeout: float = 1.5,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
    ):
        self.silence_timeout = silence_timeout
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold

        self.model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self.model.eval()

        self._silence_start: float | None = None
        self._speech_detected_ever = False

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """Process an audio chunk through Silero VAD.

        Args:
            audio_chunk: numpy array of int16 audio at 16kHz.
                         Automatically splits into 512-sample sub-chunks
                         as required by Silero VAD.

        Returns:
            dict with keys: is_speech, speech_ended, silence_duration,
            speech_detected_ever.
        """
        audio_float = audio_chunk.astype(np.float32) / 32768.0

        # Silero VAD requires exactly 512 samples at 16kHz per call.
        # Split larger chunks and take the max confidence.
        VAD_WINDOW = 512
        is_speech = False

        for i in range(0, len(audio_float), VAD_WINDOW):
            sub = audio_float[i : i + VAD_WINDOW]
            if len(sub) < VAD_WINDOW:
                break  # skip incomplete tail
            tensor = torch.from_numpy(sub)
            confidence = self.model(tensor, self.sample_rate).item()
            if confidence >= self.speech_threshold:
                is_speech = True

        if is_speech:
            self._speech_detected_ever = True
            self._silence_start = None
        else:
            if self._silence_start is None:
                self._silence_start = time.monotonic()

        silence_duration = 0.0
        if self._silence_start is not None:
            silence_duration = time.monotonic() - self._silence_start

        speech_ended = (
            self._speech_detected_ever
            and silence_duration >= self.silence_timeout
        )

        return {
            "is_speech": is_speech,
            "speech_ended": speech_ended,
            "silence_duration": silence_duration,
            "speech_detected_ever": self._speech_detected_ever,
        }

    def reset(self):
        """Reset state for a new recording session."""
        self._silence_start = None
        self._speech_detected_ever = False
        self.model.reset_states()

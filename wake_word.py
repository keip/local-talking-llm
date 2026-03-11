import numpy as np
import whisper


class WakeWordDetector:
    """Detects a wake phrase using Whisper transcription gated by VAD.

    Since openwakeword requires onnxruntime (unavailable on Python 3.14),
    this uses the already-loaded Whisper model to transcribe short speech
    segments and checks for the wake phrase via string matching.
    """

    def __init__(
        self,
        stt_model: whisper.Whisper,
        wake_phrase: str = "hey morgan",
    ):
        """
        Args:
            stt_model: Pre-loaded Whisper model instance (shared with main app).
            wake_phrase: The phrase to listen for (case-insensitive).
        """
        self.stt_model = stt_model
        self.wake_phrase = wake_phrase.lower()

    def check(self, audio_np: np.ndarray) -> tuple[bool, str]:
        """Transcribe a short audio clip and check for the wake phrase.

        Args:
            audio_np: float32 audio array at 16kHz (already normalized).

        Returns:
            Tuple of (wake_word_detected, transcribed_text).
        """
        if audio_np.size == 0:
            return False, ""

        result = self.stt_model.transcribe(audio_np, fp16=False)
        text = result["text"].strip()

        detected = self.wake_phrase in text.lower()
        return detected, text

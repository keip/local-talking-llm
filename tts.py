import nltk
import warnings
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from mlx_audio.tts.utils import load_model

warnings.filterwarnings("ignore")


class TextToSpeechService:
    def __init__(self, model_name: str = "mlx-community/Chatterbox-TTS-4bit"):
        """
        Initializes the TextToSpeechService with mlx-audio-plus Chatterbox TTS.

        Args:
            model_name (str): The HuggingFace model identifier for the MLX TTS model.
        """
        print(f"Loading TTS model: {model_name}")
        self.model = load_model(model_name)
        self.sample_rate = 24000

    def synthesize(self, text: str, audio_prompt_path: str | None = None):
        """
        Synthesizes audio from the given text using Chatterbox TTS.

        Args:
            text (str): The input text to be synthesized.
            audio_prompt_path (str, optional): Path to audio file for voice cloning.

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        kwargs = {"text": text}
        if audio_prompt_path:
            kwargs["ref_audio"] = audio_prompt_path
            kwargs["ref_text"] = "."

        audio_pieces = []
        for result in self.model.generate(**kwargs):
            audio_pieces.append(result.audio)

        audio_array = np.concatenate(audio_pieces) if len(audio_pieces) > 1 else audio_pieces[0]
        return self.sample_rate, audio_array

    def long_form_synthesize(self, text: str, audio_prompt_path: str | None = None):
        """
        Synthesizes audio from long-form text, sentence by sentence.

        Args:
            text (str): The input text to be synthesized.
            audio_prompt_path (str, optional): Path to audio file for voice cloning.

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.05 * self.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(
                sent, audio_prompt_path=audio_prompt_path
            )
            pieces += [audio_array, silence.copy()]

        return self.sample_rate, np.concatenate(pieces)

    def stream_long_form_synthesize(self, text: str, audio_prompt_path: str | None = None):
        """
        Generator that yields (sample_rate, audio_array) per sentence for streaming playback.
        Pre-synthesizes the next sentence in the background to reduce inter-sentence gaps.
        """
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.05 * self.sample_rate))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.synthesize, sentences[0], audio_prompt_path)
            for i in range(len(sentences)):
                _, audio_array = future.result()
                if i + 1 < len(sentences):
                    future = executor.submit(self.synthesize, sentences[i + 1], audio_prompt_path)
                yield self.sample_rate, np.concatenate([audio_array, silence])

    def save_voice_sample(self, text: str, output_path: str, audio_prompt_path: str | None = None):
        """
        Saves a voice sample to file.

        Args:
            text (str): The text to synthesize.
            output_path (str): Path where to save the audio file.
            audio_prompt_path (str, optional): Path to audio file for voice cloning.
        """
        _, audio_array = self.synthesize(text, audio_prompt_path=audio_prompt_path)
        sf.write(output_path, audio_array, self.sample_rate)

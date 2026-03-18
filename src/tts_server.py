import io
import struct
import argparse

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from tts import TextToSpeechService

app = FastAPI(title="ChatterBox TTS Server")

# Module-level state, initialized by main()
_tts: TextToSpeechService | None = None
_voice_path: str | None = None


class TTSRequest(BaseModel):
    text: str


@app.post("/v1/audio/speech")
async def synthesize(request: TTSRequest):
    assert _tts is not None
    sample_rate, audio_array = _tts.long_form_synthesize(
        request.text, audio_prompt_path=_voice_path
    )
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    return Response(content=buffer.getvalue(), media_type="audio/wav")


@app.post("/v1/audio/speech/stream")
async def synthesize_stream(request: TTSRequest):
    assert _tts is not None

    def generate():
        yield struct.pack("<ii", _tts.sample_rate, 1)

        try:
            for _, audio_chunk in _tts.stream_long_form_synthesize(
                request.text, audio_prompt_path=_voice_path
            ):
                yield audio_chunk.astype(np.float32).tobytes()
        except Exception as e:
            print(f"TTS synthesis error: {e}")

    return StreamingResponse(generate(), media_type="application/octet-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    global _tts, _voice_path

    parser = argparse.ArgumentParser(description="ChatterBox TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--voice", type=str, required=True, help="Path to voice WAV file on server")
    args = parser.parse_args()

    _voice_path = args.voice
    _tts = TextToSpeechService()

    # Warmup: pre-compile MLX kernels so the first real request is fast
    print("Warming up TTS model...")
    list(_tts.stream_long_form_synthesize("Hello.", audio_prompt_path=_voice_path))
    print("TTS ready.")

    import uvicorn
    print(f"Starting TTS server on {args.host}:{args.port}")
    print(f"Voice file: {args.voice}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

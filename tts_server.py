import io
import struct
import argparse
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from tts import TextToSpeechService

parser = argparse.ArgumentParser(description="ChatterBox TTS Server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
parser.add_argument("--port", type=int, default=8000, help="Server port")
parser.add_argument("--voice", type=str, required=True, help="Path to voice WAV file on server")
args = parser.parse_args()

tts = TextToSpeechService()

app = FastAPI(title="ChatterBox TTS Server")


class TTSRequest(BaseModel):
    text: str


@app.post("/v1/audio/speech")
async def synthesize(request: TTSRequest):
    sample_rate, audio_array = tts.long_form_synthesize(
        request.text, audio_prompt_path=args.voice
    )
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    return Response(content=buffer.getvalue(), media_type="audio/wav")


@app.post("/v1/audio/speech/stream")
async def synthesize_stream(request: TTSRequest):
    def generate():
        # Header: sample_rate (int32) + channels (int32)
        yield struct.pack("<ii", tts.sample_rate, 1)

        for sample_rate, audio_chunk in tts.stream_long_form_synthesize(
            request.text, audio_prompt_path=args.voice
        ):
            yield audio_chunk.astype(np.float32).tobytes()

    return StreamingResponse(generate(), media_type="application/octet-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    print(f"Starting TTS server on {args.host}:{args.port}")
    print(f"Voice file: {args.voice}")
    uvicorn.run(app, host=args.host, port=args.port)

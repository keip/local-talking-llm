# Local Talking LLM

A fully local voice assistant that listens, thinks, and speaks — powered by Whisper, Ollama/LM Studio, and Chatterbox TTS. No cloud services, no API keys, everything runs on your machine.

```
You speak → Whisper transcribes → LLM responds → Chatterbox speaks back
```

## Features

- **Speech Recognition** — OpenAI Whisper (`base.en`) for fast, accurate transcription
- **LLM Conversations** — Any model via Ollama or LM Studio (Gemma 3, Llama, Qwen, etc.)
- **Text-to-Speech** — Chatterbox TTS with voice cloning and emotion control
- **Wake Word Detection** — Hands-free "always-on" mode with customizable wake phrase
- **Voice Activity Detection** — Silero VAD for automatic speech start/stop
- **Web Dashboard** — Real-time conversation view with status indicators
- **Audio Feedback** — Beep tones for state transitions (listening, processing, etc.)

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Microphone  │────▶│   Whisper    │────▶│  LLM (Ollama│────▶│  Chatterbox  │
│   Input      │     │   STT        │     │  /LM Studio)│     │   TTS Server │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                     │
                                                                     ▼
                                                              ┌─────────────┐
                                                              │   Speaker   │
                                                              │   Output    │
                                                              └─────────────┘
```

The system runs as **two processes**: the main app handles recording, transcription, and LLM calls, while the TTS server handles speech synthesis separately (allowing it to run on a different machine if needed).

## Prerequisites

- **Python 3.11+** (3.13 recommended)
- **Ollama** or **LM Studio** running with a model loaded
- A working microphone and speakers

## Installation

### Using uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

# Clone and install
git clone https://github.com/janczibula/local-talking-llm.git
cd local-talking-llm
uv sync
source .venv/bin/activate

# Download NLTK data for sentence tokenization
python -c "import nltk; nltk.download('punkt_tab')"
```

### Using pip

```bash
git clone https://github.com/janczibula/local-talking-llm.git
cd local-talking-llm
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
python -c "import nltk; nltk.download('punkt_tab')"
```

### Set up the LLM backend

Install [Ollama](https://ollama.ai) or [LM Studio](https://lmstudio.ai) and pull a model:

```bash
# With Ollama
ollama pull gemma3

# Or use LM Studio's UI to download and serve a model
```

## Usage

The system requires two terminals — one for the TTS server and one for the main app.

### 1. Start the TTS server

```bash
# With voice cloning (provide a 10-30s clear audio sample)
tts-server --voice voices/morgan_freeman.wav

# Or via make
make tts-server
```

### 2. Start the main app

```bash
# Manual mode — press Enter to start/stop recording
local-talking-llm

# Always-on mode — uses wake word detection
local-talking-llm --mode always-on

# Custom wake phrase
local-talking-llm --mode always-on --wake-phrase "hey jarvis"
```

### 3. Open the Web UI

Navigate to `http://localhost:8080` for a real-time dashboard showing conversation history and assistant status.

## Command-Line Options

### local-talking-llm

| Option | Default | Description |
|---|---|---|
| `--model` | `gemma3` | LLM model name |
| `--tts-url` | `http://192.168.0.226:8000` | TTS server endpoint |
| `--mode` | `manual` | `manual` or `always-on` |
| `--wake-phrase` | `hey morgan` | Wake phrase for always-on mode |
| `--silence-timeout` | `1.5` | Seconds of silence to end recording |
| `--idle-timeout` | `8.0` | Seconds of idle to return to wake mode |
| `--ui-port` | `8080` | Web UI port |

### tts-server

| Option | Default | Description |
|---|---|---|
| `--voice` | (required) | Path to reference audio for voice cloning |
| `--host` | `0.0.0.0` | Server host |
| `--port` | `8000` | Server port |

## Interaction Modes

### Manual Mode

1. Press **Enter** to start recording
2. Speak your question
3. Press **Enter** to stop recording
4. The assistant transcribes, generates a response, and speaks it back

### Always-On Mode

1. The assistant listens continuously for the wake phrase
2. Say the wake phrase (e.g., "hey morgan")
3. A beep confirms activation — speak your question
4. The assistant responds and stays in conversation mode
5. After an idle timeout, it returns to listening for the wake phrase

## Project Structure

```
├── voices/
│   └── morgan_freeman.wav      # Example voice cloning reference
├── src/
│   └── local_talking_llm/
│       ├── cli.py              # CLI entry point (argparse + main)
│       ├── assistant.py        # VoiceAssistant orchestrator class
│       ├── audio.py            # Audio recording, playback, and beep tones
│       ├── tts.py              # Chatterbox TTS model wrapper
│       ├── tts_server.py       # FastAPI TTS server
│       ├── vad.py              # Voice activity detection (Silero VAD)
│       ├── wake_word.py        # Wake phrase detection via Whisper
│       └── web_ui.py           # Real-time web dashboard (WebSocket)
├── pyproject.toml              # Project config and dependencies
└── Makefile                    # Dev targets
```

## Tips

- **Voice cloning** works best with a clear 10-30 second audio sample with minimal background noise
- **Apple Silicon** users benefit from MLX acceleration via the 8-bit quantized Chatterbox model
- The TTS server can run on a **separate machine** on your network — just point `--tts-url` to it
- Use smaller Whisper models (`tiny.en`) for faster transcription if accuracy is sufficient

## Troubleshooting

**Dependencies won't install** — Use `uv sync` or `pip install -e .` instead of `pip install -r requirements.txt`. The requirements file contains pinned versions that may not work across systems.

**Microphone not detected** — Check system audio permissions and ensure `sounddevice` can see your input device (`python -c "import sounddevice; print(sounddevice.query_devices())"`)

**LLM not responding** — Verify Ollama/LM Studio is running and the model is loaded. Check the `base_url` in `src/local_talking_llm/assistant.py` matches your setup.

**Slow TTS** — The first generation is slow due to model loading and MLX kernel compilation. Subsequent calls are much faster.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

This repository was forked from [vndee/local-talking-llm](https://github.com/vndee/local-talking-llm) by [Duy Huynh](https://github.com/vndee).

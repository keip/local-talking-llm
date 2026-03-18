import argparse

from rich.console import Console

import web_ui
from assistant import VoiceAssistant


def main():
    parser = argparse.ArgumentParser(description="Local Voice Assistant with ChatterBox TTS")
    parser.add_argument("--model", type=str, default="gemma3", help="LLM model to use")
    parser.add_argument("--tts-url", type=str, default="http://192.168.0.226:8000", help="TTS server URL")
    parser.add_argument(
        "--mode", type=str, default="manual", choices=["manual", "always-on"],
        help="Interaction mode: 'manual' (press Enter) or 'always-on' (wake word)",
    )
    parser.add_argument("--wake-phrase", type=str, default="hey morgan", help="Wake phrase to activate the assistant")
    parser.add_argument("--silence-timeout", type=float, default=1.5, help="Seconds of silence for end-of-speech")
    parser.add_argument("--idle-timeout", type=float, default=8.0, help="Seconds of silence after response before returning to wake phrase mode")
    parser.add_argument("--ui-port", type=int, default=8080, help="Port for the web UI server")
    parser.add_argument("--tools-config", type=str, default="tools.yaml", help="Path to tools allowlist YAML config")
    parser.add_argument("--tool-log", type=str, default="tool_calls.log", help="Path to tool call log file")
    parser.add_argument("--max-tool-depth", type=int, default=5, help="Maximum number of chained tool calls per response")
    args = parser.parse_args()

    console = Console()
    console.print("[cyan]Local Voice Assistant with ChatterBox TTS")
    console.print("[cyan]" + "\u2501" * 42)
    console.print(f"[blue]Mode: {args.mode}")
    console.print(f"[blue]LLM model: {args.model}")
    console.print(f"[blue]TTS server: {args.tts_url}")
    console.print(f"[blue]Web UI: http://localhost:{args.ui_port}")
    console.print("[cyan]" + "\u2501" * 42)
    console.print("[cyan]Press Ctrl+C to exit.\n")

    web_ui.start_server(port=args.ui_port)
    web_ui.emit({"type": "info", "model": args.model, "mode": args.mode, "tts_url": args.tts_url})

    assistant = VoiceAssistant(args)

    try:
        if args.mode == "always-on":
            assistant.run_always_on_mode()
        else:
            assistant.run_manual_mode()
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")


if __name__ == "__main__":
    main()

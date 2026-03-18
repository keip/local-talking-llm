import fnmatch
import shlex
import subprocess
from pathlib import Path

import yaml

from . import Tool


class SystemCommandTool(Tool):
    name = "run_command"
    description = "Run an allowed system command. Only pre-approved commands will execute."
    parameters = {"command": "str"}

    def __init__(self, config_path: str | Path):
        self._allowed: list[dict] = []
        path = Path(config_path)
        if path.exists():
            with path.open() as f:
                data = yaml.safe_load(f) or {}
            self._allowed = data.get("allowed_commands", [])

    def _is_allowed(self, command: str) -> bool:
        for entry in self._allowed:
            pattern = entry.get("pattern", "")
            if fnmatch.fnmatch(command, pattern):
                return True
        return False

    def execute(self, *, command: str, **_kwargs) -> str:
        if not self._is_allowed(command):
            return f"Command not allowed: {command!r}. Only pre-approved commands may run."

        try:
            args = shlex.split(command)
        except ValueError as exc:
            return f"Invalid command syntax: {exc}"

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return "Command timed out after 10 seconds."
        except FileNotFoundError:
            return f"Command not found: {args[0]!r}"

        output = result.stdout.strip()
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return f"Command failed (exit {result.returncode}): {stderr or output}"
        return output or "(no output)"

import json
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path


class Tool(ABC):
    """Base class for all tools the voice assistant can invoke."""

    name: str
    description: str
    parameters: dict  # JSON-schema style description of args

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""


class ToolRegistry:
    """Manages available tools and handles parsing/validation of tool calls."""

    TOOL_CALL_PATTERN = re.compile(
        r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]", re.DOTALL
    )

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def build_system_prompt_section(self) -> str:
        """Generate the tool-instruction block for the LLM system prompt."""
        if not self._tools:
            return ""

        lines = [
            "IMPORTANT: You have access to tools. You MUST use them when the user asks for real-time information "
            "(weather, news, current events, searches) or asks you to perform an action.",
            "",
            "To use a tool, your ENTIRE response must be ONLY the tool call block, nothing else:",
            "",
            '[TOOL_CALL]{"tool": "tool_name", "args": {"param": "value"}}[/TOOL_CALL]',
            "",
            "Available tools:",
        ]
        for tool in self._tools.values():
            params = ", ".join(
                f"{k}: {v}" for k, v in tool.parameters.items()
            )
            lines.append(f"- {tool.name}({params}): {tool.description}")

        lines.extend([
            "",
            "RULES:",
            "- If the user asks about weather, news, current events, or anything you don't know: use web_search.",
            "- If the user asks to run a command or open an app: use run_command.",
            "- Do NOT make up answers for real-time information. Always use the tool first.",
            "- After you use a tool, you will receive the result in a [TOOL_RESULT] block. "
            "Then respond naturally using that information.",
            "- You may chain multiple tool calls across responses.",
        ])
        return "\n".join(lines)

    def parse_tool_call(self, response: str) -> tuple[str, dict] | None:
        """Extract the first [TOOL_CALL]...[/TOOL_CALL] JSON from a response.

        Returns (tool_name, args_dict) or None if no tool call found.
        """
        match = self.TOOL_CALL_PATTERN.search(response)
        if not match:
            return None
        try:
            payload = json.loads(match.group(1).strip())
            return payload["tool"], payload.get("args", {})
        except (json.JSONDecodeError, KeyError):
            return None

    def strip_tool_calls(self, response: str) -> str:
        """Remove all [TOOL_CALL]...[/TOOL_CALL] blocks from text."""
        return self.TOOL_CALL_PATTERN.sub("", response).strip()


class ToolLogger:
    """Appends JSON-lines entries for every tool invocation."""

    def __init__(self, log_path: str | Path):
        self._path = Path(log_path)

    def log(
        self,
        tool: str,
        args: dict,
        result: str,
        duration_ms: int,
        success: bool,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "args": args,
            "result_summary": result[:200],
            "duration_ms": duration_ms,
            "success": success,
        }
        with self._path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

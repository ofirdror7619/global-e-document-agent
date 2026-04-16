from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI


class LLMClient(Protocol):
    def create_completion(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ...


@dataclass
class OpenAIChatLLM:
    model: str = "gpt-4.1-mini"

    def __post_init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY") or self._read_api_key_from_config()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. You can also set src/config/config.json with openai-key."
            )
        self._client = OpenAI(api_key=api_key)

    @staticmethod
    def _read_api_key_from_config() -> str | None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "config.json"
        if not config_path.exists():
            return None
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        for key_name in ("openai-key", "openAI-key", "openai_key", "OPENAI_API_KEY"):
            value = data.get(key_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def create_completion(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=0.1,
        )
        message = response.choices[0].message
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in (message.tool_calls or [])
            ],
        }

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests


class LLMClient(Protocol):
    def create_completion(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ...


class ProviderError(Exception):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderAuthError(ProviderError):
    pass


class ProviderConnectionError(ProviderError):
    pass


class ProviderAPIError(ProviderError):
    pass


@dataclass
class GeminiChatLLM:
    model: str = "gemini-2.5-flash-lite"
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"

    def __post_init__(self) -> None:
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or self._read_api_key_from_config()
        )
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. You can also set src/config/config.json with gemini-key."
            )
        self._api_key = api_key
        self._session = requests.Session()

    @staticmethod
    def _read_api_key_from_config() -> str | None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "config.json"
        if not config_path.exists():
            return None
        try:
            # Handle UTF-8 files with or without BOM.
            data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None
        for key_name in (
            "gemini-key",
            "GEMINI_API_KEY",
            "google-api-key",
            "google_ai_studio_key",
        ):
            value = data.get(key_name)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped and not stripped.upper().startswith("YOUR_"):
                    return stripped
        return None

    def create_completion(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        system_instruction, contents = self._messages_to_gemini(messages)
        declarations = self._tools_to_gemini(tools)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": 0.1},
        }
        if system_instruction:
            payload["system_instruction"] = {"parts": [{"text": system_instruction}]}
        if declarations:
            payload["tools"] = [{"function_declarations": declarations}]

        url = f"{self.api_base}/models/{self.model}:generateContent"
        try:
            response = self._session.post(
                url,
                params={"key": self._api_key},
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise ProviderConnectionError(f"Failed calling Gemini API: {exc}") from exc

        if response.status_code == 429:
            raise ProviderRateLimitError(response.text)
        if response.status_code in {401, 403}:
            raise ProviderAuthError(response.text)
        if response.status_code >= 400:
            raise ProviderAPIError(f"Gemini API error {response.status_code}: {response.text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderAPIError("Gemini API returned invalid JSON") from exc

        if data.get("error"):
            err = data["error"]
            code = err.get("code")
            msg = err.get("message", "Unknown Gemini error")
            if code == 429:
                raise ProviderRateLimitError(msg)
            if code in {401, 403}:
                raise ProviderAuthError(msg)
            raise ProviderAPIError(msg)

        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderAPIError("No Gemini candidates returned")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for idx, part in enumerate(parts, start=1):
            text = part.get("text")
            if text:
                text_parts.append(text)
            fn_call = part.get("functionCall")
            if fn_call:
                name = fn_call.get("name", "")
                args = fn_call.get("args", {})
                tool_calls.append(
                    {
                        "id": f"call_{idx}_{name or 'tool'}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args),
                        },
                    }
                )

        return {
            "role": "assistant",
            "content": "\n".join(p.strip() for p in text_parts if p.strip()),
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def sanitize_schema(node: Any) -> Any:
            if not isinstance(node, dict):
                return node
            allowed_keys = {
                "type",
                "format",
                "description",
                "nullable",
                "enum",
                "properties",
                "required",
                "items",
            }
            cleaned: dict[str, Any] = {}
            for key, value in node.items():
                if key not in allowed_keys:
                    continue
                if key == "properties" and isinstance(value, dict):
                    cleaned[key] = {k: sanitize_schema(v) for k, v in value.items()}
                elif key == "items":
                    cleaned[key] = sanitize_schema(value)
                else:
                    cleaned[key] = value
            return cleaned

        out: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            raw_params = function.get("parameters", {"type": "object", "properties": {}})
            parameters = sanitize_schema(raw_params)
            if not parameters:
                parameters = {"type": "object", "properties": {}}
            out.append(
                {
                    "name": name,
                    "description": function.get("description", ""),
                    "parameters": parameters,
                }
            )
        return out

    @staticmethod
    def _messages_to_gemini(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []
        tool_call_name_by_id: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")
            content = str(msg.get("content", "") or "")

            if role == "system":
                if content.strip():
                    system_parts.append(content.strip())
                continue

            if role == "user":
                if content.strip():
                    contents.append({"role": "user", "parts": [{"text": content}]})
                continue

            if role == "assistant":
                parts: list[dict[str, Any]] = []
                if content.strip():
                    parts.append({"text": content})
                for call in msg.get("tool_calls") or []:
                    function = call.get("function") or {}
                    name = function.get("name", "")
                    args_raw = function.get("arguments") or "{}"
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                    call_id = call.get("id")
                    if call_id and name:
                        tool_call_name_by_id[call_id] = name
                    parts.append({"functionCall": {"name": name, "args": args}})
                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue

            if role == "tool":
                call_id = msg.get("tool_call_id")
                name = tool_call_name_by_id.get(call_id, "tool_response")
                try:
                    response_payload = json.loads(content) if content else {}
                except json.JSONDecodeError:
                    response_payload = {"output": content}
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": name,
                                    "response": response_payload,
                                }
                            }
                        ],
                    }
                )

        system_instruction = "\n\n".join(system_parts).strip()
        return system_instruction, contents

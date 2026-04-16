import json
from pathlib import Path
from typing import Any

from document_agent.agent import DocumentAgent


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0

    def create_completion(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        _ = (messages, tools)
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_agent_calls_tool_then_returns_final_answer() -> None:
    llm = ScriptedLLM(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_config_value",
                            "arguments": json.dumps({"path": "database.primary.pool.max"}),
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The primary DB pool max is 50.\nConfidence: high",
                "tool_calls": [],
            },
        ]
    )

    agent = DocumentAgent(llm=llm, documents_dir=Path("documents"), max_steps=4)
    result = agent.ask("What is the DB pool max?")

    assert "50" in result.answer
    assert any(step.action == "tool:read_config_value" for step in result.trace)
    assert any(step.action == "final_answer" for step in result.trace)


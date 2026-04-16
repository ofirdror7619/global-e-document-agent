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


def test_agent_uses_local_tools_then_single_synthesis_call() -> None:
    llm = ScriptedLLM(
        responses=[
            {
                "role": "assistant",
                "content": "The primary DB pool max is 50.\nConfidence: high",
            },
        ]
    )

    agent = DocumentAgent(llm=llm, documents_dir=Path("documents"), max_steps=4)
    result = agent.ask("What is the DB pool max?")

    assert "50" in result.answer
    assert llm.calls == 1
    assert any(step.action == "tool:search_documents" for step in result.trace)
    assert any(step.action == "tool:read_document" for step in result.trace)
    assert any(step.action == "final_answer" for step in result.trace)


def test_agent_fallback_without_llm_content() -> None:
    llm = ScriptedLLM(
        responses=[
            {"role": "assistant", "content": ""},
        ]
    )
    agent = DocumentAgent(llm=llm, documents_dir=Path("documents"), max_steps=2)
    result = agent.ask("Are there config issues?")
    assert "deterministic tools only" in result.answer
    assert any(step.action == "final_answer" for step in result.trace)

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


def test_cross_document_question_calls_tools_in_order(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "emails.txt").write_text("Q1 sales mention: 12000 USD", encoding="utf-8")
    (docs_dir / "sales-q1.csv").write_text(
        "date,region,product,units_sold,unit_price,currency,sales_rep,status\n"
        "2026-03-01,North,Widget,10,100,USD,Sarah,completed\n",
        encoding="utf-8",
    )
    (docs_dir / "notes.md").write_text("General notes", encoding="utf-8")

    llm = ScriptedLLM(responses=[{"role": "assistant", "content": "Cross-doc synthesis"}])
    agent = DocumentAgent(llm=llm, documents_dir=docs_dir, max_steps=4)

    original_execute = agent.tools.execute
    calls: list[tuple[str, dict[str, Any]]] = []

    def wrapped_execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, dict(arguments)))
        return original_execute(name, arguments)

    agent.tools.execute = wrapped_execute  # type: ignore[method-assign]
    result = agent.ask("Compare Q1 sales mentioned in emails with CSV")

    assert result.answer == "Cross-doc synthesis"
    assert llm.calls == 1
    assert len(calls) >= 3
    assert calls[0][0] == "list_documents"
    assert calls[1][0] == "search_documents"
    assert any(name == "read_document" and args.get("filename") == "emails.txt" for name, args in calls)
    assert any(
        (name == "read_document" and args.get("filename") == "sales-q1.csv")
        or (name == "analyze_sales_csv" and args.get("filename") == "sales-q1.csv")
        for name, args in calls
    )

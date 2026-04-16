from pathlib import Path

from fastapi.testclient import TestClient

import document_agent.server as server
from document_agent.agent import AgentResult, AgentTraceStep
from document_agent.llm import ProviderRateLimitError


def test_session_upload_and_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "SESSIONS_ROOT", tmp_path)
    monkeypatch.setattr(server, "ASK_CACHE", {})
    client = TestClient(server.app)

    create = client.post("/sessions")
    assert create.status_code == 200
    session_id = create.json()["session_id"]

    up = client.post(
        f"/sessions/{session_id}/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert up.status_code == 200
    assert up.json()["uploaded"] == "notes.txt"

    listed = client.get(f"/sessions/{session_id}/files")
    assert listed.status_code == 200
    assert listed.json()["files"] == ["notes.txt"]


def test_ask_endpoint_with_mocked_agent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "SESSIONS_ROOT", tmp_path)
    monkeypatch.setattr(server, "ASK_CACHE", {})

    class FakeLLM:
        def __init__(self, model: str) -> None:
            _ = model

    class FakeAgent:
        def __init__(self, llm, documents_dir, max_steps=4) -> None:
            _ = (llm, documents_dir, max_steps)

        def ask(self, question: str) -> AgentResult:
            _ = question
            return AgentResult(
                answer="mock answer",
                trace=[AgentTraceStep(step=1, action="final_answer", detail="mock")],
            )

    monkeypatch.setattr(server, "GeminiChatLLM", FakeLLM)
    monkeypatch.setattr(server, "DocumentAgent", FakeAgent)
    client = TestClient(server.app)

    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/upload",
        files={"file": ("config.json", b'{\"x\":1}', "application/json")},
    )

    res = client.post("/ask", json={"session_id": session_id, "question": "hi"})
    assert res.status_code == 200
    assert res.json()["answer"] == "mock answer"


def test_cors_preflight_allows_localhost_3000() -> None:
    server.ASK_CACHE.clear()
    client = TestClient(server.app)
    res = client.options(
        "/ask",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_ask_endpoint_surfaces_agent_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "SESSIONS_ROOT", tmp_path)
    monkeypatch.setattr(server, "ASK_CACHE", {})

    class FakeLLM:
        def __init__(self, model: str) -> None:
            _ = model

    class BrokenAgent:
        def __init__(self, llm, documents_dir, max_steps=4) -> None:
            _ = (llm, documents_dir, max_steps)

        def ask(self, question: str):
            _ = question
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "GeminiChatLLM", FakeLLM)
    monkeypatch.setattr(server, "DocumentAgent", BrokenAgent)
    client = TestClient(server.app)
    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/upload",
        files={"file": ("config.json", b'{\"x\":1}', "application/json")},
    )
    res = client.post("/ask", json={"session_id": session_id, "question": "hi"})
    assert res.status_code == 500
    assert "Agent execution failed: boom" in res.text


def test_ask_endpoint_rate_limit_returns_local_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "SESSIONS_ROOT", tmp_path)
    monkeypatch.setattr(server, "ASK_CACHE", {})

    class FakeLLM:
        def __init__(self, model: str) -> None:
            _ = model

    class RateLimitedAgent:
        def __init__(self, llm, documents_dir, max_steps=4) -> None:
            _ = (llm, documents_dir, max_steps)

        def ask(self, question: str):
            _ = question
            raise ProviderRateLimitError("quota exceeded")

        def answer_with_local_evidence(self, question: str, reason: str | None = None) -> AgentResult:
            _ = (question, reason)
            return AgentResult(
                answer="local fallback answer",
                trace=[AgentTraceStep(step=1, action="fallback_local_only", detail="fallback")],
            )

    monkeypatch.setattr(server, "GeminiChatLLM", FakeLLM)
    monkeypatch.setattr(server, "DocumentAgent", RateLimitedAgent)
    client = TestClient(server.app)

    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/upload",
        files={"file": ("config.json", b'{"x":1}', "application/json")},
    )
    res = client.post("/ask", json={"session_id": session_id, "question": "hi"})
    assert res.status_code == 200
    assert res.json()["answer"] == "local fallback answer"
    assert res.json()["response_source"] == "fresh"
    assert res.json()["mode"] == "no-llm-fallback"
    assert res.json()["confidence"] == "low"


def test_ask_endpoint_defaults_to_fresh_without_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "SESSIONS_ROOT", tmp_path)
    monkeypatch.setattr(server, "ASK_CACHE", {})

    class FakeLLM:
        def __init__(self, model: str) -> None:
            _ = model

    class CountingAgent:
        calls = 0

        def __init__(self, llm, documents_dir, max_steps=4) -> None:
            _ = (llm, documents_dir, max_steps)

        def ask(self, question: str) -> AgentResult:
            _ = question
            CountingAgent.calls += 1
            return AgentResult(answer="fresh answer", trace=[AgentTraceStep(step=1, action="final_answer", detail="fresh")])

    monkeypatch.setattr(server, "GeminiChatLLM", FakeLLM)
    monkeypatch.setattr(server, "DocumentAgent", CountingAgent)
    client = TestClient(server.app)

    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/upload",
        files={"file": ("config.json", b'{"x":1}', "application/json")},
    )
    first = client.post("/ask", json={"session_id": session_id, "question": "hi"})
    second = client.post("/ask", json={"session_id": session_id, "question": "hi"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["response_source"] == "fresh"
    assert second.json()["response_source"] == "fresh"
    assert first.json()["mode"] == "gemini"
    assert CountingAgent.calls == 2


def test_ask_endpoint_uses_cache_when_requested(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "SESSIONS_ROOT", tmp_path)
    monkeypatch.setattr(server, "ASK_CACHE", {})

    class FakeLLM:
        def __init__(self, model: str) -> None:
            _ = model

    class CountingAgent:
        calls = 0

        def __init__(self, llm, documents_dir, max_steps=4) -> None:
            _ = (llm, documents_dir, max_steps)

        def ask(self, question: str) -> AgentResult:
            _ = question
            CountingAgent.calls += 1
            return AgentResult(
                answer="cached-enabled answer",
                trace=[AgentTraceStep(step=1, action="final_answer", detail="cached-enabled")],
            )

    monkeypatch.setattr(server, "GeminiChatLLM", FakeLLM)
    monkeypatch.setattr(server, "DocumentAgent", CountingAgent)
    client = TestClient(server.app)

    session_id = client.post("/sessions").json()["session_id"]
    client.post(
        f"/sessions/{session_id}/upload",
        files={"file": ("config.json", b'{"x":1}', "application/json")},
    )
    first = client.post("/ask", json={"session_id": session_id, "question": "hi", "use_cache": True})
    second = client.post("/ask", json={"session_id": session_id, "question": "hi", "use_cache": True})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["response_source"] == "fresh"
    assert second.json()["response_source"] == "cached"
    assert second.json()["cached_at"] is not None
    assert second.json()["mode"] == "gemini"
    assert CountingAgent.calls == 1

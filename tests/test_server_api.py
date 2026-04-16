from pathlib import Path

from fastapi.testclient import TestClient

import document_agent.server as server
from document_agent.agent import AgentResult, AgentTraceStep


def test_session_upload_and_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "SESSIONS_ROOT", tmp_path)
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

    class FakeLLM:
        def __init__(self, model: str) -> None:
            _ = model

    class FakeAgent:
        def __init__(self, llm, documents_dir) -> None:
            _ = (llm, documents_dir)

        def ask(self, question: str) -> AgentResult:
            _ = question
            return AgentResult(
                answer="mock answer",
                trace=[AgentTraceStep(step=1, action="final_answer", detail="mock")],
            )

    monkeypatch.setattr(server, "OpenAIChatLLM", FakeLLM)
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

    class FakeLLM:
        def __init__(self, model: str) -> None:
            _ = model

    class BrokenAgent:
        def __init__(self, llm, documents_dir) -> None:
            _ = (llm, documents_dir)

        def ask(self, question: str):
            _ = question
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "OpenAIChatLLM", FakeLLM)
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

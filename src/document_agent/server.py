from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import DocumentAgent
from .llm import (
    GeminiChatLLM,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
)

SESSIONS_ROOT = Path(".agent_sessions")
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
ASK_CACHE: dict[str, AskResponse] = {}


class CreateSessionResponse(BaseModel):
    session_id: str


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    model: str = "gemini-2.5-flash-lite"
    use_cache: bool = False
    force_refresh: bool = False


class AskResponse(BaseModel):
    answer: str
    trace: list[dict[str, str]]
    response_source: Literal["fresh", "cached"]
    generated_at: str
    cached_at: str | None = None
    mode: Literal["gemini", "no-llm-fallback"] = "gemini"
    confidence: Literal["high", "medium", "low"] = "medium"


def _session_dir(session_id: str) -> Path:
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="Invalid session id")
    return (SESSIONS_ROOT / session_id).resolve()


def _normalized_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _session_files_hash(path: Path) -> str:
    parts: list[str] = []
    for file_path in sorted(p for p in path.glob("*") if p.is_file()):
        stat = file_path.stat()
        parts.append(f"{file_path.name}:{int(stat.st_mtime)}:{stat.st_size}")
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


app = FastAPI(title="Document Agent API", version="0.2.0")
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
raw_origins = os.getenv("AGENT_UI_ORIGINS", ",".join(default_origins))
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    session_id = uuid.uuid4().hex[:12]
    path = _session_dir(session_id)
    path.mkdir(parents=True, exist_ok=True)
    return CreateSessionResponse(session_id=session_id)


@app.get("/sessions/{session_id}/files")
def list_files(session_id: str) -> dict[str, list[str]]:
    path = _session_dir(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    files = sorted([p.name for p in path.glob("*") if p.is_file()])
    return {"files": files}


@app.post("/sessions/{session_id}/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)) -> dict[str, str]:
    path = _session_dir(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    target = (path / Path(file.filename).name).resolve()
    if path not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid filename")

    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"uploaded": target.name}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    path = _session_dir(payload.session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    if not any(path.glob("*")):
        raise HTTPException(status_code=400, detail="Upload at least one document first")

    files_hash = _session_files_hash(path)
    cache_key = f"{payload.session_id}|{_normalized_question(payload.question)}|{files_hash}"
    should_use_cache = payload.use_cache and not payload.force_refresh
    if should_use_cache:
        cached = ASK_CACHE.get(cache_key)
        if cached:
            return AskResponse(
                answer=cached.answer,
                trace=cached.trace,
                response_source="cached",
                generated_at=cached.generated_at,
                cached_at=cached.generated_at,
                mode=cached.mode,
                confidence=cached.confidence,
            )

    try:
        llm = GeminiChatLLM(model=payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    agent = DocumentAgent(llm=llm, documents_dir=path, max_steps=4)
    try:
        result = agent.ask(payload.question)
    except ProviderRateLimitError as exc:
        fallback = agent.answer_with_local_evidence(
            payload.question,
            reason=f"Gemini quota/rate limit issue: {exc}",
        )
        trace = [{"step": str(t.step), "action": t.action, "detail": t.detail} for t in fallback.trace]
        generated_at = datetime.now(timezone.utc).isoformat()
        response = AskResponse(
            answer=fallback.answer,
            trace=trace,
            response_source="fresh",
            generated_at=generated_at,
            cached_at=None,
            mode="no-llm-fallback",
            confidence="low",
        )
        ASK_CACHE[cache_key] = response
        return response
    except ProviderAuthError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"Gemini authentication failed: {exc}",
        ) from exc
    except ProviderConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini connection error: {exc}",
        ) from exc
    except ProviderAPIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {exc}",
        ) from exc

    trace = [{"step": str(t.step), "action": t.action, "detail": t.detail} for t in result.trace]
    generated_at = datetime.now(timezone.utc).isoformat()
    response = AskResponse(
        answer=result.answer,
        trace=trace,
        response_source="fresh",
        generated_at=generated_at,
        cached_at=None,
        mode="gemini",
        confidence="medium",
    )
    ASK_CACHE[cache_key] = response
    return response

# Document Agent

A custom Document Agent that answers natural-language questions over uploaded files using a hand-written tool-calling agent loop.

## What This Includes

- Custom agent orchestration loop (no prebuilt agent frameworks)
- FastAPI backend for session creation, file upload, and Q&A
- Next.js frontend for uploading documents and asking questions
- CLI interface for terminal-based usage
- Tests for tools, API endpoints, and loop behavior

## LLM API Used

- Provider: Gemini API (Google AI Studio)
- Default model: `gemini-2.5-flash-lite`

## AI Coding Tools Used

- ChatGPT
- Manual terminal testing/debugging

## Installation

1. Install Python dependencies:

```bash
pip install -e ".[dev]"
```

2. Set Gemini API key using one option:

- Environment variable (recommended):

```bash
# PowerShell
$env:GEMINI_API_KEY="your_key_here"
```

- Config file fallback in `src/config/config.json`:

```json
{
  "gemini-key": "your_key_here"
}
```

3. Install frontend dependencies:

```bash
cd web
npm install
```

## Run

1. Start backend (repo root):

```bash
uvicorn document_agent.server:app --reload --port 8000
```

2. Start frontend (new terminal):

```bash
cd web
npm run dev
```

3. Open:

- UI: `http://localhost:3000`
- Backend health: `http://127.0.0.1:8000/health`

Note: frontend calls `/api/*` and Next.js proxies to backend `http://127.0.0.1:8000`.

## CLI (Optional)

```bash
python -m document_agent.cli --model gemini-2.5-flash-lite --show-trace
```

## Tests

```bash
pytest
```

## Design Decisions

- Agent loop is explicit and app-owned: model decides tools, app executes tools, loop continues until final response.
- Tooling is isolated from transport/UI logic to keep architecture maintainable.
- Session-scoped upload directories (`.agent_sessions/`) prevent cross-user file mixing.
- Structured tools handle messy formats (CSV normalization, log scanning, nested config access).
- Backend returns actionable provider error messages to make debugging easier.

## API Endpoints

- `POST /sessions`
- `POST /sessions/{session_id}/upload`
- `GET /sessions/{session_id}/files`
- `POST /ask`

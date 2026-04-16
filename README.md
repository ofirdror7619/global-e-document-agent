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

- Agent loop is explicit and app-owned: local intent routing runs deterministic tools first, then the model synthesizes the final response.
- Tooling is isolated from transport/UI logic to keep architecture maintainable.
- Session-scoped upload directories (`.agent_sessions/`) prevent cross-user file mixing.
- Structured tools handle messy formats (CSV normalization, log scanning, nested config access).
- Backend returns actionable provider error messages to make debugging easier.
- The agent uses a quota-aware execution strategy: deterministic tools and local parsing are preferred first, and the LLM is used mainly for synthesis.

## API Endpoints

- `POST /sessions`
- `POST /sessions/{session_id}/upload`
- `GET /sessions/{session_id}/files`
- `POST /ask`

## Key flow:

1. Entry point:
  API route calls agent.ask(question) in server.py.
  If Gemini rate-limit happens, API calls agent.answer_with_local_evidence(...) instead in server.py.

2. Main orchestration (ask)
  ask() in agent.py:
  Calls _collect_evidence(question) to run deterministic tools.
  Builds LLM prompt with question + evidence.
  Calls Gemini once (create_completion).
  If Gemini returns empty text, falls back to _build_local_only_answer(...).
  Appends final_answer trace step and returns AgentResult.
  Deterministic tool loop (_collect_evidence)
  This is the actual “agent loop”: agent.py
  Order is fixed and bounded:

3. list_documents (always first).
  Compute budget: min(max_steps, _tool_budget(question)).
  If budget > 0, run search_documents (always second, phase 1).
  Extract hit filenames from search via _extract_search_hit_filenames(...).
  Build follow-up plan with _build_local_plan(question, docs, hit_filenames).
  Execute only up to remaining budget (phase 2).
  Record every tool call into both:
  trace (human/debug steps),
  evidence (structured payload for synthesis).
  Budget logic
  _tool_budget() in agent.py:
    3 for config+log/error mixed questions.
    4 for complex comparison-style questions.
    otherwise 2.

4. Follow-up planner
  _build_local_plan(...) in agent.py:
  Scores uploaded files by:
  explicit mention in question,
  filename token overlap,
  search-hit boost,
  type hints (.json, .csv, log-like names).
  Chooses top files to read_document.
  Optionally adds:
  analyze_sales_csv for csv/sales intent,
  scan_server_log for log/error intent.
  Deduplicates tool+args.
  Fallback local answer path
  answer_with_local_evidence() in agent.py:
  Reuses same _collect_evidence loop.
  Builds deterministic response with:
  concise Answer: from _build_local_compact_answer,
  Cost Efficiency Reasoning evidence block.

5. Returns without Gemini synthesis.


## Helpers:
1) Helper parsers

- _clean_region(...) normalizes region casing/spacing.
- _to_float(...) parses money-like strings ($, ,) to float or None.
- _to_int(...) parses numeric strings to int or None.

2) Safety layer

- ToolContext.safe_path(...) prevents path traversal and missing-file access.
  It ensures every requested file stays inside documents_dir.

3) Tool registry + dispatcher

- DocumentTools.__init__ defines tool_schemas (name, description, JSON params).
  tools.py
   execute(name, arguments) maps tool name -> function and wraps output consistently:
   success: {"ok": True, "result": ...}
   failure: {"ok": False, "error": "..."}

4. Actual tools

- list_documents() returns available filenames.
- read_document(filename, start_line, end_line) reads full file or line slice with metadata (line_count, line_range, content).
- search_documents(query, filename, max_hits):
  first tries exact substring match,
  if none, falls back to token-overlap scoring (matched_terms, match_score).
- analyze_sales_csv(...) parses rows, normalizes values, computes revenue metrics by status/    
  region/currency, and records anomalies (missing units, bad price, unknown status, etc.).
- scan_server_log(...) extracts log lines by level/keyword and returns summary counts 
  (total_info/warn/error) + matched lines.
- read_config_value(path, filename) navigates nested JSON keys via dot path (example: database.
  primary.pool.max).


## Tests:

1. test_agent_uses_local_tools_then_single_synthesis_call	
2. test_agent_fallback_without_llm_content
3. test_llm_uses_config_fallback_key	
4. test_session_upload_and_list	
5. test_ask_endpoint_with_mocked_agent
6. test_cors_preflight_allows_localhost_3000	
7. test_ask_endpoint_surfaces_agent_error	
8. test_ask_endpoint_rate_limit_surfaces_429	
9. test_ask_endpoint_defaults_to_fresh_without_cache	
10. test_ask_endpoint_uses_cache_when_requested	
11. test_analyze_sales_csv_detects_expected_anomalies	
12. test_read_config_value_nested_path	
13. test_search_documents_falls_back_to_token_overlap
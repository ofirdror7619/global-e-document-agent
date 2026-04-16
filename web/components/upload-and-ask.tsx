"use client";

import { useMemo, useState } from "react";

type AskResponse = {
  answer: string;
  trace?: Array<{ step: string; action: string; detail: string }>;
  response_source?: "fresh" | "cached";
  generated_at?: string;
  cached_at?: string | null;
  mode?: "gemini" | "no-llm-fallback";
  confidence?: "high" | "medium" | "low";
};

type ParsedAnswer = {
  answer: string;
};

const API_BASE = process.env.NEXT_PUBLIC_AGENT_API_URL ?? "/api";

function parseAnswerSections(rawAnswer: string): ParsedAnswer {
  const raw = rawAnswer.trim();
  if (!raw) {
    return { answer: "" };
  }

  const reasoningMarker = "Cost Efficiency Reasoning:";
  const fallbackMarker = "Evidence summary:";
  const markerIndex = raw.includes(reasoningMarker)
    ? raw.indexOf(reasoningMarker)
    : raw.indexOf(fallbackMarker);
  if (markerIndex === -1) {
    return { answer: raw };
  }

  const primary = raw.slice(0, markerIndex).trim();
  const cleanedPrimary = primary
    .replace(/^Answer:\s*/m, "")
    .replace(/^Draft answer generated from deterministic tools only\.\s*/m, "")
    .replace(/^Question:\s*.*$/m, "")
    .trim();

  const isQuotaFallback =
    raw.includes("LLM synthesis was skipped due to quota or provider limitations.") ||
    raw.includes("fallback_local_only");

  const answer =
    cleanedPrimary ||
    (isQuotaFallback
      ? "Gemini synthesis was unavailable for this request."
      : "No synthesized answer returned.");

  return { answer };
}

export function UploadAndAsk() {
  const [sessionId, setSessionId] = useState<string>("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [uploaded, setUploaded] = useState<string[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [answerSource, setAnswerSource] = useState<"fresh" | "cached" | "">("");
  const [cachedAt, setCachedAt] = useState<string>("");
  const [mode, setMode] = useState<"gemini" | "no-llm-fallback" | "">("");
  const [confidence, setConfidence] = useState<"high" | "medium" | "low" | "">("");
  const [traceSteps, setTraceSteps] = useState<Array<{ step: string; action: string; detail: string }>>([]);
  const [useCache, setUseCache] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const parsedAnswer = useMemo(() => parseAnswerSections(answer), [answer]);

  const canUpload = useMemo(() => !!sessionId && !!files?.length, [sessionId, files]);
  const canAsk = useMemo(() => !!sessionId && !!question.trim(), [sessionId, question]);

  async function createSession() {
    setError("");
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/sessions`, { method: "POST" });
      if (!res.ok) throw new Error(`Failed to create session: ${res.status}`);
      const data = (await res.json()) as { session_id: string };
      setSessionId(data.session_id);
      setUploaded([]);
      setAnswer("");
      setAnswerSource("");
      setCachedAt("");
      setMode("");
      setConfidence("");
      setTraceSteps([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setBusy(false);
    }
  }

  async function uploadFiles() {
    if (!sessionId || !files?.length) return;
    setError("");
    setBusy(true);
    try {
      const names: string[] = [];
      for (const file of Array.from(files)) {
        const form = new FormData();
        form.append("file", file);
        const res = await fetch(`${API_BASE}/sessions/${sessionId}/upload`, {
          method: "POST",
          body: form,
        });
        if (!res.ok) throw new Error(`Failed upload (${file.name}): ${res.status}`);
        const data = (await res.json()) as { uploaded: string };
        names.push(data.uploaded);
      }
      setUploaded((prev) => [...prev, ...names]);
      setFiles(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setBusy(false);
    }
  }

  async function askQuestion() {
    if (!sessionId || !question.trim()) return;
    setError("");
    setAnswer("");
    setAnswerSource("");
    setCachedAt("");
    setMode("");
    setConfidence("");
    setTraceSteps([]);
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          question,
          use_cache: useCache,
          force_refresh: !useCache,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Ask failed (${res.status}): ${text}`);
      }
      const data = (await res.json()) as AskResponse;
      setAnswer(data.answer);
      const source = data.response_source === "cached" ? "cached" : "fresh";
      setAnswerSource(source);
      setCachedAt(source === "cached" ? data.cached_at ?? data.generated_at ?? "" : "");
      setMode(data.mode ?? "gemini");
      setConfidence(data.confidence ?? "medium");
      setTraceSteps(data.trace ?? []);
    } catch (e) {
      setAnswer("");
      setAnswerSource("");
      setCachedAt("");
      setMode("");
      setConfidence("");
      setTraceSteps([]);
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setBusy(false);
    }
  }

  function formatCachedAt(value: string): string {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
  }

  function summarizeDetail(detail: string): string {
    const compact = detail.replace(/\s+/g, " ").trim();
    return compact.length > 120 ? `${compact.slice(0, 120)}...` : compact;
  }

  function summarizeTraceStep(action: string, detail: string): string {
    let parsed: any = null;
    try {
      parsed = JSON.parse(detail);
    } catch {
      return summarizeDetail(detail);
    }

    if (!action.startsWith("tool:") || !parsed || typeof parsed !== "object") {
      return summarizeDetail(detail);
    }

    const tool = action.replace("tool:", "");
    const ok = Boolean(parsed.ok);
    const result = parsed.result ?? {};

    if (!ok) {
      return `Failed: ${String(parsed.error ?? "unknown error")}`;
    }

    if (tool === "list_documents") {
      const docs = Array.isArray(result.documents) ? result.documents : [];
      return `Found ${docs.length} document(s).`;
    }
    if (tool === "search_documents") {
      const hits = Array.isArray(result.hits) ? result.hits.length : 0;
      const mode = result.match_mode ?? "unknown";
      return `Search completed with ${hits} hit(s) using ${mode}.`;
    }
    if (tool === "read_document") {
      const filename = result.filename ?? "document";
      const lineCount = result.line_count ?? "?";
      return `Read ${filename} (${lineCount} lines).`;
    }
    if (tool === "analyze_sales_csv") {
      const rows = result.row_count ?? "?";
      const revenue = result.total_revenue_usd ?? "?";
      return `Analyzed CSV (${rows} rows), total revenue: ${revenue} USD.`;
    }
    if (tool === "scan_server_log") {
      const matches = Array.isArray(result.matches) ? result.matches.length : 0;
      const summary = result.summary ?? {};
      return `Scanned logs: ${matches} match(es), errors=${summary.total_error ?? 0}, warnings=${summary.total_warn ?? 0}.`;
    }

    return summarizeDetail(detail);
  }

  return (
    <>
      <section className="card">
        <div className="card-head">
          <h3>1. Session</h3>
          <span className={`status-chip ${sessionId ? "status-ready" : "status-empty"}`}>
            {sessionId ? "Ready" : "Waiting"}
          </span>
        </div>
        <div className="row">
          <button onClick={createSession} disabled={busy}>Create Session</button>
          <div className="muted">{sessionId ? `Session: ${sessionId}` : "No session yet"}</div>
        </div>
      </section>

      <section className="card">
        <h3>2. Upload Files</h3>
        <input type="file" multiple onChange={(e) => setFiles(e.target.files)} />
        <div className="stack-row">
          <button onClick={uploadFiles} disabled={!canUpload || busy}>Upload</button>
        </div>
        <div className="stack-row">
          {uploaded.map((name, idx) => (
            <span key={`${name}-${idx}`} className="badge">{name}</span>
          ))}
        </div>
      </section>

      <section className="card">
        <h3>3. Ask</h3>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Example: Did anyone mention Q1 sales in emails, and how does it compare to CSV?"
        />
        <div className="stack-row cache-toggle-row">
          <label className="cache-toggle-label">
            <input
              type="checkbox"
              checked={useCache}
              onChange={(e) => setUseCache(e.target.checked)}
            />
            Use cache
          </label>
        </div>
        <div className="stack-row">
          <button onClick={askQuestion} disabled={!canAsk || busy}>
            {busy && <span className="swirl-icon swirl-busy" aria-hidden />}
            Ask Agent
          </button>
        </div>
      </section>

      {!!error && (
        <section className="card error-card">
          <strong>Error</strong>
          <div className="muted">{error}</div>
        </section>
      )}

      {!!answer && (
        <section className="card answer-card">
          <div className="card-head">
            <h3>Answer</h3>
            {answerSource === "fresh" && <span className="status-chip source-chip source-fresh">Fresh</span>}
            {answerSource === "cached" && (
              <span className="status-chip source-chip source-cached">
                Cached{cachedAt ? ` (${formatCachedAt(cachedAt)})` : ""}
              </span>
            )}
          </div>
          <div className="stack-row answer-meta-row">
            <span className={`status-chip mode-chip ${mode === "no-llm-fallback" ? "mode-fallback" : "mode-gemini"}`}>
              {mode === "no-llm-fallback" ? "Mode: no-llm-fallback" : "Mode: gemini"}
            </span>
            {!!confidence && (
              <span className="status-chip confidence-chip">Confidence: {confidence}</span>
            )}
          </div>
          <div className="answer">{parsedAnswer.answer}</div>
        </section>
      )}

      {!!traceSteps.length && (
        <section className="card">
          <h3>Agent Steps</h3>
          <div className="stack-row trace-list">
            {traceSteps.map((step) => (
              <div key={`${step.step}-${step.action}`} className="trace-item">
                <div className="trace-title">
                  {step.step}. {step.action}
                </div>
                <div className="trace-detail">{summarizeTraceStep(step.action, step.detail)}</div>
              </div>
            ))}
          </div>
        </section>
      )}

    </>
  );
}

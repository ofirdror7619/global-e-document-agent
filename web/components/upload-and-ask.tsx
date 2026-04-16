"use client";

import { useMemo, useState } from "react";

type AskResponse = {
  answer: string;
  trace: Array<{ step: string; action: string; detail: string }>;
};

const API_BASE = process.env.NEXT_PUBLIC_AGENT_API_URL ?? "/api";

export function UploadAndAsk() {
  const [sessionId, setSessionId] = useState<string>("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [uploaded, setUploaded] = useState<string[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [trace, setTrace] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

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
      setTrace("");
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
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Ask failed (${res.status}): ${text}`);
      }
      const data = (await res.json()) as AskResponse;
      setAnswer(data.answer);
      setTrace(data.trace.map((t) => `[${t.step}] ${t.action}\n${t.detail}`).join("\n\n"));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="card">
        <h3>1. Session</h3>
        <div className="row">
          <button onClick={createSession} disabled={busy}>Create Session</button>
          <div className="muted">{sessionId ? `Session: ${sessionId}` : "No session yet"}</div>
        </div>
      </section>

      <section className="card">
        <h3>2. Upload Files</h3>
        <input type="file" multiple onChange={(e) => setFiles(e.target.files)} />
        <div style={{ marginTop: 10 }}>
          <button onClick={uploadFiles} disabled={!canUpload || busy}>Upload</button>
        </div>
        <div style={{ marginTop: 8 }}>
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
        <div style={{ marginTop: 10 }}>
          <button onClick={askQuestion} disabled={!canAsk || busy}>Ask Agent</button>
        </div>
      </section>

      {!!error && (
        <section className="card">
          <strong>Error:</strong>
          <div className="muted">{error}</div>
        </section>
      )}

      {!!answer && (
        <section className="card">
          <h3>Answer</h3>
          <div className="answer">{answer}</div>
        </section>
      )}

      {!!trace && (
        <section className="card">
          <h3>Agent Trace</h3>
          <div className="trace">{trace}</div>
        </section>
      )}
    </>
  );
}

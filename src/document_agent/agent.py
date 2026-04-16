from __future__ import annotations

import json
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Callable

from .llm import LLMClient
from .tools import DocumentTools


SYSTEM_PROMPT = """You are a document analysis assistant.
You get a user question and structured evidence from deterministic tools.
Produce:
1) concise answer
2) short evidence bullets grounded in the evidence
3) confidence (high/medium/low)
Do not invent facts. If evidence is insufficient, say exactly what is missing.
"""


@dataclass
class AgentTraceStep:
    step: int
    action: str
    detail: str


@dataclass
class AgentResult:
    answer: str
    trace: list[AgentTraceStep]


ToolPlan = list[tuple[str, dict[str, Any]]]


class DocumentAgent:
    SEARCH_MAX_HITS = 8
    READ_END_LINE = 220
    LOG_SCAN_MAX_LINES = 40
    TOP_READ_CANDIDATES = 3
    EXPLICIT_MENTION_BOOST = 100
    SEARCH_HIT_BOOST = 60
    TERM_OVERLAP_BOOST = 10
    TOPIC_SUFFIX_BOOST = 8

    CONFIG_TERMS = {"config", "setting", "settings", "database", "db", "pool", "json"}
    LOG_TERMS = {"error", "warning", "warn", "log", "exception", "failed"}
    NOTE_TERMS = {"meeting", "minutes", "attendee", "decision", "note"}
    CSV_TERMS = {"csv", "sale", "sales", "revenue", "q1", "report"}

    def __init__(
        self,
        llm: LLMClient,
        documents_dir: str | Path = "documents",
        max_steps: int = 4,
    ) -> None:
        self.llm = llm
        self.tools = DocumentTools(documents_dir=documents_dir)
        self.max_steps = max_steps

    def ask(self, question: str) -> AgentResult:
        trace, evidence = self._collect_evidence(question)

        payload = {
            "question": question,
            "evidence": evidence,
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Question:\n"
                    f"{question}\n\n"
                    "Evidence JSON:\n"
                    f"{json.dumps(payload, ensure_ascii=True)[:12000]}"
                ),
            },
        ]

        assistant = self.llm.create_completion(messages=messages, tools=[])
        content = str(assistant.get("content", "") or "").strip()
        if not content:
            content = self._build_local_only_answer(question, evidence, quota_note=False)

        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                action="final_answer",
                detail=content[:500],
            )
        )
        return AgentResult(answer=content, trace=trace)

    def answer_with_local_evidence(self, question: str, reason: str | None = None) -> AgentResult:
        trace, evidence = self._collect_evidence(question)
        answer = self._build_local_only_answer(question, evidence, quota_note=True)
        if reason:
            answer = f"{answer}\n\nNote: {reason}"
        trace.append(
            AgentTraceStep(
                step=len(trace) + 1,
                action="fallback_local_only",
                detail="LLM unavailable or quota-limited; returned local synthesis",
            )
        )
        return AgentResult(answer=answer, trace=trace)

    def _collect_evidence(self, question: str) -> tuple[list[AgentTraceStep], list[dict[str, Any]]]:
        trace: list[AgentTraceStep] = []
        evidence: list[dict[str, Any]] = []

        docs_res = self.tools.execute("list_documents", {})
        docs = docs_res.get("result", {}).get("documents", []) if docs_res.get("ok") else []
        step = 1
        self._append_tool_step(trace, evidence, step=step, tool_name="list_documents", args=None, result=docs_res)

        budget = min(self.max_steps, self._tool_budget(question))
        step += 1
        if budget <= 0:
            return trace, evidence

        # Phase 1: broad search first.
        search_args = {"query": question, "max_hits": self.SEARCH_MAX_HITS}
        search_result = self.tools.execute("search_documents", search_args)
        self._append_tool_step(
            trace, evidence, step=step, tool_name="search_documents", args=search_args, result=search_result
        )
        step += 1

        remaining = budget - 1
        if remaining <= 0:
            return trace, evidence

        hit_filenames = self._extract_search_hit_filenames(search_result)
        followup_plan = self._build_local_plan(
            question=question,
            documents=docs,
            hit_filenames=hit_filenames,
        )

        # Phase 2: targeted follow-up calls derived from search evidence.
        for name, args in followup_plan[:remaining]:
            tool_result = self.tools.execute(name, args)
            self._append_tool_step(trace, evidence, step=step, tool_name=name, args=args, result=tool_result)
            step += 1

        return trace, evidence

    def _tool_budget(self, question: str) -> int:
        q = question.lower()
        complex_markers = [
            "compare",
            "across",
            "related",
            "mention",
            "mentioned",
            "between",
            "vs",
            "versus",
        ]
        config_markers = ("config", "setting", "settings", "pool", "rate limit")
        log_markers = ("error", "warn", "warning", "log", "exception", "failed")
        if any(m in q for m in config_markers) and any(m in q for m in log_markers):
            return 3
        return 4 if any(marker in q for marker in complex_markers) else 2

    def _build_local_plan(
        self, question: str, documents: list[str], hit_filenames: list[str] | None = None
    ) -> ToolPlan:
        q = question.lower()
        plan: ToolPlan = []

        if not documents:
            return plan

        question_terms = self._extract_terms(q)
        hit_set = {name.lower() for name in (hit_filenames or [])}

        ranked: list[tuple[int, str]] = []
        for file_name in documents:
            score = self._score_file(
                file_name,
                question=q,
                question_terms=question_terms,
                hit_set=hit_set,
            )
            ranked.append((score, file_name))

        ranked.sort(key=lambda item: (-item[0], item[1]))

        selected_reads: list[str] = []
        if question_terms & self.LOG_TERMS:
            log_read = self._pick_file(ranked, lambda name: self._is_log_like(name))
            if log_read:
                selected_reads.append(log_read)

        if question_terms & self.CONFIG_TERMS:
            config_read = self._pick_file(ranked, lambda name: Path(name).suffix.lower() == ".json")
            if config_read and config_read not in selected_reads:
                selected_reads.append(config_read)

        for _, file_name in ranked:
            if file_name in selected_reads:
                continue
            selected_reads.append(file_name)
            if len(selected_reads) >= self.TOP_READ_CANDIDATES:
                break

        # Read a few most relevant files after search; budgeting is applied later.
        for file_name in selected_reads[: self.TOP_READ_CANDIDATES]:
            plan.append(("read_document", {"filename": file_name, "end_line": self.READ_END_LINE}))

        # Run deeper analyzers opportunistically when question intent + file type match.
        if question_terms & self.CSV_TERMS:
            csv_file = self._pick_file(ranked, lambda name: Path(name).suffix.lower() == ".csv")
            if csv_file:
                plan.append(("analyze_sales_csv", {"filename": csv_file}))

        if question_terms & self.LOG_TERMS:
            log_file = self._pick_file(ranked, lambda name: self._is_log_like(name))
            if not log_file:
                log_file = self._pick_file(ranked, lambda name: Path(name).suffix.lower() == ".txt")
            if log_file:
                plan.append(("scan_server_log", {"filename": log_file, "max_lines": self.LOG_SCAN_MAX_LINES}))

        # De-duplicate identical tool+args pairs while preserving order.
        deduped: ToolPlan = []
        seen: set[str] = set()
        for name, args in plan:
            key = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=True)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append((name, args))
        return deduped

    @staticmethod
    def _normalize_term(term: str) -> str:
        return term[:-1] if len(term) > 3 and term.endswith("s") else term

    def _extract_terms(self, text: str) -> set[str]:
        return {
            self._normalize_term(token)
            for token in re.findall(r"[a-z0-9_]+", text.lower())
            if len(token) >= 2
        }

    @staticmethod
    def _is_log_like(filename: str) -> bool:
        suffix = Path(filename).suffix.lower()
        return "log" in filename.lower() or suffix == ".log"

    def _score_file(
        self, file_name: str, question: str, question_terms: set[str], hit_set: set[str]
    ) -> int:
        lower_name = file_name.lower()
        stem = Path(file_name).stem.lower()
        suffix = Path(file_name).suffix.lower()
        name_terms = self._extract_terms(stem)

        score = 0
        if lower_name in question or stem in question:
            score += self.EXPLICIT_MENTION_BOOST
        if lower_name in hit_set:
            score += self.SEARCH_HIT_BOOST

        score += len(name_terms & question_terms) * self.TERM_OVERLAP_BOOST

        if question_terms & self.CONFIG_TERMS and suffix == ".json":
            score += self.TOPIC_SUFFIX_BOOST
        if question_terms & self.LOG_TERMS and (suffix in {".log", ".txt"} or "log" in lower_name):
            score += self.TOPIC_SUFFIX_BOOST
        if question_terms & self.CSV_TERMS and suffix == ".csv":
            score += self.TOPIC_SUFFIX_BOOST
        if question_terms & self.NOTE_TERMS and (suffix in {".md", ".txt"} or "meeting" in lower_name):
            score += self.TOPIC_SUFFIX_BOOST
        return score

    @staticmethod
    def _pick_file(ranked: list[tuple[int, str]], predicate: Callable[[str], bool]) -> str | None:
        return next((name for _, name in ranked if predicate(name)), None)

    @staticmethod
    def _append_tool_step(
        trace: list[AgentTraceStep],
        evidence: list[dict[str, Any]],
        step: int,
        tool_name: str,
        args: dict[str, Any] | None,
        result: dict[str, Any],
    ) -> None:
        trace.append(
            AgentTraceStep(
                step=step,
                action=f"tool:{tool_name}",
                detail=json.dumps(result, ensure_ascii=True)[:500],
            )
        )
        event: dict[str, Any] = {"tool": tool_name, "result": result}
        if args is not None:
            event["args"] = args
        evidence.append(event)

    @staticmethod
    def _extract_search_hit_filenames(search_result: dict[str, Any]) -> list[str]:
        if not isinstance(search_result, dict):
            return []
        hits = search_result.get("result", {}).get("hits", []) if search_result.get("ok") else []
        if not isinstance(hits, list):
            return []
        out: list[str] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            name = hit.get("filename")
            if isinstance(name, str) and name:
                out.append(name)
        return out

    def _build_local_only_answer(
        self, question: str, evidence: list[dict[str, Any]], quota_note: bool
    ) -> str:
        answer = self._build_local_compact_answer(question=question, evidence=evidence)
        lines = [
            f"Answer: {answer}",
            "",
            "Cost Efficiency Reasoning:",
            "Draft answer generated from deterministic tools only.",
            f"Question: {question}",
            "",
            "Evidence summary:",
        ]
        for item in evidence:
            tool = item.get("tool", "unknown")
            result = item.get("result", {})
            rendered = json.dumps(result, ensure_ascii=True)
            lines.append(f"- {tool}: {rendered[:260]}")
        lines.append("")
        lines.append("Confidence: medium")
        if quota_note:
            lines.append("LLM synthesis was skipped due to quota or provider limitations.")
        return "\n".join(lines)

    def _build_local_compact_answer(self, question: str, evidence: list[dict[str, Any]]) -> str:
        q = question.lower()
        hits = self._search_hits(evidence)
        docs = self._read_contents(evidence)

        if "meeting" in q and "who" in q:
            for content in docs:
                match = re.search(r"\*\*Attendees:\*\*\s*(.+)", content)
                if match:
                    attendees = match.group(1).strip().rstrip(".")
                    return f"The meeting attendees were {attendees}."

        if any(k in q for k in ("error", "warning", "warn", "log")) and any(
            k in q for k in ("config", "setting", "settings")
        ):
            findings: list[str] = []
            texts = [str(h.get("text", "")) for h in hits]
            if any("Connection pool exhausted" in t for t in texts):
                findings.append("database connection pool exhaustion errors were logged")
            if any("API_RATE_LIMITING feature flag is disabled" in t for t in texts):
                findings.append("rate limiting is disabled in production config while rate-limit exceedance appears in logs")
            if any("Config mismatch: app.version" in t for t in texts):
                findings.append("a deployment/config version mismatch was logged")
            if findings:
                return "Yes. Potentially related issues found: " + "; ".join(findings) + "."

        if hits:
            top = hits[0]
            src = f"{top.get('filename', 'document')} line {top.get('line', '?')}"
            text = str(top.get("text", "")).strip()
            return f"Most relevant evidence is from {src}: {text}"

        return "I could not find enough matching evidence in the available documents to answer confidently."

    @staticmethod
    def _search_hits(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_hits: list[dict[str, Any]] = []
        for item in evidence:
            if item.get("tool") != "search_documents":
                continue
            result = item.get("result", {})
            hits = result.get("result", {}).get("hits", []) if isinstance(result, dict) else []
            if isinstance(hits, list):
                all_hits.extend(h for h in hits if isinstance(h, dict))
        return all_hits

    @staticmethod
    def _read_contents(evidence: list[dict[str, Any]]) -> list[str]:
        texts: list[str] = []
        for item in evidence:
            if item.get("tool") != "read_document":
                continue
            result = item.get("result", {})
            content = result.get("result", {}).get("content", "") if isinstance(result, dict) else ""
            if isinstance(content, str) and content:
                texts.append(content)
        return texts

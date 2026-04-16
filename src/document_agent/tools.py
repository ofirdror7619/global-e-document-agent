from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _clean_region(region: str) -> str:
    return " ".join(word.capitalize() for word in region.strip().split())


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


@dataclass
class ToolContext:
    documents_dir: Path

    def safe_path(self, relative_name: str) -> Path:
        candidate = (self.documents_dir / relative_name).resolve()
        docs_root = self.documents_dir.resolve()
        if docs_root not in candidate.parents and candidate != docs_root:
            raise ValueError("Path escapes documents directory.")
        if not candidate.exists():
            raise FileNotFoundError(f"Document not found: {relative_name}")
        return candidate


class DocumentTools:
    def __init__(self, documents_dir: str | Path) -> None:
        self.ctx = ToolContext(Path(documents_dir))
        self.tool_schemas: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "list_documents",
                    "description": "List all available documents.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_document",
                    "description": "Read a document by file name. Optional line slicing for large files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                        },
                        "required": ["filename"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search INSIDE document contents for a text query across one or all documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filename": {"type": "string"},
                            "max_hits": {"type": "integer", "minimum": 1, "maximum": 20},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_sales_csv",
                    "description": "Parse a CSV sales file with normalization and return metrics/anomalies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Defaults to sales-q1.csv"},
                            "fx_eur_to_usd": {"type": "number"},
                            "include_pending": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "scan_server_log",
                    "description": "Extract warnings/errors and optionally filter by keyword.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Defaults to server-log.txt"},
                            "keyword": {"type": "string"},
                            "level": {"type": "string", "enum": ["INFO", "WARN", "ERROR"]},
                            "max_lines": {"type": "integer", "minimum": 1, "maximum": 100},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_config_value",
                    "description": "Read a nested value in a JSON config file using dot-separated path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Example: database.primary.pool.max"},
                            "filename": {"type": "string", "description": "Defaults to config.json"},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "list_documents": self.list_documents,
            "read_document": self.read_document,
            "search_documents": self.search_documents,
            "analyze_sales_csv": self.analyze_sales_csv,
            "scan_server_log": self.scan_server_log,
            "read_config_value": self.read_config_value,
        }
        if name not in handlers:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        try:
            return {"ok": True, "result": handlers[name](**arguments)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def list_documents(self) -> dict[str, Any]:
        files = sorted([p.name for p in self.ctx.documents_dir.glob("*") if p.is_file()])
        return {"documents": files}

    def read_document(
        self, filename: str, start_line: int | None = None, end_line: int | None = None
    ) -> dict[str, Any]:
        path = self.ctx.safe_path(filename)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        if start_line or end_line:
            start = max(1, start_line or 1)
            end = min(len(lines), end_line or len(lines))
            if end < start:
                raise ValueError("end_line must be >= start_line")
            sliced = lines[start - 1 : end]
            content = "\n".join(sliced)
            line_range = {"start_line": start, "end_line": end}
        else:
            content = text
            line_range = None

        return {
            "filename": filename,
            "line_count": len(lines),
            "line_range": line_range,
            "content": content,
        }

    def search_documents(
        self, query: str, filename: str | None = None, max_hits: int = 8
    ) -> dict[str, Any]:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        query_terms = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) >= 2]
        targets = [self.ctx.safe_path(filename)] if filename else sorted(self.ctx.documents_dir.glob("*"))
        hits: list[dict[str, Any]] = []
        for path in targets:
            if not path.is_file():
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern.search(line):
                    hits.append({"filename": path.name, "line": line_no, "text": line.strip()})
                    if len(hits) >= max_hits:
                        return {"query": query, "match_mode": "exact_substring", "hits": hits}

        # Fallback: token overlap for semantic-ish matching when exact phrase does not appear.
        scored: list[dict[str, Any]] = []
        if query_terms:
            for path in targets:
                if not path.is_file():
                    continue
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    line_lower = line.lower()
                    matched_terms = [t for t in query_terms if t in line_lower]
                    if not matched_terms:
                        continue
                    scored.append(
                        {
                            "filename": path.name,
                            "line": line_no,
                            "text": line.strip(),
                            "matched_terms": matched_terms,
                            "match_score": len(set(matched_terms)),
                        }
                    )
            scored.sort(key=lambda h: (-int(h["match_score"]), h["filename"], int(h["line"])))

        return {
            "query": query,
            "match_mode": "token_overlap" if scored else "none",
            "hits": scored[:max_hits],
        }

    def analyze_sales_csv(
        self, filename: str = "sales-q1.csv", fx_eur_to_usd: float = 1.08, include_pending: bool = False
    ) -> dict[str, Any]:
        path = self.ctx.safe_path(filename)
        rows: list[dict[str, Any]] = []
        anomalies: list[dict[str, Any]] = []
        by_status: dict[str, float] = {}
        by_region: dict[str, float] = {}
        currencies: dict[str, float] = {}

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=2):
                units = _to_int(row.get("units_sold"))
                price = _to_float(row.get("unit_price"))
                currency = str(row.get("currency", "")).strip().upper()
                status = str(row.get("status", "")).strip().lower()
                region = _clean_region(str(row.get("region", "")))

                if units is None:
                    anomalies.append(
                        {
                            "line": idx,
                            "type": "missing_units_sold",
                            "row": row,
                        }
                    )
                    units = 0

                if price is None:
                    anomalies.append(
                        {
                            "line": idx,
                            "type": "invalid_unit_price",
                            "row": row,
                        }
                    )
                    price = 0.0

                raw_amount = units * price
                usd_amount = raw_amount * fx_eur_to_usd if currency == "EUR" else raw_amount

                if row.get("region", "") != region:
                    anomalies.append(
                        {"line": idx, "type": "inconsistent_region_format", "value": row.get("region", "")}
                    )

                if not row.get("sales_rep"):
                    anomalies.append({"line": idx, "type": "missing_sales_rep", "row": row})

                if status not in {"completed", "pending", "refunded"}:
                    anomalies.append({"line": idx, "type": "unknown_status", "value": status})

                rows.append(
                    {
                        "date": row.get("date"),
                        "region": region,
                        "product": row.get("product"),
                        "units_sold": units,
                        "unit_price": price,
                        "currency": currency,
                        "sales_rep": row.get("sales_rep"),
                        "status": status,
                        "amount_usd": round(usd_amount, 2),
                    }
                )

                by_status[status] = by_status.get(status, 0.0) + usd_amount
                by_region[region] = by_region.get(region, 0.0) + usd_amount
                currencies[currency] = currencies.get(currency, 0.0) + raw_amount

        confirmed = sum(r["amount_usd"] for r in rows if r["status"] == "completed")
        pending = sum(r["amount_usd"] for r in rows if r["status"] == "pending")
        refunded = sum(r["amount_usd"] for r in rows if r["status"] == "refunded")
        total = confirmed + pending if include_pending else confirmed

        return {
            "fx_eur_to_usd": fx_eur_to_usd,
            "row_count": len(rows),
            "confirmed_revenue_usd": round(confirmed, 2),
            "pending_revenue_usd": round(pending, 2),
            "refunded_revenue_usd": round(refunded, 2),
            "total_revenue_usd": round(total, 2),
            "by_status_usd": {k: round(v, 2) for k, v in by_status.items()},
            "by_region_usd": {k: round(v, 2) for k, v in by_region.items()},
            "currency_raw_totals": {k: round(v, 2) for k, v in currencies.items()},
            "anomalies": anomalies,
        }

    def scan_server_log(
        self,
        filename: str = "server-log.txt",
        keyword: str | None = None,
        level: str | None = None,
        max_lines: int = 30,
    ) -> dict[str, Any]:
        path = self.ctx.safe_path(filename)
        lines = path.read_text(encoding="utf-8").splitlines()
        extracted: list[dict[str, Any]] = []

        for line_no, line in enumerate(lines, start=1):
            match = re.search(r"\[(INFO|WARN|ERROR)\]", line)
            line_level = match.group(1) if match else "UNKNOWN"
            if level and line_level != level:
                continue
            if keyword and keyword.lower() not in line.lower():
                continue
            extracted.append({"line": line_no, "level": line_level, "text": line})
            if len(extracted) >= max_lines:
                break

        summary = {
            "total_info": sum(1 for l in lines if "[INFO]" in l),
            "total_warn": sum(1 for l in lines if "[WARN]" in l),
            "total_error": sum(1 for l in lines if "[ERROR]" in l),
        }

        return {"summary": summary, "matches": extracted}

    def read_config_value(self, path: str, filename: str = "config.json") -> dict[str, Any]:
        cfg = json.loads(self.ctx.safe_path(filename).read_text(encoding="utf-8"))
        node: Any = cfg
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise KeyError(f"Path not found: {path}")
        return {"path": path, "value": node}

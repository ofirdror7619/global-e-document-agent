from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .tools import DocumentTools


SYSTEM_PROMPT = """You are a document analysis agent.
You can call tools to inspect files and compute structured insights.
Use tools whenever you need evidence. You may call multiple tools across multiple steps.
Do not conclude "not found" after a single failed exact search. If a search returns no exact hits, try broader queries,
token-level search behavior, or read likely documents directly before concluding absence.
Important:
- `list_documents` only tells you what files exist; it does not answer content questions.
- For content questions, search inside files and/or read file content before answering.
- For analytical questions, use the relevant analysis tools and connect evidence across files.
When ready, provide:
1) a concise answer
2) brief evidence bullets with file names/line hints when possible
3) confidence level (high/medium/low)
Do not invent facts not supported by tool outputs.
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


class DocumentAgent:
    def __init__(
        self,
        llm: LLMClient,
        documents_dir: str | Path = "documents",
        max_steps: int = 8,
    ) -> None:
        self.llm = llm
        self.tools = DocumentTools(documents_dir=documents_dir)
        self.max_steps = max_steps

    def ask(self, question: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list[AgentTraceStep] = []

        for step in range(1, self.max_steps + 1):
            assistant = self.llm.create_completion(messages=messages, tools=self.tools.tool_schemas)
            tool_calls = assistant.get("tool_calls") or []
            content = assistant.get("content", "")

            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            if tool_calls:
                for call in tool_calls:
                    name = call["function"]["name"]
                    args_raw = call["function"]["arguments"] or "{}"
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}

                    tool_result = self.tools.execute(name, args)
                    trace.append(
                        AgentTraceStep(
                            step=step,
                            action=f"tool:{name}",
                            detail=json.dumps(tool_result, ensure_ascii=True)[:500],
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(tool_result),
                        }
                    )
                continue

            if content.strip():
                trace.append(AgentTraceStep(step=step, action="final_answer", detail=content[:500]))
                return AgentResult(answer=content.strip(), trace=trace)

            trace.append(AgentTraceStep(step=step, action="empty_response", detail="Model returned no content"))

        fallback = (
            "I could not finish reasoning within the step budget. "
            "Please narrow the question or ask me to continue."
        )
        return AgentResult(answer=fallback, trace=trace)

# Document Agent

## Overview

This project implements a **Document Agent** — an AI-powered system that allows users to ask natural language questions about a collection of heterogeneous documents (Markdown, CSV, JSON, logs, and plain text).

The agent reads, analyzes, and reasons across documents using a combination of:

* deterministic tools
* structured parsing
* and an LLM as a reasoning and synthesis engine

The focus of this project is not just answering questions, but **designing and implementing an agent loop from scratch**, without relying on pre-built frameworks.

---

## Features

* 🧠 Custom agent loop (no LangChain / frameworks)
* 🛠 Multiple tools (file reading, search, structured queries)
* 📄 Support for multiple formats:

  * Markdown (meetings)
  * CSV (sales data)
  * JSON (config)
  * Logs (server-log)
  * Plain text (emails)
* 🔍 Cross-document reasoning
* ⚡ Quota-aware execution (minimizes LLM calls)
* 🧾 Transparent reasoning trace (tool usage visibility)
* 🛟 Graceful fallback mode (`no-llm-fallback`) when provider quota is exceeded
* 🧪 Unit tests for tools, parsers, and agent loop
* 💻 CLI + API server interface

---

## How It Works

### Agent Loop

The core of the system is a custom agent loop:

1. Receive user question
2. Decide next action (tool call or answer)
3. Execute tool
4. Observe result
5. Repeat until final answer

This loop is implemented manually to demonstrate full control over agent behavior.

---

### Execution Strategy (Important)

To reduce cost and improve reliability, the agent uses a **hybrid approach**:

* Deterministic tools are preferred first
* Structured data is parsed locally (CSV, logs, JSON)
* The LLM is used primarily for:

  * reasoning across documents
  * synthesizing final answers

This makes the system:

* more efficient (fewer API calls)
* more deterministic
* less prone to hallucinations

---

### Tools

The agent can use the following tools:

* `list_documents` — list available files
* `read_document` — read file content
* `search_documents` — keyword search across all documents
* `query_structured_data` — query CSV / JSON / logs

Each tool is implemented as deterministic logic (no LLM dependency).

---

### Reasoning Trace

The agent exposes its reasoning process:

```
Agent Steps:
1. search_documents("Q1 sales")
2. read_document("emails.txt")
3. query_structured_data("sales-q1.csv")
4. synthesized final answer
```

This improves transparency and debuggability.

---

## Installation

```bash
git clone https://github.com/ofirdror7619/global-e-document-agent
cd global-e-document-agent
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Running the Project

### CLI

```bash
python -m src.ui.cli
```

Then ask:

```
> Did anyone mention Q1 sales in the emails?
```

---

### Server

```bash
python -m src.server
```

---

## Testing

```bash
pytest
```

Tests include:

* tool behavior
* parsing logic
* agent loop flow (with mocked LLM)

---

## LLM Provider

This project uses:

👉 **Google Gemini API (Free Tier)**

Reasons:

* no setup cost
* sufficient reasoning capabilities
* large context window

---

## Handling Quotas / Failures

The system is designed to handle API limitations:

* minimizes number of LLM calls
* uses deterministic tools where possible
* includes fallback behavior when quota is exceeded

Example fallback:

* local evidence-based answer without LLM
* marked as lower confidence

---

## Design Decisions

### 1. No Agent Frameworks

The agent loop is implemented manually to:

* comply with assignment requirements
* demonstrate understanding of orchestration

---

### 2. Separation of Concerns

* Agent logic → decision making
* Tools → data access
* Parsers → format-specific logic
* UI → interaction layer

This keeps the system modular and testable.

---

### 3. Deterministic First, LLM Second

Instead of relying heavily on the LLM:

* structured data is parsed locally
* tools return reliable data
* LLM is used for synthesis only

---

### 4. Quota-Aware Design

Given free-tier constraints:

* limited agent steps
* reduced prompt size
* minimized retries

---

### 5. Transparency Over Magic

The agent exposes:

* which tools were used
* what data was retrieved

Instead of hiding reasoning behind a single prompt.

---

## Limitations

* No vector search / embeddings (intentionally avoided for simplicity)
* No long-term memory between sessions
* Heuristic routing (can be improved)

---

## Future Improvements

* Smarter planning (reduce tool calls further)
* Better ambiguity handling
* User feedback loop (corrections)
* Streaming responses
* UI improvements

---

## AI Tools Used

* ChatGPT (design, architecture, debugging)
* AI-assisted coding for faster iteration

---

## Summary

This project demonstrates:

* building an agent from first principles
* integrating LLMs into real systems
* making practical tradeoffs under constraints

The goal was not to build the most complex system, but a **clean, working, and well-reasoned one**.

---

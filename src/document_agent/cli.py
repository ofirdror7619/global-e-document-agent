from __future__ import annotations

import argparse
from pathlib import Path

from .agent import DocumentAgent
from .llm import GeminiChatLLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document Agent CLI")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model name")
    parser.add_argument(
        "--documents-dir",
        default="documents",
        help="Path to documents directory",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Show agent tool trace after each answer",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Maximum local tool steps per question",
    )
    return parser


def run_repl(model: str, documents_dir: str, show_trace: bool, max_steps: int) -> None:
    llm = GeminiChatLLM(model=model)
    agent = DocumentAgent(llm=llm, documents_dir=Path(documents_dir), max_steps=max_steps)

    print("Document Agent ready. Type your question, or 'exit' to quit.")
    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        result = agent.ask(question)
        print("\nAnswer:\n")
        print(result.answer)
        if show_trace:
            print("\nTrace:")
            for item in result.trace:
                print(f"- step {item.step}: {item.action}")
                print(f"  {item.detail}")


def main() -> None:
    args = build_parser().parse_args()
    run_repl(
        model=args.model,
        documents_dir=args.documents_dir,
        show_trace=args.show_trace,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

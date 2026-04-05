import asyncio
import re
import sys

from rag.pipeline import run_pipeline_stream




_MAX_QUERY_CHARS = 2_000
_MAX_QUERY_TOKENS_EST = 500



_ROLE_INJECTION_RE = re.compile(
    r"(?i)^(system\s*:|assistant\s*:|<\|im_start\||<\|system\||ignore (all |previous |above )?(instructions?|rules?|system prompt))",
    re.MULTILINE,
)

_ROLE_LABEL_RE = re.compile(
    r"(?i)\b(system|assistant)\s*:\s*",
)


def _sanitise_input(raw: str) -> tuple[str, list[str]]:
    """
    Sanitise raw user input before passing to the pipeline.

    Returns:
        (sanitised_text, list_of_warnings)

    Applies:
        1. Whitespace stripping
        2. Hard length cap
        3. Role-injection detection and removal
    """
    warnings: list[str] = []
    text = raw.strip()

    if len(text) > _MAX_QUERY_CHARS:
        text = text[:_MAX_QUERY_CHARS]
        warnings.append(
            f"Your query was truncated to {_MAX_QUERY_CHARS} characters "
            f"to prevent context overflow."
        )

    if _ROLE_INJECTION_RE.search(text):
        warnings.append(
            "Possible role-injection pattern detected and removed from your query."
        )
        text = _ROLE_INJECTION_RE.sub("", text).strip()
        text = _ROLE_LABEL_RE.sub("", text).strip()

    if not text:
        text = "[empty query after sanitisation]"

    return text, warnings


async def main() -> None:
    print("Welcome to DRAGON. Start typing to begin.")
    while True:
        try:
            raw = input("\nAsk a question (or 'exit'): ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if raw.strip().lower() == "exit":
            break

        if raw.strip().lower() == "/reset":
            print("\n⚠️  Resetting database and forcing a complete re-index...")
            from rag.pipeline import drop_index
            from rag.vector_store import build_or_load_index
            drop_index()
            await build_or_load_index(force_reindex=True)
            print("✅ Database completely wiped and re-indexed from scratch!")
            continue

        if raw.strip().lower() == "/refresh":
            print("\n🔄 Scanning for new or changed files incrementally...")
            from rag.pipeline import drop_index
            from rag.vector_store import build_or_load_index
            drop_index()
            await build_or_load_index(force_reindex=False)
            print("✅ Refresh complete!")
            continue

        question, warnings = _sanitise_input(raw)
        for w in warnings:
            print(w)

        if question == "[empty query after sanitisation]":
            print("Please enter a non-empty question.")
            continue

        try:
            async for chunk in run_pipeline_stream(question):
                print(chunk, end="", flush=True)
            print()  
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())

"""
Prompt-building and output-cleaning for the shunt worker calls.

Ported verbatim from Spotify's shunt plugin README (see `auto_router_shunt.py`'s module
docstring for the source): the same system prompts, the same `<file path="...">` wrapping for
bulk-read, and the same fenced-code stripping for code-write.
"""

import re
from collections.abc import Mapping
from typing import Final

BULK_READ_SYSTEM_PROMPT: Final = (
    "You are a precise code analyst. Read the provided files and answer the question "
    "concisely. Output structured bullets only. No greetings, no prose, no preambles, no "
    "summaries. Lead every bullet with the exact name, type, or line number. Use nested "
    "bullets for details. Skip anything the caller did not ask for."
)

CODE_WRITE_SYSTEM_PROMPT: Final = (
    "You generate code files based on a spec and reference files. Match the existing "
    "patterns, conventions, naming, and style exactly. Output only the code, no explanations, "
    "no markdown fences unless asked. If the spec is ambiguous, make reasonable choices that "
    "match the patterns in the reference code."
)

WORKER_TEMPERATURE: Final = 0.2

_FENCE_LINE: Final = re.compile(r"^```.*$", re.MULTILINE)


def build_bulk_read_message(question: str, files: Mapping[str, str]) -> str:
    """The user message for a bulk-read call: each file wrapped in `<file path="...">` tags."""
    file_blocks: Final = "".join(f'<file path="{path}">\n{content}\n</file>\n\n' for path, content in files.items())
    return f"{file_blocks}Question: {question}"


def build_code_write_message(spec: str, reference_path: str, reference_content: str) -> str:
    """The user message for a code-write call: the spec plus one reference file."""
    return f"Spec: {spec}\n\nReference:\n{reference_content}" if reference_path else f"Spec: {spec}"


def strip_code_fences(text: str) -> str:
    """Remove markdown code-fence lines, matching shunt's `sed '/^```/d'` post-processing.

    The model is instructed not to wrap output in fences, but does anyway often enough that
    shunt's own script strips them unconditionally rather than trusting the instruction.
    """
    return _FENCE_LINE.sub("", text).strip()

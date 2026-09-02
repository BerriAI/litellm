from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

FunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True, slots=True)
class Source:
    path: str
    names: tuple[str, ...] = ()


HTTP_HANDLER: Final = "litellm/llms/custom_httpx/llm_http_handler.py"
BASE_CONFIG: Final = "litellm/llms/base_llm"
ROUTES: Final = MappingProxyType(
    {
        "chat_completions": (
            Source("litellm/main.py", ("completion", "acompletion", "_complete_anthropic", "_complete_bedrock")),
            Source("litellm/utils.py", ("get_non_default_completion_params", "get_optional_params")),
            Source(
                HTTP_HANDLER,
                ("completion", "async_completion", "_make_common_sync_call", "_make_common_async_call"),
            ),
            Source(f"{BASE_CONFIG}/chat/transformation.py"),
            Source("litellm/llms/anthropic/chat/handler.py", ("completion", "acompletion_function")),
            Source("litellm/llms/bedrock/chat/converse_handler.py", ("completion", "async_completion")),
            Source(
                "litellm/llms/bedrock/chat/converse_transformation.py",
                ("_transform_request", "_async_transform_request", "_transform_response"),
            ),
        ),
        "audio_transcription": (
            Source("litellm/main.py", ("transcription", "atranscription")),
            Source("litellm/utils.py", ("get_non_default_transcription_params", "get_optional_params_transcription")),
            Source(
                HTTP_HANDLER,
                (
                    "audio_transcriptions",
                    "async_audio_transcriptions",
                    "_prepare_audio_transcription_request",
                    "_transform_audio_transcription_response",
                ),
            ),
            Source(f"{BASE_CONFIG}/audio_transcription/transformation.py"),
            Source(f"{BASE_CONFIG}/chat/transformation.py"),
            Source(
                "litellm/llms/openai/transcriptions/handler.py", ("audio_transcriptions", "async_audio_transcriptions")
            ),
            Source(
                "litellm/llms/azure/audio_transcriptions.py", ("audio_transcriptions", "async_audio_transcriptions")
            ),
        ),
        "messages": (
            Source("litellm/anthropic_interface/messages/__init__.py", ("create", "acreate")),
            Source("litellm/llms/anthropic/experimental_pass_through/messages/handler.py"),
            Source(
                HTTP_HANDLER,
                (
                    "anthropic_messages_handler",
                    "async_anthropic_messages_handler",
                    "_async_post_anthropic_messages_with_http_error_retry",
                    "_finalize_anthropic_messages_response",
                    "_resolve_anthropic_messages_timeout",
                ),
            ),
            Source(f"{BASE_CONFIG}/anthropic_messages/transformation.py"),
        ),
        "ocr": (
            Source(
                "litellm/ocr/main.py", ("ocr", "aocr", "_prepare_ocr_request", "convert_file_document_to_url_document")
            ),
            Source(
                HTTP_HANDLER,
                ("ocr", "async_ocr", "_prepare_ocr_request", "_async_prepare_ocr_request", "_transform_ocr_response"),
            ),
            Source(f"{BASE_CONFIG}/ocr/transformation.py"),
            Source("litellm/llms/mistral/ocr/transformation.py"),
        ),
    }
)


def functions(nodes: list[ast.stmt], owner: str = "") -> Iterator[tuple[str, FunctionNode]]:
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            yield from functions(node.body, f"{owner}{node.name}.")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield f"{owner}{node.name}", node


def call_sites(node: ast.AST) -> Iterator[ast.Call]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    if isinstance(node, ast.Call):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from call_sites(child)


def source_lines(repo: Path, source: Source, *, signatures: bool, calls: bool) -> Iterator[str]:
    path: Final = repo / source.path
    if not path.is_file():
        yield f"  NOT FOUND: {source.path}"
        return
    tree: Final = ast.parse(path.read_text(), filename=str(path))
    selected: Final = tuple(
        (name, node) for name, node in functions(tree.body) if not source.names or node.name in source.names
    )
    for missing in sorted(frozenset(source.names) - frozenset(node.name for _, node in selected)):
        yield f"  NOT FOUND: {source.path} :: {missing}"
    for name, node in selected:
        yield f"  {source.path}:{node.lineno}  {name}"
        if signatures:
            prefix: Final = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            returns: Final = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            yield f"    {prefix} {node.name}({ast.unparse(node.args)}){returns}"
        if calls:
            for call in sorted(
                (call for statement in node.body for call in call_sites(statement)),
                key=lambda call: (call.lineno, call.col_offset),
            ):
                yield f"      L{call.lineno}: {ast.unparse(call.func)}"


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="List Python route functions without importing LiteLLM")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--route", choices=("all", *ROUTES), default="all")
    parser.add_argument("--signatures", action="store_true", help="Include parameter and return declarations")
    parser.add_argument("--calls", action="store_true", help="Include direct call sites in source order")
    args: Final = parser.parse_args()
    sys.stdout.write(
        f"Python source: {args.repo.resolve()}\n"
        "Source inventory, not an execution trace. Includes branches; call sites are not runtime order.\n"
        "Scope: SDK entrypoints, shared handlers/bases, and selected provider paths. No enforcement.\n\n"
    )
    for route, sources in ROUTES.items():
        if args.route in ("all", route):
            sys.stdout.write(f"{route}\n")
            for source in sources:
                sys.stdout.write(
                    "\n".join(source_lines(args.repo, source, signatures=args.signatures, calls=args.calls))
                )
                sys.stdout.write("\n")
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()

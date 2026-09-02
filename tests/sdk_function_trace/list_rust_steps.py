# /// script
# requires-python = ">=3.10"
# dependencies = ["tree-sitter==0.25.2", "tree-sitter-rust==0.24.2"]
# ///
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import tree_sitter_rust
from tree_sitter import Language, Node, Parser

ROUTES: Final = ("chat_completions", "audio_transcription", "messages", "ocr")
FUNCTIONS: Final = ("function_item", "function_signature_item")


def text(source: bytes, node: Node | None) -> str:
    return source[node.start_byte : node.end_byte].decode() if node is not None else ""


def declarations(source: bytes, node: Node, owner: str = "") -> Iterator[tuple[str, Node]]:
    for child in node.named_children:
        name: Final = text(source, child.child_by_field_name("name"))
        body: Final = child.child_by_field_name("body")
        if child.type in FUNCTIONS:
            yield f"{owner}{name}", child
        elif child.type == "macro_invocation":
            yield f"{owner}{text(source, child.child_by_field_name('macro'))}! [unexpanded macro]", child
        elif child.type == "expression_statement":
            yield from declarations(source, child, owner)
        elif (
            child.type == "mod_item"
            and name != "tests"
            and body is not None
            or child.type == "trait_item"
            and body is not None
        ):
            yield from declarations(source, body, f"{owner}{name}::")
        elif child.type == "impl_item" and body is not None:
            target: Final = text(source, child.child_by_field_name("type"))
            trait: Final = text(source, child.child_by_field_name("trait"))
            implementation: Final = f"<{target} as {trait}>" if trait else target
            yield from declarations(source, body, f"{owner}{implementation}::")


def call_sites(node: Node) -> Iterator[Node]:
    if node.type in (*FUNCTIONS, "impl_item", "trait_item", "mod_item"):
        return
    if node.type in ("call_expression", "macro_invocation"):
        yield node
    for child in node.named_children:
        yield from call_sites(child)


def call_name(source: bytes, node: Node | None) -> str:
    if node is None:
        return ""
    if node.type == "call_expression":
        return f"{call_name(source, node.child_by_field_name('function'))}(...)"
    if node.type == "field_expression":
        return (
            f"{call_name(source, node.child_by_field_name('value'))}.{text(source, node.child_by_field_name('field'))}"
        )
    if node.type in ("await_expression", "try_expression"):
        suffix: Final = ".await" if node.type == "await_expression" else "?"
        return f"{call_name(source, node.named_children[0])}{suffix}"
    return " ".join(text(source, node).split())


def route_files(repo: Path, route: str) -> tuple[Path, ...]:
    crates: Final = repo / "litellm-rust/crates"
    bridge: Final = crates / f"python-bridge/src/routes/{route}.rs"
    patterns: Final = (
        f"core/src/{route}/**/*.rs",
        f"core/src/providers/*/{route}/**/*.rs",
        f"core/src/providers/*/{route}.rs",
        f"ai-gateway/src/{route}/**/*.rs",
        f"ai-gateway/src/io/{route}.rs",
    )
    return (
        bridge,
        *sorted(
            frozenset(
                path
                for pattern in patterns
                for path in crates.glob(pattern)
                if path.name != "tests.rs" and not path.name.startswith("test_")
            )
        ),
    )


def source_lines(parser: Parser, repo: Path, path: Path, *, signatures: bool, calls: bool) -> Iterator[str]:
    relative: Final = path.relative_to(repo)
    if not path.is_file():
        yield f"  NOT FOUND: {relative}"
        return
    source: Final = path.read_bytes()
    tree: Final = parser.parse(source)
    if tree.root_node.has_error:
        yield f"  PARSE ERROR: {relative} (listing may be incomplete)"
    for name, node in declarations(source, tree.root_node):
        yield f"  {relative}:{node.start_point.row + 1}  {name}"
        if node.type not in FUNCTIONS:
            continue
        body: Final = node.child_by_field_name("body")
        if signatures:
            end: Final = body.start_byte if body is not None else node.end_byte
            signature: Final = source[node.start_byte : end].decode().strip().removesuffix(";")
            yield f"    {' '.join(signature.split())}"
        if calls and body is not None:
            for call in sorted(call_sites(body), key=lambda call: call.start_byte):
                label: Final = (
                    f"{text(source, call.child_by_field_name('macro'))}! [macro]"
                    if call.type == "macro_invocation"
                    else call_name(source, call.child_by_field_name("function"))
                )
                yield f"      L{call.start_point.row + 1}: {label}"


def main() -> None:
    arguments: Final = argparse.ArgumentParser(description="List Rust route functions without building LiteLLM")
    arguments.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    arguments.add_argument("--route", choices=("all", *ROUTES), default="all")
    arguments.add_argument("--signatures", action="store_true", help="Include parameter and return declarations")
    arguments.add_argument("--calls", action="store_true", help="Include call sites in source order")
    args: Final = arguments.parse_args()
    repo: Final = args.repo.resolve()
    parser: Final = Parser(Language(tree_sitter_rust.language()))
    sys.stdout.write(
        f"Rust source: {repo}\n"
        "Source inventory, not an execution trace. Includes branches; call sites are not runtime order.\n"
        "Scope: bridge, core route/provider, and gateway route/I/O modules. No enforcement.\n"
        "Macros are listed but not expanded; cfg conditions are not evaluated. Test modules/files are omitted.\n\n"
    )
    for route in ROUTES:
        if args.route in ("all", route):
            sys.stdout.write(f"{route}\n")
            for path in route_files(repo, route):
                sys.stdout.write(
                    "\n".join(source_lines(parser, repo, path, signatures=args.signatures, calls=args.calls))
                )
                sys.stdout.write("\n")
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()

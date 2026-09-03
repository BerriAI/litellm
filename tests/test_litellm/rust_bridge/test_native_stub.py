"""Pin the ``_native.pyi`` stub to the compiled module's public surface."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

import litellm.rust_bridge

# keep in sync with crates/python-bridge/src/lib.rs surface test
PINNED_SURFACE: Final = (
    "RustBridgeDeclined",
    "RustUpstreamError",
    "ocr",
    "aocr",
    "transcription",
    "atranscription",
    "messages",
    "amessages",
    "chat_completions_decline",
    "chat_completions",
    "achat_completions",
    "ResponsesWebSocketConnection",
    "gil_stats",
    "build_info",
)


def _stub_path() -> Path:
    module_file: Final = litellm.rust_bridge.__file__
    assert module_file is not None
    return Path(module_file).with_name("_native.pyi")


def _parse_stub() -> ast.Module:
    return ast.parse(_stub_path().read_text(encoding="utf-8"), filename=str(_stub_path()))


def _module_level_names(module: ast.Module) -> frozenset[str]:
    def names(body: list[ast.stmt]) -> Iterator[str]:
        for node in body:
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                yield node.name
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        yield target.id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                yield node.target.id

    return frozenset(names(module.body))


def test_stub_pins_the_native_surface() -> None:
    stub_names: Final = _module_level_names(_parse_stub())
    public_names: Final = {name for name in stub_names if not name.startswith("__")}

    assert public_names == set(PINNED_SURFACE)
    assert "__version__" in stub_names


def test_stub_names_exist_on_the_compiled_module() -> None:
    try:
        from litellm.rust_bridge import _native
    except ImportError:
        pytest.skip("compiled _native extension is not importable in this environment")

    runtime_names: Final = frozenset(dir(_native))

    assert _module_level_names(_parse_stub()) <= runtime_names

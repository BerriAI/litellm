import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import bindings


def test_binding_distinguishes_disable_from_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = SimpleNamespace(chat_completions=lambda: "native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    binding: Final = bindings.NativeBinding(lambda module: module.chat_completions)

    assert binding.load() is native.chat_completions
    binding.override(None)
    assert binding.load() is None
    replacement: Final = SimpleNamespace(chat_completions=lambda: "replacement")
    binding.override(replacement.chat_completions)
    assert binding.load() is replacement.chat_completions
    binding.reset()
    assert binding.load() is native.chat_completions


@pytest.mark.parametrize("native", (None, SimpleNamespace(), SimpleNamespace(chat_completions=3)))
def test_missing_or_invalid_export_is_unavailable(monkeypatch: pytest.MonkeyPatch, native: object) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    binding: Final = bindings.NativeBinding(lambda module: module.chat_completions)

    assert binding.load() is None


def test_selection_is_lazy_and_preserves_other_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: pytest.fail("must not load during construction"))
    binding: Final = bindings.NativeBinding(lambda module: module.chat_completions)
    native: Final = SimpleNamespace(chat_completions=lambda: "native", achat_completions=None)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)

    assert binding.load() is native.chat_completions
    assert bindings.NativeBinding(lambda module: module.achat_completions).load() is None


@pytest.mark.parametrize("invalid", (None, str, lambda: None))
def test_native_exception_types_reject_non_exception_classes(monkeypatch: pytest.MonkeyPatch, invalid: object) -> None:
    native: Final = SimpleNamespace(RustBridgeDeclined=invalid, RustUpstreamError=RuntimeError)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)

    assert bindings.native_declined_types() == ()
    assert bindings.native_upstream_types() == (RuntimeError,)


@pytest.mark.parametrize(
    ("expression", "expected_rule"),
    (
        ("NativeBinding(lambda native: native.chat_completion)", "reportAttributeAccessIssue"),
        (
            "wrong: NativeBinding[RustAchatCompletions] = NativeBinding(lambda native: native.chat_completions)",
            "reportAssignmentType",
        ),
        ("NativeBinding(lambda native: native.ocrr)", "reportAttributeAccessIssue"),
        (
            "wrong: NativeBinding[RustAmessages] = NativeBinding(lambda native: native.messages)",
            "reportAssignmentType",
        ),
        (
            "wrong: NativeBinding[RustAocr] = NativeBinding(lambda native: native.ocr)",
            "reportAssignmentType",
        ),
        (
            "wrong: NativeBinding[RustAtranscription] = NativeBinding(lambda native: native.transcription)",
            "reportAssignmentType",
        ),
    ),
)
def test_selectors_are_checked_by_type_checker(tmp_path: Path, expression: str, expected_rule: str) -> None:
    source: Final = tmp_path / "binding_contract.py"
    source.write_text(
        "from typing_extensions import assert_type\n"
        "from litellm.rust_bridge.bindings import NativeBinding\n"
        "from litellm.rust_bridge.protocols import RustChatCompletions, RustAchatCompletions, "
        "RustMessages, RustAmessages, RustOcr, RustAocr, RustTranscription, RustAtranscription\n"
        "binding = NativeBinding(lambda native: native.chat_completions)\n"
        "assert_type(binding, NativeBinding[RustChatCompletions])\n"
        "assert_type(NativeBinding(lambda native: native.messages), NativeBinding[RustMessages])\n"
        "assert_type(NativeBinding(lambda native: native.ocr), NativeBinding[RustOcr])\n"
        "assert_type(NativeBinding(lambda native: native.transcription), NativeBinding[RustTranscription])\n"
        + expression
        + "\n"
    )
    config: Final = tmp_path / "pyrightconfig.json"
    config.write_text(
        json.dumps(
            {
                "include": [str(source)],
                "extraPaths": [str(Path(__file__).resolve().parents[3])],
                "typeCheckingMode": "basic",
            }
        )
    )
    result: Final = subprocess.run(
        [sys.executable, "-m", "basedpyright", "--project", str(config), "--outputjson"],
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics: Final = json.loads(result.stdout)["generalDiagnostics"]
    assert result.returncode == 1, result.stdout + result.stderr
    assert [(item["rule"], item["range"]["start"]["line"]) for item in diagnostics] == [
        (expected_rule, len(source.read_text().splitlines()) - 1)
    ]


@pytest.mark.parametrize("export", (None, 3))
def test_missing_execution_export_does_not_inspect_readiness(export):
    from types import ModuleType

    native = ModuleType("test_native")
    lookups = []

    def missing(name: str):
        lookups.append(name)
        raise AttributeError(name)

    native.__getattr__ = missing
    if export is not None:
        native.messages = export
    binding = bindings.NativeBinding(lambda module: module.messages, route="messages", module_loader=lambda: native)
    assert binding.load() is None
    assert "ready_endpoints" not in lookups


def test_discovery_reuses_one_module_and_does_not_cache_binding():
    from types import ModuleType

    native = ModuleType("test_native")
    native.ready_endpoints = {"messages": frozenset({"callbacks"})}

    def first():
        return "first"

    def second():
        return "second"

    native.messages = first
    loads = []

    def load():
        loads.append(native)
        return native

    binding = bindings.NativeBinding(lambda module: module.messages, route="messages", module_loader=load)
    assert binding.load() is first
    assert len(loads) == 1
    native.messages = second
    assert binding.load() is second
    assert len(loads) == 2
    native.ready_endpoints = {"messages": frozenset()}
    assert binding.load() is None
    assert len(loads) == 3

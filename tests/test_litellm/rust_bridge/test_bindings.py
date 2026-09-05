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

    assert bindings.native_exception_types() is None


@pytest.mark.parametrize(
    ("expression", "expected_rule"),
    (
        ("NativeBinding(lambda native: native.chat_completion)", "reportAttributeAccessIssue"),
        (
            "wrong: NativeBinding[RustAchatCompletions] = NativeBinding(lambda native: native.chat_completions)",
            "reportAssignmentType",
        ),
        (
            'EndpointBinding.native(route="chat", select=lambda native: native.chat_completion, enabled=always_enabled)',
            "reportAttributeAccessIssue",
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
        "from litellm.rust_bridge.runtime import EndpointBinding, EndpointDispatch, always_enabled\n"
        "binding = NativeBinding(lambda native: native.chat_completions)\n"
        "assert_type(binding, NativeBinding[RustChatCompletions])\n"
        'bridge = EndpointBinding.native(route="chat", select=lambda native: native.chat_completions, enabled=always_enabled)\n'
        "assert_type(bridge, EndpointBinding[RustChatCompletions])\n"
        'endpoint = EndpointDispatch.native(route="chat", sync=lambda native: native.chat_completions, '
        "asynchronous=lambda native: native.achat_completions, enabled=always_enabled)\n"
        "assert_type(endpoint, EndpointDispatch[RustChatCompletions, RustAchatCompletions])\n"
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
    assert [(item["rule"], item["range"]["start"]["line"]) for item in diagnostics] == [(expected_rule, 13)]

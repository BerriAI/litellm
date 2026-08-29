from __future__ import annotations

import queue
from pathlib import Path
from typing import Final

import pytest

from tests.test_litellm._fixture_recorder import generate_case_inputs
from tests.test_litellm.ocr.generate_fixtures import (
    discover_targets,
    parse_generator_args,
    require_targets,
)


def _unused_sdk_call(**kwargs: object) -> object:
    raise AssertionError(f"unexpected SDK call with {tuple(kwargs)}")


def test_parse_args_has_no_model_selection() -> None:
    args: Final = parse_generator_args(["--examples", "2", "--concurrency", "3", "--fixture-dir", "/tmp/ocr"])

    assert args.examples == 2
    assert args.concurrency == 3
    assert args.fixture_dir == Path("/tmp/ocr")
    with pytest.raises(SystemExit):
        parse_generator_args(["--model", "mistral/mistral-ocr-latest"])


@pytest.mark.parametrize(
    "environ",
    (
        {},
        {"MISTRAL_API_KEY": ""},
        {"LITELLM_API_KEY": "generic-key"},
    ),
)
def test_discovery_requires_provider_specific_key(environ: dict[str, str]) -> None:
    assert discover_targets(environ, _unused_sdk_call) == ()


def test_no_discovered_targets_has_actionable_error() -> None:
    with pytest.raises(SystemExit, match="Set MISTRAL_API_KEY"):
        require_targets(())


@pytest.mark.parametrize(
    ("configured", "expected"),
    (
        (None, "https://api.mistral.ai"),
        ("https://mistral.example/v1", "https://mistral.example"),
        ("https://mistral.example/", "https://mistral.example"),
    ),
)
def test_mistral_target_uses_canonical_model_and_normalized_base(
    configured: str | None,
    expected: str,
) -> None:
    environ: Final = {
        "MISTRAL_API_KEY": "mistral-secret",
        **({"MISTRAL_API_BASE": configured} if configured is not None else {}),
    }
    targets: Final = discover_targets(environ, _unused_sdk_call)

    assert len(targets) == 1
    target: Final = targets[0]
    assert target.name == "mistral-ocr"
    assert target.provider_spec.upstream_base == expected
    assert "mistral-secret" not in repr(target)
    case_inputs: Final = generate_case_inputs(target.strategy, examples=1)
    assert len(case_inputs) == 1
    assert case_inputs[0].canonical_input()["model"] == "mistral/mistral-ocr-latest"


def test_mistral_target_invocation_forwards_discovered_credentials() -> None:
    calls: Final[queue.SimpleQueue[dict[str, object]]] = queue.SimpleQueue()

    def sdk_call(**kwargs: object) -> object:
        calls.put(kwargs)
        return object()

    target: Final = discover_targets({"MISTRAL_API_KEY": "mistral-secret"}, sdk_call)[0]
    case_input: Final = generate_case_inputs(target.strategy, examples=1)[0]

    target.invoke("http://127.0.0.1:1234", case_input)

    kwargs: Final = calls.get_nowait()
    assert kwargs["api_base"] == "http://127.0.0.1:1234"
    assert kwargs["api_key"] == "mistral-secret"
    assert kwargs["model"] == "mistral/mistral-ocr-latest"

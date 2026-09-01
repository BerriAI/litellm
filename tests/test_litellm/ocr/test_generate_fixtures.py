from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from tests.route_parity.fixture_recorder import generate_case_inputs
from tests.test_litellm.ocr.fixtures.generate import (
    discover_targets,
    parse_generator_args,
    require_targets,
)
from tests.test_litellm.ocr.fixtures.models import OcrSdkInputBase


class _UnusedOcrClient:
    def execute(self, api_base: str, api_key: str, case_input: OcrSdkInputBase) -> None:
        raise AssertionError(f"unexpected SDK call to {api_base} with {api_key!r} and {case_input!r}")


@dataclass(frozen=True, slots=True)
class _RecordingOcrClient:
    calls: queue.SimpleQueue[dict[str, object]]

    def execute(self, api_base: str, api_key: str, case_input: OcrSdkInputBase) -> None:
        self.calls.put({"api_base": api_base, "api_key": api_key, **case_input.as_sdk_kwargs()})


_UNUSED_OCR_CLIENT: Final = _UnusedOcrClient()


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
    assert discover_targets(environ, _UNUSED_OCR_CLIENT) == ()


def test_no_discovered_targets_has_actionable_error() -> None:
    with pytest.raises(SystemExit, match="supported provider API key"):
        require_targets(())


def test_discovery_is_explicit_per_available_provider_boundary() -> None:
    targets: Final = discover_targets(
        {
            "MISTRAL_API_KEY": "mistral-secret",
            "REDUCTO_API_KEY": "reducto-secret",
            "AZURE_AI_API_KEY": "azure-secret",
            "AZURE_AI_API_BASE": "https://azure.example",
            "AZURE_AI_OCR_MODEL": "mistral-ocr-deployment",
            "AZURE_DOCUMENT_INTELLIGENCE_API_KEY": "document-secret",
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": "https://document.example",
            "VERTEX_AI_API_KEY": "vertex-secret",
            "VERTEXAI_PROJECT": "project-1",
        },
        _UNUSED_OCR_CLIENT,
    )

    assert tuple(target.name for target in targets) == (
        "mistral-ocr",
        "azure-mistral",
        "azure-document-intelligence",
        "vertex-mistral",
        "vertex-deepseek",
        "reducto-v3",
        "reducto-legacy",
    )
    assert all("secret" not in repr(target) for target in targets)


def test_azure_mistral_discovery_requires_and_normalizes_deployment_model() -> None:
    incomplete: Final = {
        "AZURE_AI_API_KEY": "azure-secret",
        "AZURE_AI_API_BASE": "https://azure.example",
    }
    assert discover_targets(incomplete, _UNUSED_OCR_CLIENT) == ()

    target: Final = discover_targets({**incomplete, "AZURE_AI_OCR_MODEL": "mistral-ocr-deployment"}, _UNUSED_OCR_CLIENT)[
        0
    ]
    assert target.required_inputs[0].canonical_input()["model"] == "azure_ai/mistral-ocr-deployment"


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
    targets: Final = discover_targets(environ, _UNUSED_OCR_CLIENT)

    assert len(targets) == 1
    target: Final = targets[0]
    assert target.name == "mistral-ocr"
    assert target.provider_spec.upstream_base == expected
    assert "mistral-secret" not in repr(target)
    case_inputs: Final = generate_case_inputs(target.strategy, examples=1)
    assert len(case_inputs) == 1
    assert case_inputs[0].canonical_input()["model"] == "mistral/mistral-ocr-latest"
    assert len(target.required_inputs) == 14
    covered_params: Final = {
        key
        for case_input in target.required_inputs
        for key in case_input.as_sdk_kwargs()
        if key not in {"model", "document", "custom_llm_provider"}
    }
    assert covered_params == {
        "pages",
        "include_image_base64",
        "image_limit",
        "image_min_size",
        "bbox_annotation_format",
        "document_annotation_format",
        "document_annotation_prompt",
        "extract_header",
        "extract_footer",
        "table_format",
        "confidence_scores_granularity",
        "include_blocks",
        "id",
    }


def test_mistral_target_invocation_forwards_discovered_credentials() -> None:
    calls: Final[queue.SimpleQueue[dict[str, object]]] = queue.SimpleQueue()

    client: Final = _RecordingOcrClient(calls)
    target: Final = discover_targets({"MISTRAL_API_KEY": "mistral-secret"}, client)[0]
    case_input: Final = generate_case_inputs(target.strategy, examples=1)[0]

    target.invocation.execute("http://127.0.0.1:1234", case_input)

    kwargs: Final = calls.get_nowait()
    assert kwargs["api_base"] == "http://127.0.0.1:1234"
    assert kwargs["api_key"] == "mistral-secret"
    assert kwargs["model"] == "mistral/mistral-ocr-latest"

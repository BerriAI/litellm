from __future__ import annotations

import queue
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pytest
from hypothesis import find, settings
from hypothesis.strategies import SearchStrategy

from tests.route_parity.fixtures.inputs import generate_case_inputs
from tests.route_parity.fixtures.media import structured_pdf_data_uri
from tests.route_parity.fixtures.pipeline import parse_recording_args
from tests.test_litellm.ocr.fixtures.azure import (
    AZURE_DOCUMENT_INTELLIGENCE_RECORDING_MODELS,
    AZURE_MISTRAL_MODELS,
)
from tests.test_litellm.ocr.fixtures.base import OcrSdkInputBase
from tests.test_litellm.ocr.fixtures.common import OcrFixtureClient, OcrRecordingTarget
from tests.test_litellm.ocr.fixtures.mistral import MISTRAL_MODELS, MISTRAL_PROVIDER_REJECTED_INPUTS
from tests.test_litellm.ocr.fixtures.record import (
    discover_targets as discover_targets_with_media,
)
from tests.test_litellm.ocr.fixtures.record import (
    require_targets,
)
from tests.test_litellm.ocr.fixtures.reducto import REDUCTO_LEGACY_MODELS, REDUCTO_V3_MODELS
from tests.test_litellm.ocr.fixtures.vertex import VERTEX_DEEPSEEK_MODELS, VERTEX_MISTRAL_MODELS


class _UnusedOcrClient:
    def execute(self, api_base: str, api_key: str, case_input: OcrSdkInputBase) -> None:
        raise AssertionError(f"unexpected SDK call to {api_base} with {api_key!r} and {case_input!r}")


@dataclass(frozen=True, slots=True)
class _RecordingOcrClient:
    calls: queue.SimpleQueue[dict[str, object]]

    def execute(self, api_base: str, api_key: str, case_input: OcrSdkInputBase) -> None:
        self.calls.put({"api_base": api_base, "api_key": api_key, **case_input.as_sdk_kwargs()})


_UNUSED_OCR_CLIENT: Final = _UnusedOcrClient()
_MISTRAL_PARAMS: Final = frozenset(
    {
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
    }
)
_MISTRAL_2512_PARAMS: Final = _MISTRAL_PARAMS - {"include_blocks"}
_MISTRAL_2505_PARAMS: Final = _MISTRAL_2512_PARAMS - {"extract_header", "extract_footer", "table_format"}
_AZURE_MISTRAL_PARAMS: Final = _MISTRAL_2505_PARAMS - {"document_annotation_prompt"}
_FIND_SETTINGS: Final = settings(max_examples=2_000, deadline=None, derandomize=True, database=None)
_INLINE_IMAGE_DATA_URI: Final = "data:image/png;base64,dGVzdA=="


def discover_targets(environ: Mapping[str, str], client: OcrFixtureClient) -> tuple[OcrRecordingTarget, ...]:
    return discover_targets_with_media(environ, client, _INLINE_IMAGE_DATA_URI)


def _model(case_input: OcrSdkInputBase) -> str:
    model: Final = case_input.canonical_input().get("model")
    assert isinstance(model, str)
    return model


def _find_input(
    strategy: SearchStrategy[OcrSdkInputBase],
    predicate: Callable[[OcrSdkInputBase], bool],
) -> OcrSdkInputBase:
    return find(strategy, predicate, settings=_FIND_SETTINGS)


def _document_transport(case_input: OcrSdkInputBase) -> tuple[str, str]:
    document: Final = cast(dict[str, object], case_input.canonical_input()["document"])
    document_type: Final = cast(str, document["type"])
    source: Final = document["image_url"] if document_type == "image_url" else document["document_url"]
    assert isinstance(source, str)
    return document_type, "data" if source.startswith("data:") else "remote"


def test_parse_args_has_no_model_selection() -> None:
    args: Final = parse_recording_args(["--examples", "2", "--concurrency", "3", "--fixture-dir", "/tmp/ocr"])

    assert args.examples == 2
    assert args.concurrency == 3
    assert args.fixture_dir == Path("/tmp/ocr")
    with pytest.raises(SystemExit):
        parse_recording_args(["--model", "mistral/mistral-ocr-latest"])


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


def test_azure_mistral_discovery_enumerates_registered_models() -> None:
    environ: Final = {
        "AZURE_AI_API_KEY": "azure-secret",
        "AZURE_AI_API_BASE": "https://azure.example",
    }
    target: Final = discover_targets(environ, _UNUSED_OCR_CLIENT)[0]

    for model in AZURE_MISTRAL_MODELS:
        assert (
            _model(
                _find_input(
                    target.strategy,
                    lambda case_input, expected_model=model: _model(case_input) == expected_model,
                )
            )
            == model
        )


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
    assert case_inputs[0].canonical_input()["model"] in MISTRAL_MODELS


def test_mistral_target_invocation_forwards_discovered_credentials() -> None:
    calls: Final[queue.SimpleQueue[dict[str, object]]] = queue.SimpleQueue()

    client: Final = _RecordingOcrClient(calls)
    target: Final = discover_targets({"MISTRAL_API_KEY": "mistral-secret"}, client)[0]
    case_input: Final = generate_case_inputs(target.strategy, examples=1)[0]

    target.invocation.execute("http://127.0.0.1:1234", case_input)

    kwargs: Final = calls.get_nowait()
    assert kwargs["api_base"] == "http://127.0.0.1:1234"
    assert kwargs["api_key"] == "mistral-secret"
    assert kwargs["model"] in MISTRAL_MODELS


def test_every_target_strategy_reaches_every_recording_model_and_coverage_param() -> None:
    targets: Final = discover_targets(
        {
            "MISTRAL_API_KEY": "mistral-secret",
            "REDUCTO_API_KEY": "reducto-secret",
            "AZURE_AI_API_KEY": "azure-secret",
            "AZURE_AI_API_BASE": "https://azure.example",
            "AZURE_DOCUMENT_INTELLIGENCE_API_KEY": "document-secret",
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": "https://document.example",
            "VERTEX_AI_API_KEY": "vertex-secret",
            "VERTEXAI_PROJECT": "project-1",
        },
        _UNUSED_OCR_CLIENT,
    )
    expected: Final[dict[str, tuple[tuple[str, ...], frozenset[str]]]] = {
        "mistral-ocr": (MISTRAL_MODELS, _MISTRAL_PARAMS),
        "azure-mistral": (AZURE_MISTRAL_MODELS, _AZURE_MISTRAL_PARAMS),
        "azure-document-intelligence": (
            AZURE_DOCUMENT_INTELLIGENCE_RECORDING_MODELS,
            frozenset({"pages", "features", "req_format"}),
        ),
        "vertex-mistral": (VERTEX_MISTRAL_MODELS, _MISTRAL_2505_PARAMS),
        "vertex-deepseek": (VERTEX_DEEPSEEK_MODELS, frozenset[str]()),
        "reducto-v3": (REDUCTO_V3_MODELS, frozenset({"formatting", "retrieval", "settings"})),
        "reducto-legacy": (REDUCTO_LEGACY_MODELS, frozenset[str]()),
    }

    for target in targets:
        expected_models, expected_params = expected[target.name]
        for model in expected_models:
            assert (
                _model(
                    _find_input(
                        target.strategy,
                        lambda case_input, expected_model=model: _model(case_input) == expected_model,
                    )
                )
                == model
            )
        for param in expected_params:
            reached = _find_input(
                target.strategy,
                lambda case_input, expected_param=param: expected_param in case_input.as_sdk_kwargs(),
            )
            assert param in reached.as_sdk_kwargs()
            document = cast(dict[str, object], reached.canonical_input()["document"])
            assert document == {"type": "document_url", "document_url": structured_pdf_data_uri()}


@pytest.mark.parametrize("target_name", ("mistral-ocr", "azure-mistral", "vertex-mistral"))
def test_mistral_recording_targets_reach_every_transport_branch(target_name: str) -> None:
    targets: Final = discover_targets(
        {
            "MISTRAL_API_KEY": "mistral-secret",
            "AZURE_AI_API_KEY": "azure-secret",
            "AZURE_AI_API_BASE": "https://azure.example",
            "VERTEX_AI_API_KEY": "vertex-secret",
            "VERTEXAI_PROJECT": "project-1",
        },
        _UNUSED_OCR_CLIENT,
    )
    target: Final = next(candidate for candidate in targets if candidate.name == target_name)

    for transport in (
        ("image_url", "remote"),
        ("image_url", "data"),
        ("document_url", "remote"),
        ("document_url", "data"),
    ):
        reached = _find_input(
            target.strategy,
            lambda case_input, expected=transport: _document_transport(case_input) == expected,
        )
        assert _document_transport(reached) == transport


def test_vertex_deepseek_recording_reaches_documented_image_branch() -> None:
    targets: Final = discover_targets(
        {
            "VERTEX_AI_API_KEY": "vertex-secret",
            "VERTEXAI_PROJECT": "project-1",
        },
        _UNUSED_OCR_CLIENT,
    )
    target: Final = next(candidate for candidate in targets if candidate.name == "vertex-deepseek")
    case_input: Final = _find_input(
        target.strategy,
        lambda candidate: _document_transport(candidate) == ("image_url", "data"),
    )

    assert _document_transport(case_input) == ("image_url", "data")


def test_only_intentional_provider_failures_are_fixed_inputs() -> None:
    targets: Final = discover_targets(
        {
            "MISTRAL_API_KEY": "mistral-secret",
            "REDUCTO_API_KEY": "reducto-secret",
            "AZURE_AI_API_KEY": "azure-secret",
            "AZURE_AI_API_BASE": "https://azure.example",
            "AZURE_DOCUMENT_INTELLIGENCE_API_KEY": "document-secret",
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": "https://document.example",
            "VERTEX_AI_API_KEY": "vertex-secret",
            "VERTEXAI_PROJECT": "project-1",
        },
        _UNUSED_OCR_CLIENT,
    )

    mistral: Final = next(target for target in targets if target.name == "mistral-ocr")
    assert mistral.required_inputs == MISTRAL_PROVIDER_REJECTED_INPUTS
    generated: Final = generate_case_inputs(mistral.strategy, examples=20)
    assert all(case_input not in mistral.required_inputs for case_input in generated)
    assert all(target.required_inputs == () for target in targets if target is not mistral)


def test_mistral_adapters_preserve_omitted_optional_params() -> None:
    targets: Final = discover_targets(
        {
            "AZURE_AI_API_KEY": "azure-secret",
            "AZURE_AI_API_BASE": "https://azure.example",
            "VERTEX_AI_API_KEY": "vertex-secret",
            "VERTEXAI_PROJECT": "project-1",
        },
        _UNUSED_OCR_CLIENT,
    )

    baselines: Final = tuple(
        _find_input(
            target.strategy,
            lambda case_input: _MISTRAL_PARAMS.isdisjoint(case_input.as_sdk_kwargs()),
        )
        for target in targets
    )
    assert all(_MISTRAL_PARAMS.isdisjoint(baseline.as_sdk_kwargs()) for baseline in baselines)

"""Differential parity for the azure response path.

The v1 reference is ``convert_to_model_response_object`` with azure.py's
exact argument set (``convert_tool_call_to_json_mode=json_mode``,
``_response_headers``); ``AzureOpenAIConfig.transform_response`` raises
NotImplementedError and never runs. ``json_mode`` is always None for v2-sent
requests (the serializer fails closed on the synthetic json-tool strategy),
so the azure parser is the openai parser and azure's response extras must
ride v1's exact paths: choice-level ``content_filter_results`` into the
choice's provider_specific_fields, top-level ``prompt_filter_results`` as an
unknown-key setattr. Second gate: the vendored characterization fixtures
(recorded azure SDK dumps) against their snapshots, v1-at-HEAD and v2 both.

The azure_ai (Foundry openai-compat) rows pin the ``azure_ai/{model}`` rename:
v1 presets ``model_response.model`` in ``AzureAIStudioConfig
.transform_response`` and the convert re-prefixes the wire model.
"""

import copy
import json

import pytest

from litellm.types.utils import ModelResponse
from litellm.utils import convert_to_model_response_object

from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation.providers.azure.response import parse_response
from litellm.translation.providers.azure_ai.response import (
    parse_response as azure_ai_parse_response,
)
from litellm.translation_seam import build_translation_deps, to_model_response

from . import _azure_corpus as corpus

MODEL = "gpt-4.1"

_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
}

_FILTERS = {
    "hate": {"filtered": False, "severity": "safe"},
    "self_harm": {"filtered": False, "severity": "safe"},
    "sexual": {"filtered": False, "severity": "safe"},
    "violence": {"filtered": False, "severity": "safe"},
}

_RESPONSES = {
    "content_and_prompt_filter_results": {
        "id": "chatcmpl-AZ1",
        "object": "chat.completion",
        "created": 1718000000,
        "model": "gpt-4.1-2025-04-14",
        "system_fingerprint": "fp_az",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "content_filter_results": _FILTERS,
                "message": {
                    "content": "Hello there.",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
            }
        ],
        "usage": {"completion_tokens": 6, "prompt_tokens": 12, "total_tokens": 18},
        "prompt_filter_results": [
            {"prompt_index": 0, "content_filter_results": _FILTERS}
        ],
    },
    "tool_calls_rewrites_stop": {
        "id": "chatcmpl-AZ2",
        "object": "chat.completion",
        "created": 1718000001,
        "model": "gpt-4.1-2025-04-14",
        "system_fingerprint": "fp_az",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "content_filter_results": {},
                "message": {
                    "content": None,
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"completion_tokens": 9, "prompt_tokens": 21, "total_tokens": 30},
    },
}


def _v1_model_response(raw: dict) -> dict:
    return corpus.run_v1_response_transform(raw).model_dump()


def _v2_model_response(raw: dict) -> dict:
    parsed = parse_request(copy.deepcopy(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    response = parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(body, usage_style="openai").model_dump()


def _norm(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(_RESPONSES))
def test_v2_response_matches_v1(name: str, frozen_ambient) -> None:
    raw = _RESPONSES[name]
    assert _norm(_v2_model_response(raw)) == _norm(_v1_model_response(raw))


# ---------------------------------------------------------------------------
# Second gate: vendored characterization fixtures.
# ---------------------------------------------------------------------------

_FIXTURES = sorted(
    path.stem for path in (corpus.FIXTURES_DIR / "responses" / "azure").glob("*.json")
)


@pytest.mark.parametrize("fixture_id", _FIXTURES)
def test_v1_still_matches_response_snapshot(fixture_id: str, frozen_ambient) -> None:
    raw = corpus.load_json(
        corpus.FIXTURES_DIR / "responses" / "azure" / f"{fixture_id}.json"
    )
    snapshot = corpus.SNAPSHOTS_DIR / "responses" / "azure" / f"{fixture_id}.json"
    assert corpus.canonical_json(corpus.run_v1_response_transform(raw)) == (
        snapshot.read_text()
    ), (
        f"v1 drifted from the characterization snapshot for {fixture_id}; "
        "regenerate the corpus and ship the diff as its own PR"
    )


@pytest.mark.parametrize("fixture_id", _FIXTURES)
def test_v2_matches_response_snapshot(fixture_id: str, frozen_ambient) -> None:
    raw = corpus.load_json(
        corpus.FIXTURES_DIR / "responses" / "azure" / f"{fixture_id}.json"
    )
    snapshot = corpus.SNAPSHOTS_DIR / "responses" / "azure" / f"{fixture_id}.json"
    assert corpus.canonical_json(_v2_model_response(raw)) == snapshot.read_text()


# ---------------------------------------------------------------------------
# azure_ai: the Foundry rename (httpx transform_response path).
# ---------------------------------------------------------------------------

_AZURE_AI_MODEL = "Mistral-large-2407"

_AZURE_AI_RESPONSE = {
    "id": "chatcmpl-FND1",
    "object": "chat.completion",
    "created": 1718000002,
    "model": "mistral-large-2407",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "logprobs": None,
            "message": {"content": "Bonjour.", "role": "assistant"},
        }
    ],
    "usage": {"completion_tokens": 3, "prompt_tokens": 9, "total_tokens": 12},
}


def _v1_azure_ai_model_response(raw: dict) -> dict:
    model_response = ModelResponse()
    model_response.model = f"azure_ai/{_AZURE_AI_MODEL}"
    result = convert_to_model_response_object(
        response_object=copy.deepcopy(raw),
        model_response_object=model_response,
        _response_headers={},
    )
    assert isinstance(result, ModelResponse)
    return result.model_dump()


def _v2_azure_ai_model_response(raw: dict) -> dict:
    request = {
        "model": _AZURE_AI_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    }
    parsed = parse_request(request)
    assert parsed.is_ok(), parsed.error.summary
    response = azure_ai_parse_response(copy.deepcopy(raw), parsed.ok)
    assert response.is_ok(), response.error.summary
    body = serialize_response(response.ok, build_translation_deps(), "openai")
    return to_model_response(body, usage_style="openai").model_dump()


def test_azure_ai_model_rename_matches_v1(frozen_ambient) -> None:
    assert _norm(_v2_azure_ai_model_response(_AZURE_AI_RESPONSE)) == _norm(
        _v1_azure_ai_model_response(_AZURE_AI_RESPONSE)
    )


def test_azure_ai_model_rename_without_wire_model(frozen_ambient) -> None:
    raw = {k: v for k, v in _AZURE_AI_RESPONSE.items() if k != "model"}
    assert _norm(_v2_azure_ai_model_response(raw)) == _norm(
        _v1_azure_ai_model_response(raw)
    )

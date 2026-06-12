"""Differential parity for the bedrock response transforms.

Each recorded provider payload (characterization fixtures) goes through v1's
``transform_response`` and v2's ``parse_response`` -> ``serialize_response``
-> ``to_model_response``; the ``ModelResponse`` dumps must be identical and
must still match the committed corpus snapshot (drift guard). uuid/time are
frozen because both sides mint ambient ids; the chatcmpl id itself is the
ambient envelope and is normalized.
"""

import copy
import json

import pytest

from litellm.translation.engine.pipeline import (
    _RESPONSE_PARSERS,
    response_dialect,
)
from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.response import serialize_response
from litellm.translation_seam import build_translation_deps, to_model_response

from . import _bedrock_corpus as corpus

_MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


def _fixture_ids(provider_key: str) -> list:
    return sorted(
        path.stem
        for path in (corpus.FIXTURES_DIR / "responses" / provider_key).glob("*.json")
    )


def _norm(payload: dict) -> str:
    return json.dumps({**payload, "id": "chatcmpl-X"}, sort_keys=True, default=str)


def _v2_model_response(provider_key: str, payload: dict) -> dict:
    model, _, _ = corpus.resolve(provider_key)
    parsed = parse_request(
        {"model": model, "max_tokens": 256, "messages": copy.deepcopy(_MESSAGES)}
    )
    assert parsed.is_ok(), parsed.error.summary
    result = _RESPONSE_PARSERS[provider_key](copy.deepcopy(payload), parsed.ok)
    assert result.is_ok(), result.error.summary
    dialect = response_dialect(provider_key)
    body = serialize_response(result.ok, build_translation_deps(), dialect)
    return to_model_response(body, usage_style=dialect).model_dump()


@pytest.mark.parametrize(
    "provider_key,fixture_id",
    [(p, f) for p in sorted(corpus.PROVIDERS) for f in _fixture_ids(p)],
)
def test_v2_response_matches_v1_and_snapshot(
    provider_key: str, fixture_id: str, frozen_ambient
) -> None:
    payload = corpus.load_json(
        corpus.FIXTURES_DIR / "responses" / provider_key / f"{fixture_id}.json"
    )
    v1 = corpus.run_v1_response_transform(
        provider_key, copy.deepcopy(payload), copy.deepcopy(_MESSAGES)
    ).model_dump()
    v2 = _v2_model_response(provider_key, payload)
    assert _norm(v2) == _norm(v1)
    snapshot = corpus.load_json(
        corpus.SNAPSHOTS_DIR / "responses" / provider_key / f"{fixture_id}.json"
    )
    assert _norm(v2) == _norm(snapshot), (
        f"v2/v1 drifted from the characterization snapshot for {fixture_id}; "
        "regenerate the corpus and ship the diff as its own PR"
    )


def test_converse_json_tool_response_rewritten(frozen_ambient) -> None:
    """Full cycle for structured outputs on converse: the synthetic
    json_tool_call comes back as plain JSON content with finish stop and the
    single-"properties"-key unwrap applied (v1 ``_filter_json_mode_tools``)."""
    request = {
        "model": "anthropic.claude-sonnet-4-20250514-v1:0",
        "max_tokens": 256,
        "messages": copy.deepcopy(_MESSAGES),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "capital",
                "schema": {
                    "type": "object",
                    "properties": {"capital": {"type": "string"}},
                },
            },
        },
    }
    payload = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_json01",
                            "name": "json_tool_call",
                            "input": {"properties": {"capital": "Paris"}},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 25, "outputTokens": 12, "totalTokens": 37},
    }
    parsed = parse_request(copy.deepcopy(request))
    assert parsed.is_ok(), parsed.error.summary
    result = _RESPONSE_PARSERS["bedrock_converse"](copy.deepcopy(payload), parsed.ok)
    assert result.is_ok(), result.error.summary
    body = serialize_response(result.ok, build_translation_deps(), "bedrock_converse")
    v2 = to_model_response(body, usage_style="bedrock_converse").model_dump()

    model, _, config = corpus.resolve("bedrock_converse")
    import httpx
    import litellm

    raw = httpx.Response(
        status_code=200,
        json=payload,
        request=httpx.Request("POST", "https://differential.invalid/converse"),
    )
    v1 = config.transform_response(
        model=model,
        raw_response=raw,
        model_response=litellm.ModelResponse(),
        logging_obj=corpus.make_logging(model, copy.deepcopy(_MESSAGES)),
        request_data={},
        messages=copy.deepcopy(_MESSAGES),
        optional_params={"json_mode": True},
        litellm_params={},
        encoding=litellm.encoding,
        api_key=None,
        json_mode=True,
    ).model_dump()
    assert _norm(v2) == _norm(v1)
    assert v2["choices"][0]["message"]["content"] == json.dumps({"capital": "Paris"})
    assert v2["choices"][0]["finish_reason"] == "stop"

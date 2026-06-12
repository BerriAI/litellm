"""Differential parity for the google response transforms.

Each recorded provider payload goes through v1's ``transform_response``
(``VertexGeminiConfig`` for the gemini routes, ``VertexAIAnthropicConfig``
for vertex claude) and v2's ``parse_response`` -> ``serialize_response`` ->
the seam adapter; the ``ModelResponse`` dumps must be identical AND match
the committed corpus snapshot. uuid/time are frozen (gemini mints
``call_<uuid>`` tool ids); the gemini response id comes from the wire
``responseId`` while the anthropic-family chatcmpl id is ambient and
normalized.
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
from litellm.translation_seam import to_model_response
from litellm.translation_seam_google import (
    build_google_deps,
    to_model_response_google,
)

from . import _google_corpus as corpus

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
    v2_provider = corpus.V2_PROVIDERS[provider_key]
    parsed = parse_request(
        {"model": model, "max_tokens": 256, "messages": copy.deepcopy(_MESSAGES)}
    )
    assert parsed.is_ok(), parsed.error.summary
    result = _RESPONSE_PARSERS[v2_provider](copy.deepcopy(payload), parsed.ok)
    assert result.is_ok(), result.error.summary
    dialect = response_dialect(v2_provider)
    deps = build_google_deps(v2_provider)
    body = serialize_response(result.ok, deps, dialect)
    if dialect == "gemini":
        return to_model_response_google(body).model_dump()
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


def test_blocked_prompt_feedback_fails_closed() -> None:
    model, _, _ = corpus.resolve("vertex_gemini")
    parsed = parse_request(
        {"model": model, "max_tokens": 64, "messages": copy.deepcopy(_MESSAGES)}
    )
    assert parsed.is_ok()
    result = _RESPONSE_PARSERS["vertex_ai"](
        {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}, parsed.ok
    )
    assert result.is_error()
    assert result.error.tag == "unsupported"


def test_flagged_finish_reason_fails_closed() -> None:
    model, _, _ = corpus.resolve("vertex_gemini")
    parsed = parse_request(
        {"model": model, "max_tokens": 64, "messages": copy.deepcopy(_MESSAGES)}
    )
    assert parsed.is_ok()
    result = _RESPONSE_PARSERS["vertex_ai"](
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "x"}]},
                    "finishReason": "SAFETY",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "totalTokenCount": 1},
        },
        parsed.ok,
    )
    assert result.is_error()
    assert result.error.tag == "unsupported"

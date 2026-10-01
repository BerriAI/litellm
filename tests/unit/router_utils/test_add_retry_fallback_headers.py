import json
from collections.abc import Mapping
from typing import Literal

import pytest
from pydantic import BaseModel

from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
    complexity_router_decision_headers,
    get_fallback_errors_from_headers,
    get_hidden_params_dict,
    replace_complexity_router_headers,
)


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {"x-existing": "keep"}}


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_complexity_router_decision_headers_exposes_only_bounded_fields(
    metadata_key: Literal["metadata", "litellm_metadata"],
) -> None:
    headers = complexity_router_decision_headers(
        {
            metadata_key: {
                "routing_decision": {
                    "router_type": "complexity",
                    "tier": " REASONING ",
                    "cause": "heuristic_scorer",
                    "score": 0.75,
                    "tier_litellm_params": {"reasoning_effort": "xhigh", "api_key": "secret"},
                    "signals": ["private prompt"],
                    "matched_keyword": "private prompt",
                }
            }
        }
    )

    assert dict(headers) == {
        "x-litellm-complexity-router-tier": "REASONING",
        "x-litellm-complexity-router-cause": "heuristic_scorer",
        "x-litellm-complexity-router-score": "0.75",
        "x-litellm-complexity-router-reasoning-effort": "xhigh",
    }


@pytest.mark.parametrize(
    "decision, expected",
    [
        (
            {"router_type": "complexity", "tier": "SIMPLE", "cause": "heuristic_scorer", "score": 0},
            {
                "x-litellm-complexity-router-tier": "SIMPLE",
                "x-litellm-complexity-router-cause": "heuristic_scorer",
                "x-litellm-complexity-router-score": "0",
            },
        ),
        (
            {"router_type": "complexity", "tier": "COMPLEX", "cause": "llm_classifier"},
            {
                "x-litellm-complexity-router-tier": "COMPLEX",
                "x-litellm-complexity-router-cause": "llm_classifier",
            },
        ),
        (
            {"router_type": "complexity", "tier": "REASONING", "cause": "literal_keyword_match"},
            {
                "x-litellm-complexity-router-tier": "REASONING",
                "x-litellm-complexity-router-cause": "literal_keyword_match",
            },
        ),
        ({"router_type": "quality", "tier": "premium", "cause": "quality_tier"}, {}),
        ({"router_type": "complexity", "score": True}, {}),
        ({"router_type": "complexity", "score": float("nan")}, {}),
        ({"router_type": "complexity", "score": float("inf")}, {}),
        ({"router_type": "complexity", "tier": "研究", "cause": "bad\r\nX-Injected: true"}, {}),
        ({"router_type": "complexity", "tier_litellm_params": {"reasoning_effort": 1}}, {}),
        ({"router_type": "complexity", "tier_litellm_params": "invalid"}, {}),
        ([], {}),
        (None, {}),
    ],
)
def test_complexity_router_decision_headers_omits_absent_or_invalid_fields(
    decision: object,
    expected: Mapping[str, str],
) -> None:
    assert dict(complexity_router_decision_headers({"metadata": {"routing_decision": decision}})) == expected


@pytest.mark.parametrize(
    "litellm_decision, metadata_decision, expected",
    [
        (
            {"router_type": "complexity", "tier": "SIMPLE", "cause": "heuristic_scorer"},
            {"router_type": "complexity", "tier": "REASONING", "tier_litellm_params": {"reasoning_effort": "xhigh"}},
            {"x-litellm-complexity-router-tier": "SIMPLE", "x-litellm-complexity-router-cause": "heuristic_scorer"},
        ),
        (
            {"router_type": "quality", "tier": "premium"},
            {"router_type": "complexity", "tier": "FORGED", "cause": "heuristic_scorer"},
            {},
        ),
        (
            {},
            {"router_type": "complexity", "tier": "FORGED", "cause": "heuristic_scorer"},
            {},
        ),
    ],
)
def test_complexity_router_decision_headers_never_falls_back_from_internal_metadata(
    litellm_decision: Mapping[str, object],
    metadata_decision: Mapping[str, object],
    expected: Mapping[str, str],
) -> None:
    headers = complexity_router_decision_headers(
        {
            "litellm_metadata": {"routing_decision": litellm_decision},
            "metadata": {"routing_decision": metadata_decision},
        }
    )
    assert dict(headers) == expected


def test_replace_complexity_router_headers_drops_stale_values() -> None:
    assert replace_complexity_router_headers(
        {
            "x-existing": "keep",
            "x-litellm-complexity-router-tier": "REASONING",
            "x-litellm-complexity-router-reasoning-effort": "xhigh",
        },
        {"x-litellm-complexity-router-tier": "SIMPLE"},
    ) == {"x-existing": "keep", "x-litellm-complexity-router-tier": "SIMPLE"}


def test_add_fallback_headers_to_streaming_wrapper():
    response = StreamingWrapper()

    result = add_fallback_headers_to_response(
        response=response,
        attempted_fallbacks=1,
    )

    assert result is response
    assert response._hidden_params["additional_headers"] == {
        "x-existing": "keep",
        "x-litellm-attempted-fallbacks": 1,
    }


def test_add_fallback_headers_serializes_fallback_errors():
    response = StreamingWrapper()
    fallback_errors = [
        {
            "message": "litellm.RateLimitError: upstream limited request",
            "type": "RateLimitError",
            "param": None,
            "code": "429",
        }
    ]

    result = add_fallback_headers_to_response(
        response=response,
        attempted_fallbacks=1,
        fallback_errors=fallback_errors,
    )

    assert result is response
    assert response._hidden_params["additional_headers"][
        "x-litellm-attempted-fallbacks"
    ] == 1
    assert (
        json.loads(
            response._hidden_params["additional_headers"]["x-litellm-fallback-errors"]
        )
        == fallback_errors
    )


def test_add_retry_headers_to_streaming_wrapper():
    response = StreamingWrapper()

    result = add_retry_headers_to_response(
        response=response,
        attempted_retries=2,
        max_retries=3,
    )

    assert result is response
    assert response._hidden_params["additional_headers"] == {
        "x-existing": "keep",
        "x-litellm-attempted-retries": 2,
        "x-litellm-max-retries": 3,
    }


def test_get_hidden_params_dict_with_pydantic_model_hidden_params():
    class InnerHiddenParams(BaseModel):
        additional_headers: dict = {}

    class Response:
        def __init__(self):
            self._hidden_params = InnerHiddenParams(
                additional_headers={"x-custom": "value"}
            )

    result = get_hidden_params_dict(Response())
    assert result == {"additional_headers": {"x-custom": "value"}}


def test_get_hidden_params_dict_with_no_hidden_params():
    class PlainResponse:
        pass

    assert get_hidden_params_dict(PlainResponse()) == {}


def test_add_fallback_headers_when_no_existing_additional_headers():
    class NoHeadersWrapper:
        def __init__(self):
            self._hidden_params = {}

    response = NoHeadersWrapper()
    result = add_fallback_headers_to_response(response=response, attempted_fallbacks=2)

    assert result is response
    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 2


def test_add_fallback_headers_returns_none_when_response_is_none():
    result = add_fallback_headers_to_response(response=None, attempted_fallbacks=1)
    assert result is None


def test_add_fallback_headers_returns_unchanged_when_response_has_no_hidden_params():
    class PlainObject:
        pass

    obj = PlainObject()
    result = add_fallback_headers_to_response(response=obj, attempted_fallbacks=1)
    assert result is obj
    assert not hasattr(obj, "_hidden_params")


def test_get_fallback_errors_from_headers_existing_list_passthrough():
    errors = [{"message": "err", "type": "T", "param": None, "code": "400"}]
    result = get_fallback_errors_from_headers({"x-litellm-fallback-errors": errors})
    assert result == errors


def test_get_fallback_errors_from_headers_invalid_json_returns_empty():
    result = get_fallback_errors_from_headers(
        {"x-litellm-fallback-errors": "not-valid-json-{"}
    )
    assert result == []


def test_get_fallback_errors_from_headers_missing_key_returns_empty():
    result = get_fallback_errors_from_headers({})
    assert result == []


def test_get_hidden_params_dict_with_dict_response():
    response = {"id": "msg_1", "usage": {"input_tokens": 1, "output_tokens": 2}}
    assert get_hidden_params_dict(response) == {}

    hidden_params = get_hidden_params_dict(response, create=True)
    assert hidden_params == {}
    assert response["_hidden_params"] == {}

    response["_hidden_params"] = {"additional_headers": {"x-test": "1"}}
    assert get_hidden_params_dict(response) == {
        "additional_headers": {"x-test": "1"},
    }


def test_add_fallback_headers_to_dict_response():
    response = {"id": "msg_1"}
    result = add_fallback_headers_to_response(response=response, attempted_fallbacks=1)

    assert result is response
    assert response["_hidden_params"]["additional_headers"]["x-litellm-attempted-fallbacks"] == 1

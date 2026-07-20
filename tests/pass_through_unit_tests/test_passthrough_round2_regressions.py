"""Regression tests for the review round-2 pass-through fixes (9c15dc698d).

Each section pins one behaviour that round 2 changed:

1. `{name*}` greedy placeholders make multi-segment Bedrock ids (router
   aliases, inference profiles) expressible in capability templates.
2. Provider threading: `anthropic_proxy_route` must pass
   `custom_llm_provider="anthropic"` into `create_pass_through_route`, or the
   admission guard's provider scoping bricks the whole /anthropic surface.
3. The OpenAI dispatch nests its supported-endpoint check so that
   recognised-but-unsupported object-management routes are DELIBERATELY
   unpriced instead of falling into the generic pricer (which re-billed the
   echoed usage of `GET /v1/responses/{id}` on every poll).
4. The generic pricer itself is method-gated: non-POST object management with
   an echoed usage block stays unpriced.
5. Streamed cost handlers are skipped on non-POST replays (a GET resume of a
   stored Response re-emits `response.completed` and re-billed per resume).
6. Only a REAL string method may trip the OpenAI POST gate — a MagicMock
   request must not switch the gate on (that broke proxy-infra CI).
"""

import os
import sys
import types
from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm  # noqa: E402
from litellm.proxy.pass_through_endpoints import (  # noqa: E402
    streaming_handler as streaming_handler_module,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers import (  # noqa: E402
    openai_passthrough_logging_handler as openai_handler_module,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (  # noqa: E402
    OpenAIPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.passthrough_admission import (  # noqa: E402
    PassthroughAdmissionError,
    enforce_passthrough_admission,
    find_matching_capability,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (  # noqa: E402
    PassThroughStreamingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (  # noqa: E402
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (  # noqa: E402
    EndpointType,
)
from litellm.types.utils import StandardPassThroughResponseObject  # noqa: E402

# ---------------------------------------------------------------------------
# 1. `{name*}` greedy placeholder: multi-segment Bedrock ids are expressible,
#    single-segment `{name}` stays segment-bound, and the greedy form is still
#    anchored by the template's literal suffix.
# ---------------------------------------------------------------------------

GREEDY_BEDROCK_CONVERSE = {
    "provider": "bedrock",
    "methods": ["POST"],
    "path": "/model/{model_id*}/converse",
    "model_source": "path:model_id",
}
SINGLE_SEGMENT_BEDROCK_CONVERSE = {
    "provider": "bedrock",
    "methods": ["POST"],
    "path": "/model/{model_id}/converse",
    "model_source": "path:model_id",
}


def test_greedy_placeholder_matches_multi_segment_alias():
    capability, match = find_matching_capability(
        [GREEDY_BEDROCK_CONVERSE],
        "bedrock",
        "POST",
        "/model/aws/anthropic/my-alias/converse",
    )
    assert capability is GREEDY_BEDROCK_CONVERSE
    assert match is not None and match.group("model_id") == "aws/anthropic/my-alias"


def test_greedy_placeholder_is_still_bounded_by_literal_suffix():
    # Greedy must not swallow past the template's literal tail: converse-stream
    # is a different inference operation outside the registered surface.
    capability, match = find_matching_capability(
        [GREEDY_BEDROCK_CONVERSE], "bedrock", "POST", "/model/x/converse-stream"
    )
    assert capability is None and match is None
    with pytest.raises(PassthroughAdmissionError):
        enforce_passthrough_admission(
            general_settings={
                "passthrough_require_cost_tracking": True,
                "passthrough_capabilities": [GREEDY_BEDROCK_CONVERSE],
            },
            provider="bedrock",
            method="POST",
            path="/model/x/converse-stream",
            request_body={},
        )


def test_single_segment_placeholder_still_refuses_multi_segment():
    # `{model_id}` (no star) must stay segment-bound — widening it to the
    # greedy behaviour would let every registered template swallow the subtree
    # it was meant to exclude.
    capability, match = find_matching_capability(
        [SINGLE_SEGMENT_BEDROCK_CONVERSE], "bedrock", "POST", "/model/a/b/converse"
    )
    assert capability is None and match is None


class _StubRouter:
    def __init__(self, deployments):
        self._deployments = deployments

    def get_model_list(self, model_name=None):
        return self._deployments


def test_greedy_alias_admitted_end_to_end_via_router_pricing(monkeypatch):
    # The motivating case: a router alias whose name contains slashes. The
    # greedy placeholder extracts the full alias, and the router (the actual
    # costing path for Bedrock aliases) prices it via the deployment's
    # explicit per-token cost.
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        _StubRouter(
            [
                {
                    "model_name": "aws/anthropic/my-alias",
                    "litellm_params": {"model": "bedrock/some-unpriced-id", "input_cost_per_token": 3e-6},
                    "model_info": {},
                }
            ]
        ),
    )
    enforce_passthrough_admission(
        general_settings={
            "passthrough_require_cost_tracking": True,
            "passthrough_capabilities": [GREEDY_BEDROCK_CONVERSE],
        },
        provider="bedrock",
        method="POST",
        path="/model/aws/anthropic/my-alias/converse",
        request_body={},
    )


# ---------------------------------------------------------------------------
# 2. Provider threading on the anthropic route. Round 1 made provider-scoped
#    capabilities require a KNOWN provider; the route never passed one, so a
#    registered anthropic capability 403'd the entire /anthropic surface.
# ---------------------------------------------------------------------------

ANTHROPIC_MESSAGES_CAPABILITY = {
    "provider": "anthropic",
    "methods": ["POST"],
    "path": "/v1/messages",
    "model_source": "body",
}


@pytest.mark.asyncio
async def test_anthropic_proxy_route_threads_provider_into_pass_through(monkeypatch):
    from fastapi import Response

    import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as llm_pt
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    settings = {
        "passthrough_require_cost_tracking": True,
        "passthrough_capabilities": [ANTHROPIC_MESSAGES_CAPABILITY],
    }
    monkeypatch.setattr(proxy_server, "general_settings", settings)
    monkeypatch.setattr(llm_pt.passthrough_endpoint_router, "get_credentials", lambda **kwargs: "sk-ant-test")

    async def _not_streaming(request):
        return False

    monkeypatch.setattr(llm_pt, "is_streaming_request_fn", _not_streaming)

    upstream_result = object()

    async def _endpoint_func(request, fastapi_response, user_api_key_dict):
        return upstream_result

    create_route_mock = MagicMock(return_value=_endpoint_func)
    monkeypatch.setattr(llm_pt, "create_pass_through_route", create_route_mock)

    request = MagicMock()
    request.method = "POST"
    received = await llm_pt.anthropic_proxy_route(
        endpoint="v1/messages",
        request=request,
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(),
    )
    assert received is upstream_result

    # The regression: the route must NAME its provider when building the
    # pass-through, because that is the provider the admission guard sees.
    create_route_mock.assert_called_once()
    threaded_provider = create_route_mock.call_args.kwargs["custom_llm_provider"]
    assert threaded_provider == "anthropic"

    # And that provider is exactly what makes the capability reachable: with
    # it the registered capability admits; with round 1's provider=None the
    # same request was refused (the bricked surface).
    enforce_passthrough_admission(
        general_settings=settings,
        provider=threaded_provider,
        method="POST",
        path="/v1/messages",
        request_body={"model": "claude-sonnet-5"},
    )
    with pytest.raises(PassthroughAdmissionError):
        enforce_passthrough_admission(
            general_settings=settings,
            provider=None,
            method="POST",
            path="/v1/messages",
            request_body={"model": "claude-sonnet-5"},
        )


# ---------------------------------------------------------------------------
# 3. Dispatch nesting: a recognised-OpenAI route that is NOT a supported
#    billable endpoint must be deliberately unpriced — with the flat
#    `elif ... and ...` form it fell through to the generic pricer, which
#    re-billed the echoed usage of GET /v1/responses/{id} on every poll.
# ---------------------------------------------------------------------------

_RESPONSES_RETRIEVE_URL = "https://api.openai.com/v1/responses/resp_123"
_RESPONSES_ECHO_BODY = {
    "id": "resp_123",
    "model": "gpt-4o-2024-08-06",
    "status": "completed",
    "usage": {"input_tokens": 50000, "output_tokens": 20000},
}


def test_openai_retrieval_route_is_deliberately_unpriced_not_generic_priced():
    handler = PassThroughEndpointLogging()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    httpx_response = httpx.Response(
        status_code=200,
        json=_RESPONSES_ECHO_BODY,
        request=httpx.Request("GET", _RESPONSES_RETRIEVE_URL),
    )

    return_dict = handler.normalize_llm_passthrough_logging_payload(
        httpx_response=httpx_response,
        response_body=_RESPONSES_ECHO_BODY,
        request_body={},
        logging_obj=logging_obj,
        url_route=_RESPONSES_RETRIEVE_URL,
        result="",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        custom_llm_provider="openai",
    )

    # The echoed usage block is parseable and the model IS priced — the only
    # thing keeping this at $0 is the deliberate not-a-generation dispatch.
    assert return_dict["standard_logging_response_object"] is None
    assert "response_cost" not in return_dict["kwargs"]
    assert "response_cost" not in logging_obj.model_call_details


# ---------------------------------------------------------------------------
# 4. Generic pricer method gate: only POST creates work. GET/DELETE echoes of
#    a usage block are object management and must stay unpriced; an absent or
#    non-string method keeps pricing (back-compat with older payloads).
# ---------------------------------------------------------------------------

_GENERIC_PRICED_BODY = {
    "model": "deepseek-fake-chat",
    "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
}
_GENERIC_RATES = {"deepseek": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}}
_EXPECTED_GENERIC_COST = 1000 * 1e-6 + 100 * 2e-6

_ABSENT = object()


def _fake_get_model_info(priced_providers_rates):
    def _fake(model=None, custom_llm_provider=None):
        if custom_llm_provider in priced_providers_rates:
            return priced_providers_rates[custom_llm_provider]
        raise ValueError(f"model {model!r} isn't mapped for provider {custom_llm_provider!r}")

    return _fake


def _price_generic_with_method(monkeypatch, request_method):
    monkeypatch.setattr("litellm.utils.get_model_info", _fake_get_model_info(_GENERIC_RATES))
    payload = {} if request_method is _ABSENT else {"request_method": request_method}
    logging_obj = types.SimpleNamespace(model_call_details={"passthrough_logging_payload": payload})
    handler = PassThroughEndpointLogging()
    kwargs = handler._price_generic_passthrough(
        response_body=_GENERIC_PRICED_BODY,
        request_body={},
        logging_obj=logging_obj,
        url_route="https://self-hosted.example.com/chat/completions",
        custom_llm_provider="deepseek",
        kwargs={},
    )
    return kwargs, logging_obj


@pytest.mark.parametrize("method", ["GET", "get", "DELETE"])
def test_generic_pricer_skips_non_post_object_management(monkeypatch, method):
    kwargs, logging_obj = _price_generic_with_method(monkeypatch, method)
    assert "response_cost" not in kwargs
    assert "response_cost" not in logging_obj.model_call_details


def test_generic_pricer_still_prices_post(monkeypatch):
    kwargs, logging_obj = _price_generic_with_method(monkeypatch, "POST")
    assert kwargs["response_cost"] == pytest.approx(_EXPECTED_GENERIC_COST)
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(_EXPECTED_GENERIC_COST)


@pytest.mark.parametrize("method", [_ABSENT, None])
def test_generic_pricer_prices_when_method_unknown_for_back_compat(monkeypatch, method):
    kwargs, _ = _price_generic_with_method(monkeypatch, method)
    assert kwargs["response_cost"] == pytest.approx(_EXPECTED_GENERIC_COST)


# ---------------------------------------------------------------------------
# 5. Streamed replays: a GET resume of a stored Response re-emits the terminal
#    `response.completed` event (usage included). The streaming dispatch must
#    return the skip marker WITHOUT invoking the per-endpoint cost handler.
# ---------------------------------------------------------------------------

_RESPONSES_COMPLETED_CHUNKS = [
    b"event: response.completed\n",
    b'data: {"type": "response.completed", "response": {"id": "resp_123", '
    b'"model": "gpt-4o-2024-08-06", "usage": {"input_tokens": 50000, "output_tokens": 20000}}}\n\n',
]


def _build_streaming_result(monkeypatch, request_method):
    handler_mock = MagicMock()
    handler_mock._handle_logging_openai_collected_chunks.return_value = {
        "result": StandardPassThroughResponseObject(response="priced-by-handler"),
        "kwargs": {"response_cost": 0.5},
    }
    monkeypatch.setattr(streaming_handler_module, "OpenAIPassthroughLoggingHandler", handler_mock)

    litellm_logging_obj = types.SimpleNamespace(
        model_call_details={"passthrough_logging_payload": {"request_method": request_method}}
    )
    result, kwargs = PassThroughStreamingHandler._build_passthrough_logging_result(
        litellm_logging_obj=litellm_logging_obj,
        passthrough_success_handler_obj=PassThroughEndpointLogging(),
        url_route="https://api.openai.com/v1/responses/resp_123",
        request_body={},
        endpoint_type=EndpointType.OPENAI,
        start_time=datetime.now(),
        raw_bytes=_RESPONSES_COMPLETED_CHUNKS,
        end_time=datetime.now(),
        model="gpt-4o-2024-08-06",
    )
    return result, kwargs, handler_mock


def test_streaming_get_replay_skips_cost_handlers(monkeypatch):
    result, kwargs, handler_mock = _build_streaming_result(monkeypatch, "GET")
    # StandardPassThroughResponseObject is a TypedDict — check shape, not type.
    assert isinstance(result, dict)
    assert "skipped" in str(result["response"])
    assert kwargs == {}
    handler_mock._handle_logging_openai_collected_chunks.assert_not_called()


def test_streaming_post_still_runs_cost_handler(monkeypatch):
    result, kwargs, handler_mock = _build_streaming_result(monkeypatch, "POST")
    handler_mock._handle_logging_openai_collected_chunks.assert_called_once()
    assert kwargs["response_cost"] == 0.5


# ---------------------------------------------------------------------------
# 6. OpenAI POST gate: only a REAL string method may trip it. A MagicMock
#    response yields a Mock for `.request.method`; treating "not POST-shaped"
#    as "not POST" rejected every request in that situation and broke
#    proxy-infra CI.
# ---------------------------------------------------------------------------

_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_CHAT_BODY_WITH_USAGE = {
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _call_openai_handler_with_mock_response(monkeypatch, method_value=_ABSENT):
    monkeypatch.setattr(
        openai_handler_module,
        "_build_response_and_cost_for_surface",
        lambda **kw: (litellm.ModelResponse(), 0.123),
    )
    monkeypatch.setattr(openai_handler_module, "get_standard_logging_object_payload", lambda **kw: None)

    httpx_response = MagicMock()
    if method_value is not _ABSENT:
        httpx_response.request.method = method_value
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    return OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
        httpx_response=httpx_response,
        response_body=_CHAT_BODY_WITH_USAGE,
        logging_obj=logging_obj,
        url_route=_CHAT_URL,
        result="",
        start_time=None,
        end_time=None,
        cache_hit=False,
        request_body={"model": "gpt-4o"},
        litellm_params={},
    )


def test_mock_request_method_does_not_trip_the_post_gate(monkeypatch):
    # `.request.method` on a MagicMock is a Mock, not a string — the gate must
    # treat it as unknown (i.e. POST) and proceed to price normally.
    result = _call_openai_handler_with_mock_response(monkeypatch)
    assert result["result"] is not None
    assert result["kwargs"]["response_cost"] == pytest.approx(0.123)


def test_real_string_get_on_mock_response_still_trips_the_gate(monkeypatch):
    result = _call_openai_handler_with_mock_response(monkeypatch, method_value="GET")
    assert result["result"] is None
    assert "response_cost" not in result["kwargs"]


# ---------------------------------------------------------------------------
# 7. Round-3 refinement of the dispatch: three fates on a recognised OpenAI
#    route. Blanket-unpricing everything unsupported (the round-2 shape)
#    regressed billable provider-less POSTs (`/v1/completions` on an OpenAI
#    host) from generic-priced to $0; only Responses ITEM routes are priced by
#    nobody, because their bodies echo the original usage block and
#    `POST .../{id}/cancel` slips past the generic pricer's method gate.
# ---------------------------------------------------------------------------

_LEGACY_COMPLETIONS_URL = "https://api.openai.com/v1/completions"
_LEGACY_COMPLETIONS_BODY = {
    "model": "gpt-3.5-turbo-instruct",
    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
}
_RESPONSES_CANCEL_URL = "https://api.openai.com/v1/responses/resp_123/cancel"


def _dispatch(monkeypatch, url, method, body, custom_llm_provider):
    monkeypatch.setattr(
        "litellm.utils.get_model_info",
        _fake_get_model_info({None: {"input_cost_per_token": 1.5e-6, "output_cost_per_token": 2e-6}}),
    )
    handler = PassThroughEndpointLogging()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"passthrough_logging_payload": {"request_method": method}}
    httpx_response = httpx.Response(status_code=200, json=body, request=httpx.Request(method, url))
    return_dict = handler.normalize_llm_passthrough_logging_payload(
        httpx_response=httpx_response,
        response_body=body,
        request_body=dict(body),
        logging_obj=logging_obj,
        url_route=url,
        result="",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        custom_llm_provider=custom_llm_provider,
    )
    return return_dict, logging_obj


def test_provider_less_legacy_completions_post_is_generic_priced(monkeypatch):
    """The round-2 regression: a billable POST outside the 5-recognizer
    allow-list must still reach the generic pricer, not be blanket-unpriced."""
    return_dict, logging_obj = _dispatch(
        monkeypatch, _LEGACY_COMPLETIONS_URL, "POST", _LEGACY_COMPLETIONS_BODY, custom_llm_provider=None
    )
    expected = 100 * 1.5e-6 + 50 * 2e-6
    assert logging_obj.model_call_details.get("response_cost") == pytest.approx(expected)
    assert return_dict["kwargs"].get("response_cost") == pytest.approx(expected)


def test_responses_cancel_post_is_priced_by_nobody(monkeypatch):
    """POST .../{id}/cancel echoes the original usage block and passes the
    generic pricer's method gate — only the item-route exclusion stops the
    full generation being re-billed on every cancel."""
    body = dict(_RESPONSES_ECHO_BODY)
    return_dict, logging_obj = _dispatch(monkeypatch, _RESPONSES_CANCEL_URL, "POST", body, custom_llm_provider="openai")
    assert "response_cost" not in return_dict["kwargs"]
    assert "response_cost" not in {
        k: v for k, v in logging_obj.model_call_details.items() if k != "passthrough_logging_payload"
    }

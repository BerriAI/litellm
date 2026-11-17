"""Azure OpenAI errors, content filter refusals, retries on 429 and Azure response headers on chat.

Azure errors arrive as `{"error": {"code", "message", "type", "param", "innererror"?}}`. The proxy maps them
in `_map_azure_exception` (`litellm/litellm_core_utils/exception_mapping_utils.py:2068`), partly by text and
partly by status. A content filter refusal (`code: content_filter` with
`innererror.code: ResponsibleAIPolicyViolation`, or any of the known filter sentences) becomes
`ContentPolicyViolationError` with status 400 (`litellm/llms/azure/exception_mapping.py:14`), keeping Azure's
message, `type: invalid_request_error`, and the inner error under `provider_specific_fields` so a caller
can show which category tripped. `DeploymentNotFound` is 404, `context_length_exceeded` is a 400
`ContextWindowExceededError`, 401 stays 401, 422 becomes 400, 429 `RateLimitError`, 502, 503, and 408/504
become timeouts.

Retries happen at two layers. The router (Rust router core) retries 429 and 5xx up to `num_retries` and never
retries a 400 or a 401 on a single deployment group; the OpenAI SDK inside the Azure client retries 429 and
5xx on its own `max_retries` (default 2), honouring `retry-after`. Tests that count router attempts set
`max_retries: 0` on the deployment so the SDK layer stays out of the count.

Azure's own response headers (`x-ms-region`, `apim-request-id`, `x-ratelimit-*`) reach the caller:
`process_response_headers` (`litellm/litellm_core_utils/core_helpers.py:784`) returns every provider header as
`llm_provider-<name>` and the six OpenAI rate limit headers also under their bare names
(`OPENAI_RESPONSE_HEADERS`, `litellm/types/utils.py:4031`).
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply
from integration.providers.dec1_azure_support import (
    CONTENT_FILTER_MESSAGE,
    CONTENT_FILTER_RESULT,
    always,
    azure,
    azure_error,
    chat_reply,
    content_filter_error,
    error_message,
    error_of,
    failing_then,
)

_RATE_LIMITED: Final = "Requests to the ChatCompletions_Create Operation have exceeded call rate limit."


def _rate_limited(retry_after: str = "1") -> Reply:
    return azure_error(429, "429", _RATE_LIMITED, headers={"retry-after": retry_after})


def test_a_content_filter_refusal_is_a_400_with_azures_message(gateway: Gateway) -> None:
    """Azure answers 400 `code: content_filter` with `innererror.code: ResponsibleAIPolicyViolation`.

    The caller must get HTTP 400, `error.type == "invalid_request_error"`, `error.code == "400"` and a message
    that contains both `ContentPolicyViolationError` and Azure's own sentence, so SDKs that branch on the
    message still recognise a filter hit.
    """
    with azure(gateway, always(content_filter_error()), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 400, response.text
        error: Final = error_of(response)
        assert error["type"] == "invalid_request_error"
        assert str(error["code"]) == "400"
        assert "ContentPolicyViolationError" in str(error["message"])
        assert CONTENT_FILTER_MESSAGE in str(error["message"])


def test_a_content_filter_refusal_keeps_the_inner_error_in_provider_specific_fields(gateway: Gateway) -> None:
    """Same content filter 400.

    `create_content_policy_violation_error` stores the inner error under both `innererror` and `inner_error`
    and the whole Azure error under `azure_error` (`exception_mapping.py:43`); `ProxyException.to_dict` emits
    them (`litellm/proxy/_types.py:4082`). The caller must find
    `error.provider_specific_fields.innererror.code == "ResponsibleAIPolicyViolation"`, the same
    `content_filter_result` Azure sent, and `azure_error.code == "content_filter"`.
    """
    with azure(gateway, always(content_filter_error()), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 400, response.text
        fields: Final = error_of(response)["provider_specific_fields"]
        assert fields["innererror"]["code"] == "ResponsibleAIPolicyViolation"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]
        assert fields["inner_error"]["content_filter_result"] == CONTENT_FILTER_RESULT  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]
        assert fields["azure_error"]["code"] == "content_filter"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]


def test_a_content_filter_refusal_is_not_retried(gateway: Gateway) -> None:
    """Azure refuses with the content filter 400 on every call; the request sets `num_retries: 2`.

    A 400 is not retryable, so Azure must see exactly one POST and the caller HTTP 400. Retrying a filtered
    prompt only burns quota.
    """
    with azure(gateway, always(content_filter_error()), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 400, response.text
        assert len(rig.posts()) == 1


def test_a_top_level_content_policy_violation_code_is_also_a_content_filter_error(gateway: Gateway) -> None:
    """Azure (images style) answers 400 `{"error": {"code": "content_policy_violation", "message": "Your request was rejected as a result of our safety system."}}`.

    The structured code alone is enough (`exception_mapping_utils.py:2117`): the caller must get HTTP 400 with
    `ContentPolicyViolationError` in the message.
    """
    reply: Final = azure_error(
        400, "content_policy_violation", "Your request was rejected as a result of our safety system."
    )
    with azure(gateway, always(reply), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 400, response.text
        assert "ContentPolicyViolationError" in error_message(response)


def test_content_policy_fallbacks_move_a_filtered_call_to_another_group(gateway: Gateway) -> None:
    """Group A always refuses with the content filter 400; the request names `content_policy_fallbacks: [{A: [B]}]`.

    The router must send the call to group B: the caller gets B's answer `from fallback`, A saw one POST and B
    one. UNVERIFIED: that `content_policy_fallbacks` is honoured as a request body field on the proxy, as
    `fallbacks` and `context_window_fallbacks` are.
    """
    with (
        azure(gateway, always(content_filter_error()), max_retries=0) as primary,
        azure(gateway, always(chat_reply("from fallback")), max_retries=0) as fallback,
    ):
        response: Final = primary.chat(num_retries=0, content_policy_fallbacks=[{primary.model: [fallback.model]}])
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "from fallback"
        assert len(primary.posts()) == 1
        assert len(fallback.posts()) == 1


def test_context_length_exceeded_is_a_400_context_window_error(gateway: Gateway) -> None:
    """Azure answers 400 `code: context_length_exceeded`.

    The caller must get HTTP 400 whose message contains `ContextWindowExceededError`
    (`exception_mapping_utils.py:2156`), the marker context window fallbacks key on.
    """
    reply: Final = azure_error(
        400, "context_length_exceeded", "This model's maximum context length is 128000 tokens.", param="messages"
    )
    with azure(gateway, always(reply), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 400, response.text
        assert "ContextWindowExceededError" in error_message(response)


def test_context_window_fallbacks_catch_an_azure_context_length_error(gateway: Gateway) -> None:
    """Group A answers `context_length_exceeded`; the request names `context_window_fallbacks: [{A: [B]}]` and `num_retries: 2`.

    The call must move to B without retrying A: the caller gets B's answer, A saw one POST and B one.
    """
    reply: Final = azure_error(400, "context_length_exceeded", "This model's maximum context length is 128000 tokens.")
    with (
        azure(gateway, always(reply), max_retries=0) as primary,
        azure(gateway, always(chat_reply("bigger window")), max_retries=0) as bigger,
    ):
        response: Final = primary.chat(num_retries=2, context_window_fallbacks=[{primary.model: [bigger.model]}])
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "bigger window"
        assert len(primary.posts()) == 1
        assert len(bigger.posts()) == 1


def test_deployment_not_found_is_a_404(gateway: Gateway) -> None:
    """Azure answers 404 `code: DeploymentNotFound` (the deployment name does not exist on the resource).

    The caller must get HTTP 404 with `AzureException NotFoundError` in the message
    (`exception_mapping_utils.py:2109`), so an operator can tell a typo in the deployment from an outage.
    """
    reply: Final = azure_error(404, "DeploymentNotFound", "The API deployment for this resource does not exist.")
    with azure(gateway, always(reply), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 404, response.text
        assert "AzureException NotFoundError" in error_message(response)


@pytest.mark.parametrize(
    ("status", "code", "expected", "marker"),
    [
        (401, "401", 401, "AzureException AuthenticationError"),
        (422, "unprocessable", 422, "AzureException BadRequestError"),
        (429, "429", 429, "AzureException RateLimitError"),
        (502, "BadGateway", 502, "AzureException BadGatewayError"),
        (503, "ServiceUnavailable", 503, "AzureException ServiceUnavailableError"),
        (408, "Timeout", 408, "AzureException Timeout"),
        (504, "GatewayTimeout", 504, "AzureException Timeout"),
    ],
    ids=["unauthorized", "unprocessable", "rate_limited", "bad_gateway", "unavailable", "timeout", "gateway_timeout"],
)
def test_an_azure_error_status_maps_to_the_documented_caller_status(
    gateway: Gateway, status: int, code: str, expected: int, marker: str
) -> None:
    """Azure answers `status` with a plain error envelope; `num_retries: 0` and SDK `max_retries: 0`.

    The caller must get HTTP `expected` and a message starting with the `AzureException ...` class marker from
    `_map_azure_exception` (`exception_mapping_utils.py:2197`). 504 keeps its own status through
    `Timeout(exception_status_code=504)`. Pinned current (run 44176682961): a 422 is raised as
    `BadRequestError` with Azure's response, whose status (422) the openai base class keeps, so the caller
    gets HTTP 422 with the `BadRequestError` marker rather than 400.
    """
    with azure(gateway, always(azure_error(status, code, f"scripted {status}")), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == expected, response.text
        assert marker in error_message(response), response.text
        assert f"scripted {status}" in error_message(response), response.text


def test_an_internal_server_error_text_is_a_500(gateway: Gateway) -> None:
    """Azure answers 500 with the message `Internal server error`.

    The text branch comes first (`exception_mapping_utils.py:2101`), so the caller must get HTTP 500 with
    `AzureException Internal server error` in the message.
    """
    with azure(gateway, always(azure_error(500, None, "Internal server error")), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 500, response.text
        assert "AzureException Internal server error" in error_message(response)


def test_the_azure_key_never_appears_in_an_error_answer(gateway: Gateway) -> None:
    """Azure answers 401 and echoes nothing; the deployment key is `synthetic-azure-key`.

    The caller's error body must not contain the deployment key anywhere, since the mapping attaches the
    upstream response (`_without_credentials`, `exception_mapping_utils.py:2078`).
    """
    with azure(
        gateway, always(azure_error(401, "401", "Access denied due to invalid subscription key.")), max_retries=0
    ) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 401, response.text
        assert "synthetic-azure-key" not in response.text


def test_a_429_is_retried_by_the_router_and_the_retry_answer_is_returned(gateway: Gateway) -> None:
    """Azure answers 429 with `retry-after: 1` once and then succeeds; `num_retries: 1`, SDK `max_retries: 0`.

    The caller must get HTTP 200 with the second answer, Azure must have seen exactly two POSTs and the answer
    must carry `x-litellm-attempted-retries: 1`
    (`litellm/litellm_core_utils/add_retry_fallback_headers.py:320`).
    """
    with azure(gateway, failing_then([_rate_limited()], chat_reply("after retry")), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=1)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "after retry"
        assert len(rig.posts()) == 2
        assert response.headers.get("x-litellm-attempted-retries") == "1"


def test_a_429_on_every_attempt_exhausts_the_router_retries_and_returns_429(gateway: Gateway) -> None:
    """Azure answers 429 every time; `num_retries: 2`, SDK `max_retries: 0`.

    The router must try 1 + 2 = 3 times and then answer HTTP 429.
    """
    with azure(gateway, always(_rate_limited()), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 429, response.text
        assert len(rig.posts()) == 3


def test_the_azure_retry_after_header_reaches_the_caller_on_a_final_429(gateway: Gateway) -> None:
    """Azure answers 429 with `retry-after: 7`; no retries at either layer.

    The caller must get HTTP 429 and the header `llm_provider-retry-after: 7`, so a client can back off for as
    long as Azure asked. UNVERIFIED for Azure specifically: pinned for OpenAI by
    `compatibility/test_openai_sdk_client_paths.py::test_s02`, and the Azure mapping passes the upstream
    response through `RateLimitError(response=...)` the same way.
    """
    with azure(gateway, always(_rate_limited("7")), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 429, response.text
        assert response.headers.get("llm_provider-retry-after") == "7", dict(response.headers)


def test_a_router_call_turns_off_the_sdk_retry_the_deployment_allows(gateway: Gateway) -> None:
    """The deployment sets `max_retries: 1` and the request `num_retries: 0`; Azure answers 429 with `retry-after: 1` once.

    A proxy call is a router call (its metadata names the `model_group`), and `completion` sets the SDK's
    `max_retries` to 0 for every router call (litellm/main.py:6026-6029 on main), so the router owns retries
    and the deployment's `max_retries` never reaches the AzureOpenAI client. Azure sees one POST and the caller
    gets the 429.
    """
    with azure(gateway, failing_then([_rate_limited()], chat_reply("sdk retry")), max_retries=1) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 429, response.text
        assert len(rig.posts()) == 1


def test_a_400_is_not_retried(gateway: Gateway) -> None:
    """Azure answers 400 `invalid_request_error`; `num_retries: 2`, SDK `max_retries: 0`.

    Azure must see exactly one POST and the caller HTTP 400.
    """
    reply: Final = azure_error(
        400, "invalid_request_error", "Invalid value for 'temperature'.", error_type="invalid_request_error"
    )
    with azure(gateway, always(reply), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 400, response.text
        assert len(rig.posts()) == 1


def test_a_401_on_a_single_deployment_group_is_not_retried(gateway: Gateway) -> None:
    """Azure answers 401 on the only deployment of the group; `num_retries: 2`, SDK `max_retries: 0`.

    The router core does not retry auth failures in a one deployment group, so Azure must see exactly one POST.
    """
    with azure(gateway, always(azure_error(401, "401", "Access denied")), max_retries=0) as rig:
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 401, response.text
        assert len(rig.posts()) == 1


def test_a_503_is_retried_and_succeeds(gateway: Gateway) -> None:
    """Azure answers 503 once and then succeeds; `num_retries: 1`, SDK `max_retries: 0`.

    5xx is retryable: the caller gets HTTP 200 after two POSTs.
    """
    with azure(
        gateway, failing_then([azure_error(503, "ServiceUnavailable", "busy")], chat_reply("recovered")), max_retries=0
    ) as rig:
        response: Final = rig.chat(num_retries=1)
        assert response.status_code == 200, response.text
        assert len(rig.posts()) == 2


def test_a_429_falls_back_to_the_named_fallback_group(gateway: Gateway) -> None:
    """Group A always answers 429; the request names `fallbacks: [B]` and `num_retries: 0`.

    The caller must get B's answer, A must have been called once and B once.
    """
    with (
        azure(gateway, always(_rate_limited()), max_retries=0) as primary,
        azure(gateway, always(chat_reply("from fallback")), max_retries=0) as fallback,
    ):
        response: Final = primary.chat(num_retries=0, fallbacks=[fallback.model])
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "from fallback"
        assert len(primary.posts()) == 1
        assert len(fallback.posts()) == 1


_AZURE_HEADERS: Final = {
    "x-ms-region": "Sweden Central",
    "apim-request-id": "apim-dec1-0001",
    "x-ratelimit-remaining-requests": "99",
    "x-ratelimit-remaining-tokens": "149000",
    "x-ratelimit-limit-requests": "100",
    "x-ratelimit-limit-tokens": "150000",
}


def test_azure_region_and_apim_request_id_reach_the_caller_as_llm_provider_headers(gateway: Gateway) -> None:
    """Azure answers 200 with `x-ms-region: Sweden Central` and `apim-request-id: apim-dec1-0001`.

    The caller must get `llm_provider-x-ms-region: Sweden Central` and `llm_provider-apim-request-id: apim-dec1-0001`.
    These are how operators find the Azure region and correlate a call with Azure support.
    """
    with azure(gateway, always(chat_reply(headers=_AZURE_HEADERS))) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.headers.get("llm_provider-x-ms-region") == "Sweden Central"
        assert response.headers.get("llm_provider-apim-request-id") == "apim-dec1-0001"


def test_azure_rate_limit_headers_reach_the_caller_under_their_bare_names(gateway: Gateway) -> None:
    """Same 200 answer with the four `x-ratelimit-*` headers.

    Because they are in `OPENAI_RESPONSE_HEADERS`, the caller must get `x-ratelimit-remaining-requests: 99`,
    `x-ratelimit-remaining-tokens: 149000`, `x-ratelimit-limit-requests: 100` and
    `x-ratelimit-limit-tokens: 150000` under their bare names, which OpenAI SDK clients read for client side
    throttling. UNVERIFIED: whether the proxy's own key rate limit headers would shadow these; none are set here.
    """
    with azure(gateway, always(chat_reply(headers=_AZURE_HEADERS))) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        for name in (
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
        ):
            assert response.headers.get(name) == _AZURE_HEADERS[name], (name, dict(response.headers))


def test_azure_rate_limit_headers_are_also_sent_with_the_llm_provider_prefix(gateway: Gateway) -> None:
    """Same 200 answer.

    The raw copy is kept too (`core_helpers.py:798`): the caller must also get
    `llm_provider-x-ratelimit-remaining-requests: 99`.
    """
    with azure(gateway, always(chat_reply(headers=_AZURE_HEADERS))) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.headers.get("llm_provider-x-ratelimit-remaining-requests") == "99"

"""Spend rows and errors for `/v1/responses` calls that the bridge served from a chat provider.

A bridged call is billed like any Responses call: one `LiteLLM_SpendLogs` row with `call_type: aresponses`, filed
under the managed id the caller got, priced from the chat usage (11 prompt and 7 completion tokens at 0.001 and
0.002 per token is 0.025), with the deployment's provider in `custom_llm_provider`. A streamed bridged call is
billed from the usage chunk the bridge asked for with `stream_options.include_usage`. A provider error keeps its
status (429 stays 429, 400 stays 400) and the provider's message, writes one `failure` row, and is sent upstream
once because the deployments set no retries. When the Rust spend writer replaces the Python one these rows are
what budgets and dashboards read, so each column is pinned with a literal.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply
from integration.providers.dec1_respapi_support import (
    CALL_COST,
    COMPLETION_TOKENS,
    PROMPT_TOKENS,
    Call,
    anthropic,
    chat_bridge,
    deepseek,
    error_message,
    openai_error,
    peer,
    responses,
    served,
    spend_rows,
    stream_events,
)


def test_a_bridged_call_writes_one_aresponses_row_priced_from_the_chat_usage(gateway: Gateway) -> None:
    """One non streamed call on the chat bridge, priced 0.001 input and 0.002 output per token.

    One SpendLogs row must land for the model group with `call_type: aresponses`, `spend == 0.025`,
    `prompt_tokens: 11`, `completion_tokens: 7`, `total_tokens: 18` and `status: success`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        served(responses(gateway, model))
        row: Final = spend_rows(gateway, model)[0]
        assert row["call_type"] == "aresponses"
        assert float(str(row["spend"])) == pytest.approx(CALL_COST)
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (
            PROMPT_TOKENS,
            COMPLETION_TOKENS,
            PROMPT_TOKENS + COMPLETION_TOKENS,
        )
        assert row["status"] == "success"


def test_the_bridged_row_is_filed_under_the_id_the_caller_got(gateway: Gateway) -> None:
    """The row's `request_id` must equal the managed `resp_` id in the response body, which is what the
    `previous_response_id` lookup and the logs page search by."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        identity: Final = served(responses(gateway, model))["id"]
        assert spend_rows(gateway, model)[0]["request_id"] == identity


@pytest.mark.parametrize(
    ("carrier", "provider_name"),
    [("chat", "openai"), ("deepseek", "deepseek"), ("anthropic", "anthropic")],
)
def test_the_bridged_row_names_the_deployment_provider(gateway: Gateway, carrier: str, provider_name: str) -> None:
    """One call on each carrier. The row's `custom_llm_provider` must be the deployment's provider (`openai` for
    the `use_chat_completions_api` deployment, `deepseek`, `anthropic`), never `litellm_proxy` or empty, and
    `call_type` stays `aresponses`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = (
            chat_bridge(scenario, provider)
            if carrier == "chat"
            else deepseek(scenario, provider)
            if carrier == "deepseek"
            else anthropic(scenario, provider)
        )
        served(responses(gateway, model))
        row: Final = spend_rows(gateway, model)[0]
        assert (row["custom_llm_provider"], row["call_type"]) == (provider_name, "aresponses")


def test_an_anthropic_bridged_call_is_priced_from_the_anthropic_usage(gateway: Gateway) -> None:
    """Anthropic answers with `input_tokens: 11`, `output_tokens: 7`. The row's `spend` must be 0.025 with the same
    deployment prices, showing the price comes from the chat usage the bridge saw."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        served(responses(gateway, model))
        assert float(str(spend_rows(gateway, model)[0]["spend"])) == pytest.approx(CALL_COST)


def test_a_streamed_bridged_call_is_priced_from_the_usage_chunk(gateway: Gateway) -> None:
    """A streamed call on the chat bridge whose usage arrives only in the final chat chunk (11 and 7 tokens).

    One row must land with `spend == 0.025` and `total_tokens: 18`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        stream_events(gateway, model)
        row: Final = spend_rows(gateway, model)[0]
        assert float(str(row["spend"])) == pytest.approx(CALL_COST)
        assert row["total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS


def test_a_streamed_bridged_row_is_filed_under_the_streamed_id(gateway: Gateway) -> None:
    """The `response.completed` snapshot's `id` must equal the row's `request_id` (the spend suite pins this for
    DeepSeek; repeated here on the explicit chat bridge)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        events: Final = stream_events(gateway, model)
        completed: Final = next(event for event in events if event["type"] == "response.completed")
        identity: Final = completed["response"]["id"]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]
        assert spend_rows(gateway, model)[0]["request_id"] == identity


def test_the_bridged_answer_carries_the_response_cost_header(gateway: Gateway) -> None:
    """The same priced non streamed call. The caller must get `x-litellm-response-cost` whose float value is 0.025.
    UNVERIFIED: assumes the Responses route sets the cost header the way chat completions does."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        response: Final = responses(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(CALL_COST)


def test_a_provider_429_on_a_bridged_call_keeps_its_status_and_message(gateway: Gateway) -> None:
    """The chat provider answers 429 `dec1 slow down`. The caller must get HTTP 429 whose error message contains
    `dec1 slow down`, and the peer receives exactly one POST."""
    with peer(lambda call: openai_error(429, "dec1 slow down")) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        response: Final = responses(gateway, model)
        assert response.status_code == 429, response.text
        assert "dec1 slow down" in error_message(response)
        assert len(provider.posts()) == 1


def test_a_provider_400_on_a_bridged_call_is_a_400(gateway: Gateway) -> None:
    """The chat provider answers 400 `dec1 bad field`. The caller must get HTTP 400 with that message."""
    with peer(lambda call: openai_error(400, "dec1 bad field")) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        response: Final = responses(gateway, model)
        assert response.status_code == 400, response.text
        assert "dec1 bad field" in error_message(response)


def test_an_anthropic_overloaded_error_on_a_bridged_call_keeps_anthropics_message(gateway: Gateway) -> None:
    """Anthropic answers 529 `{"type": "error", "error": {"type": "overloaded_error", "message": "dec1
    overloaded"}}`. The caller must get an error status of 500 or above whose body contains `dec1 overloaded`.
    UNVERIFIED: the exact status the Anthropic exception mapping gives 529."""

    def respond(call: Call) -> Reply:
        return Reply(
            status=529,
            body=b'{"type":"error","error":{"type":"overloaded_error","message":"dec1 overloaded"}}',
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        response: Final = responses(gateway, model)
        assert response.status_code >= 500, response.text
        assert "dec1 overloaded" in response.text


def test_a_failed_bridged_call_writes_one_failure_row(gateway: Gateway) -> None:
    """The chat provider answers 429. One SpendLogs row must land for the model group with `status: failure`,
    `call_type: aresponses` and `spend == 0`, so failed turns are visible but never billed."""
    with peer(lambda call: openai_error(429, "dec1 slow down")) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        assert responses(gateway, model).status_code == 429
        row: Final = spend_rows(gateway, model)[0]
        assert (row["status"], row["call_type"]) == ("failure", "aresponses")
        assert float(str(row["spend"])) == 0.0
        assert provider.posts()


def test_an_untranslatable_field_refusal_writes_no_provider_call(gateway: Gateway) -> None:
    """`partial_images: 2` is a Responses only field with no chat form. Without `drop_params` the bridge answers 400
    naming it (`does not support parameters: ['partial_images']`) and the peer receives no POST, so a refused
    request never costs a provider call."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        response: Final = responses(gateway, model, partial_images=2)
        assert response.status_code == 400, response.text
        assert "does not support parameters: ['partial_images']" in error_message(response)
        assert provider.posts() == ()

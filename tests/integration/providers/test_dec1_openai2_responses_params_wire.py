"""Contract: newer OpenAI Responses request fields on a native `openai/` deployment, and deployment identity headers.

`OpenAIResponsesAPIConfig.get_supported_openai_params` is every key of `ResponsesAPIRequestParams`
(`litellm/llms/openai/responses/transformation.py:453`, the TypedDict at `litellm/types/llms/openai.py:1385`),
which includes `safety_identifier`, `prompt` (stored prompt templates), `prompt_cache_key`,
`prompt_cache_retention`, `max_tool_calls` and `top_logprobs`. Each reaches `<api_base>/responses` with its name
and value unchanged. `conversation` (the Conversations API link) is not a named parameter at all and travels as an
unknown top level field. The older parametrized test in `test_dec1_openai_responses_wire.py` already pins
`store`, `background`, `include`, `metadata`, `instructions`, `text`, `parallel_tool_calls`, `truncation`,
`user`, `service_tier`, `max_output_tokens` and `prompt_cache_key`; this file covers the rest.

The deployment `organization` setting reaches chat completions as the `openai-organization` header through the
OpenAI SDK client (`litellm/main.py:2823`), but the Responses API is sent with raw httpx and
`validate_environment` (`transformation.py:1058`) only sets `Authorization`, so a deployment's organization is
lost on `/v1/responses`. That is pinned as a BUG skip.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration.providers.dec1_openai_support import (
    data_events,
    deployment,
    marker,
    peer,
    responses,
    served,
    stream_lines,
)
from pydantic import JsonValue

_MODEL: Final = "openai/gpt-4o-mini"
_CONVERSATION_DROPPED_BUG: Final = (
    "BUG: the Responses `conversation` field is silently dropped before OpenAI; it is not a declared Responses"
    " request param (litellm/types/llms/openai.py on main has no conversation key)"
)
_PROMPT: Final[dict[str, JsonValue]] = {
    "id": "pmpt_dec1openai2",
    "version": "3",
    "variables": {
        "customer_name": "Synthetic Customer",
        "policy": {"type": "input_text", "text": "Refunds within 30 days."},
        "attachment": {"type": "input_file", "file_id": "file-dec1-openai2-policy"},
    },
}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("safety_identifier", "dec1-hashed-end-user"),
        ("prompt_cache_retention", "24h"),
        ("prompt", _PROMPT),
        ("max_tool_calls", 3),
        ("top_logprobs", 2),
        ("conversation", "conv_dec1openai2"),
        ("conversation", {"id": "conv_dec1openai2"}),
        ("stream_options", None),
    ],
    ids=[
        "safety_identifier",
        "prompt_cache_retention",
        "prompt_template",
        "max_tool_calls",
        "top_logprobs",
        "conversation_string",
        "conversation_object",
        "stream_options_null_placeholder",
    ],
)
def test_each_newer_responses_field_is_forwarded_unchanged(gateway: Gateway, name: str, value: JsonValue) -> None:
    """The caller sends one newer Responses field to an `openai/gpt-4o-mini` deployment.

    The provider body carries `<name>: <value>` exactly as sent: a stored prompt keeps its id, version and every
    variable (text and `input_file` variables included), and `conversation` keeps its string or object form.
    The `stream_options: null` cell checks the opposite: a null optional field is not sent at all. UNVERIFIED:
    the null drop relies on `ResponsesAPIRequestUtils.get_requested_response_api_optional_param` skipping None.
    The `conversation` cells are skipped as a product bug, see `_CONVERSATION_DROPPED_BUG`.
    """
    if name == "conversation":
        pytest.skip(_CONVERSATION_DROPPED_BUG)
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, **{name: value}))
        body: Final = provider.only().body
        if value is None:
            assert name not in body, body
        else:
            assert body[name] == value, body


def test_prompt_cache_key_and_retention_travel_together(gateway: Gateway) -> None:
    """The caller sends `prompt_cache_key: "dec1-tenant-7"` and `prompt_cache_retention: "in_memory"` together.

    The provider body carries both, unchanged and side by side. They are separate knobs (routing key and cache
    lifetime); a port that treats retention as a variant of the key would drop one.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, prompt_cache_key="dec1-tenant-7", prompt_cache_retention="in_memory"))
        body: Final = provider.only().body
        assert (body["prompt_cache_key"], body["prompt_cache_retention"]) == ("dec1-tenant-7", "in_memory"), body


def test_a_prompt_template_call_keeps_the_caller_input_and_instructions(gateway: Gateway) -> None:
    """The caller sends `prompt` with variables plus its own `input` and `instructions`.

    The provider body carries all three: the proxy does not resolve the OpenAI stored prompt itself and does not
    drop `input` because a prompt is present. LiteLLM's own prompt management only runs on `prompt_id`, which is
    not set here.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, prompt=_PROMPT, input=text, instructions="Be brief."))
        body: Final = provider.only().body
        assert (body["prompt"], body["input"], body["instructions"]) == (_PROMPT, text, "Be brief."), body
        assert "prompt_id" not in body and "prompt_variables" not in body, body


def test_safety_identifier_and_user_are_both_forwarded(gateway: Gateway) -> None:
    """The caller sends `user: "dec1-user"` and `safety_identifier: "dec1-safety"`.

    The provider body carries both under their own names. `safety_identifier` replaced `user` for abuse
    detection at OpenAI, so collapsing one into the other changes what OpenAI attributes to the end user.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, user="dec1-user", safety_identifier="dec1-safety"))
        body: Final = provider.only().body
        assert (body["user"], body["safety_identifier"]) == ("dec1-user", "dec1-safety"), body


def test_a_streamed_call_forwards_prompt_cache_fields_and_stream_true(gateway: Gateway) -> None:
    """The caller streams with `prompt_cache_key` and `prompt_cache_retention: "24h"`.

    The provider body has `stream: true` and both cache fields unchanged, and the caller receives the event
    stream ending in `response.completed`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _, lines = stream_lines(
            gateway,
            "/v1/responses",
            {
                "model": model,
                "input": marker(),
                "stream": True,
                "prompt_cache_key": "dec1-stream-key",
                "prompt_cache_retention": "24h",
            },
        )
        body: Final = provider.only().body
        assert (body["stream"], body["prompt_cache_key"], body["prompt_cache_retention"]) == (
            True,
            "dec1-stream-key",
            "24h",
        ), body
        assert data_events(lines)[-1]["type"] == "response.completed"


def test_a_deployment_organization_reaches_openai_on_responses(gateway: Gateway) -> None:
    """The deployment sets `organization: org-dec1-openai2-resp` and the caller sends a Responses request.

    The provider request must carry `openai-organization: org-dec1-openai2-resp`, as chat completions on the same
    deployment does, so usage is billed to the configured org.
    """
    pytest.skip(
        "BUG: OpenAIResponsesAPIConfig.validate_environment only sets Authorization, so a deployment "
        "organization never reaches /v1/responses while chat completions sends it as openai-organization"
    )
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, organization="org-dec1-openai2-resp")
        served(responses(gateway, model))
        assert provider.only().headers["openai-organization"] == "org-dec1-openai2-resp"


def test_a_deployment_openai_project_header_reaches_openai_on_responses(gateway: Gateway) -> None:
    """The deployment sets `extra_headers: {"OpenAI-Project": "proj_dec1_resp"}` and the caller sends a
    Responses request.

    The provider request carries `openai-project: proj_dec1_resp`; per-deployment extra headers are the only way
    to pin a project. UNVERIFIED: deployment `extra_headers` are merged into the Responses request headers by
    `BaseLLMHTTPHandler.response_api_handler`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, extra_headers={"OpenAI-Project": "proj_dec1_resp"})
        served(responses(gateway, model))
        assert provider.only().headers["openai-project"] == "proj_dec1_resp"


def test_a_caller_openai_organization_header_is_not_forwarded_on_responses(gateway: Gateway) -> None:
    """The caller sends `OpenAI-Organization: org-dec1-caller` and `OpenAI-Project: proj_dec1_caller` on
    `/v1/responses` against a shared proxy with no forwarding settings.

    Neither header reaches the provider, so a caller cannot bill another org or project through the deployment
    key; the chat completions twin is pinned in `test_dec1_openai_chat_params_wire.py`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(
            responses(
                gateway,
                model,
                headers={"OpenAI-Organization": "org-dec1-caller", "OpenAI-Project": "proj_dec1_caller"},
            )
        )
        headers: Final = provider.only().headers
        assert "openai-organization" not in headers and "openai-project" not in headers, dict(headers)

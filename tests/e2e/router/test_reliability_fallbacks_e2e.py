"""Live e2e: per-request fallbacks reroute a failing deployment's traffic to a
healthy one.

Each test registers a primary deployment that fails (an unreachable base URL, or
a 1ms deadline) and calls it with a `router_settings_override` mapping it to the
real `gpt-5.5`. The proof the fallback fired is twofold: the response is a
completion from `gpt-5.5`, and the proxy reports at least one attempted fallback
in the x-litellm-attempted-fallbacks header. Empty content is accepted only when
`finish_reason == "length"` and the response billed completion tokens, since
gpt-5.5 counts reasoning against max_tokens and can consume the whole budget
before emitting any text; a fallback that produced nothing at all still fails.

The context-window and content-policy cases are different reroutes from a plain
failure: the provider refuses the prompt itself, on length or on policy, and
`context_window_fallbacks` / `content_policy_fallbacks` are the settings that
reroute those, not `fallbacks`. The policy refusal is a real one, from an Azure
OpenAI content filter rejecting a jailbreak prompt, and a control call first
proves the refusal reaches the customer as a 400 when no reroute is configured.
Azure intermittently answers without running its prompt filter at all (the
body's prompt_filter_results carry no verdict), which is not a pass, so both
calls send the prompt again, a bounded number of times, until the filter ran.

The caller-credential case (#45309) turns on `forward_llm_provider_auth_headers`, so a caller brings its
own Anthropic key in `x-api-key`. The key is invalid, real Anthropic rejects it with a 401, and the
fallback reroutes to a Bedrock deployment that signs with the proxy's IAM credentials. The caller must
get a 200 from Bedrock. Before the fix the proxy handed Bedrock the caller's Anthropic key as a bearer
token, which Bedrock rejects. The Anthropic deployment carries the proxy's real key, so until the setting
reaches the worker that serves the call, Anthropic answers without a fallback, and the test resends
until a call either falls back or fails. The setting is proxy-wide, so that class is marked
`proxy_wide_config` and its teardown puts the setting back as it found it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import Headers, NoBody, StreamingResponse, unwrap
from e2e_metadata import Domain, Mode, Provider, Subject, meta
from lifecycle import ResourceManager
from models import ChatMessage, LiteLLMParamsBody, ReliabilityChatBody, RouterSettingsOverride
from pydantic import BaseModel, ConfigDict, Field
from reliability_support import (
    AZURE_MODEL,
    CONTENT_POLICY_PROMPT,
    REAL_MODEL,
    SMALL_CONTEXT_MODEL,
    azure_prompt_filter_skipped,
    chat_override,
    completion_tokens_of,
    content_of,
    create_bad_base_deployment,
    create_content_filtered_deployment,
    create_small_context_deployment,
    create_timeout_deployment,
    finish_reason_of,
    oversized_prompt,
    reasoning_tokens_of,
)

pytestmark = pytest.mark.e2e

FALLBACK_MODEL: Final = "gpt-5.5"


def _assert_served_by_fallback(resp: StreamingResponse) -> None:
    assert resp.status_code == 200, f"expected 200 after fallback, got {resp.status_code}: {resp.body[:300]}"
    content = content_of(resp)
    finish_reason = finish_reason_of(resp)
    completion_tokens = completion_tokens_of(resp) or 0
    reasoning_tokens = reasoning_tokens_of(resp) or 0
    assert isinstance(content, str), (
        f"the gpt-5.5 fallback should have returned a completion body, got content {content!r} (body={resp.body[:300]})"
    )
    assert content or (finish_reason == "length" and completion_tokens > 0), (
        f"the gpt-5.5 fallback returned empty content with finish_reason={finish_reason!r}, "
        f"completion_tokens={completion_tokens}, reasoning_tokens={reasoning_tokens}; empty "
        f"content is only acceptable when the budget was spent on non-visible reasoning "
        f"(body={resp.body[:300]})"
    )
    attempted = resp.headers.get("x-litellm-attempted-fallbacks")
    assert attempted is not None, "response is missing the x-litellm-attempted-fallbacks header"
    assert int(attempted) >= 1, f"x-litellm-attempted-fallbacks should be >= 1, got {attempted!r}"


AZURE_FILTER_ATTEMPTS: Final = 3


def _chat_once_azure_runs_its_filter(send: Callable[[], StreamingResponse]) -> StreamingResponse:
    for attempt in range(1, AZURE_FILTER_ATTEMPTS):
        resp = send()
        if not azure_prompt_filter_skipped(resp):
            return resp
        print(
            "e2e: azure answered without running its prompt filter; sending the jailbreak prompt again "
            f"({attempt}/{AZURE_FILTER_ATTEMPTS - 1})",
            flush=True,
        )
    return send()


def _filter_verdict(resp: StreamingResponse) -> str:
    if azure_prompt_filter_skipped(resp):
        return "azure skipped its prompt filter on every attempt"
    return "the filter ran and let the prompt through"


class TestReliabilityFallbacks:
    @pytest.mark.covers("reliability.fallback.5xx.routes_to_fallback")
    @meta(
        Subject(
            domain=Domain.ROUTING,
            providers=(Provider.OPENAI,),
            models=(FALLBACK_MODEL, REAL_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_5xx_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-fail-{unique_marker()}"
        model_id = create_bad_base_deployment(client.proxy, primary)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(
            client.proxy,
            scoped_key,
            primary,
            f"say hi {unique_marker()}",
            override=RouterSettingsOverride(fallbacks=[{primary: [FALLBACK_MODEL]}]),
        )
        _assert_served_by_fallback(resp)

    @pytest.mark.covers("reliability.fallback.timeout.routes_to_fallback")
    @meta(
        Subject(
            domain=Domain.ROUTING,
            providers=(Provider.OPENAI,),
            models=(FALLBACK_MODEL, REAL_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_timeout_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-tofail-{unique_marker()}"
        model_id = create_timeout_deployment(client.proxy, primary)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(
            client.proxy,
            scoped_key,
            primary,
            f"say hi {unique_marker()}",
            override=RouterSettingsOverride(fallbacks=[{primary: [FALLBACK_MODEL]}]),
        )
        _assert_served_by_fallback(resp)

    @pytest.mark.covers("reliability.fallback.context_window.routes_to_fallback")
    @meta(
        Subject(
            domain=Domain.ROUTING,
            providers=(Provider.OPENAI,),
            models=(FALLBACK_MODEL, SMALL_CONTEXT_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_context_window_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-ctxfail-{unique_marker()}"
        model_id = create_small_context_deployment(client.proxy, primary)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        resp = chat_override(
            client.proxy,
            scoped_key,
            primary,
            oversized_prompt(unique_marker()),
            override=RouterSettingsOverride(context_window_fallbacks=[{primary: [FALLBACK_MODEL]}]),
        )
        _assert_served_by_fallback(resp)

    @pytest.mark.covers("reliability.fallback.content_policy.routes_to_fallback")
    @meta(
        Subject(
            domain=Domain.ROUTING,
            providers=(Provider.AZURE, Provider.OPENAI,),
            models=(AZURE_MODEL, FALLBACK_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_content_policy_routes_to_fallback(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary = f"reliability-policyfail-{unique_marker()}"
        model_id = create_content_filtered_deployment(client.proxy, primary)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        refused = _chat_once_azure_runs_its_filter(
            lambda: chat_override(client.proxy, scoped_key, primary, f"{CONTENT_POLICY_PROMPT} {unique_marker()}")
        )
        assert refused.status_code == 400, (
            f"the content filter should have refused the jailbreak prompt with a 400, got {refused.status_code} "
            f"({_filter_verdict(refused)}): {refused.body[:300]}"
        )

        resp = _chat_once_azure_runs_its_filter(
            lambda: chat_override(
                client.proxy,
                scoped_key,
                primary,
                f"{CONTENT_POLICY_PROMPT} {unique_marker()}",
                override=RouterSettingsOverride(content_policy_fallbacks=[{primary: [FALLBACK_MODEL]}]),
            )
        )
        _assert_served_by_fallback(resp)


ANTHROPIC_MODEL: Final = "anthropic/claude-haiku-5-5"
BEDROCK_MODEL: Final = "bedrock/us.anthropic.claude-haiku-5-5"


class ForwardingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    forward_llm_provider_auth_headers: bool = False


class UiSettingsRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    values: ForwardingSettings


class CallerProviderKeyHeaders(Headers):
    x_litellm_api_key: str = Field(serialization_alias="x-litellm-api-key", repr=False)
    x_api_key: str = Field(serialization_alias="x-api-key", repr=False)


def _forwarding(client: ComplexityRouterClient) -> bool:
    return unwrap(
        client.proxy.transport.get(
            "/get/ui_settings",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=UiSettingsRead,
        )
    ).values.forward_llm_provider_auth_headers


def _set_forwarding(client: ComplexityRouterClient, enabled: bool) -> None:
    unwrap(
        client.proxy.transport.patch(
            "/update/ui_settings",
            headers=client.proxy.transport.master,
            json=ForwardingSettings(forward_llm_provider_auth_headers=enabled),
            response_type=NoBody,
        )
    )


def _chat(client: ComplexityRouterClient, key: str, caller_key: str, primary: str, fallback: str) -> StreamingResponse:
    return client.proxy.transport.send(
        "/chat/completions",
        headers=CallerProviderKeyHeaders(x_litellm_api_key=key, x_api_key=caller_key),
        json=ReliabilityChatBody(
            model=primary,
            messages=[ChatMessage(role="user", content=f"Say hi. {unique_marker()}")],
            max_tokens=64,
            router_settings_override=RouterSettingsOverride(fallbacks=[{primary: [fallback]}]),
            cache={"no-cache": True},
        ),
    )


def _fell_back(response: StreamingResponse) -> bool:
    return int(response.headers.get("x-litellm-attempted-fallbacks", "0")) >= 1


def _first_call_after_forwarding_applies(
    client: ComplexityRouterClient, key: str, caller_key: str, primary: str, fallback: str
) -> StreamingResponse:
    deadline: Final = time.monotonic() + POLL_TIMEOUT
    while True:
        response = _chat(client, key, caller_key, primary, fallback)  # rebind-ok: resend until the setting lands
        if not response.ok or _fell_back(response) or time.monotonic() > deadline:
            return response
        time.sleep(POLL_INTERVAL)


@pytest.mark.proxy_wide_config
class TestFallbackCallerCredentialScope:
    @meta(
        Subject(
            domain=Domain.ROUTING,
            providers=(Provider.ANTHROPIC, Provider.BEDROCK),
            models=(ANTHROPIC_MODEL, BEDROCK_MODEL),
            mode=Mode.NONSTREAM,
        )
    )
    def test_rejected_caller_anthropic_key_falls_back_to_bedrock_iam(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary: Final = f"e2e-caller-key-anthropic-{unique_marker()}"
        fallback: Final = f"e2e-caller-key-bedrock-{unique_marker()}"
        anthropic_id: Final = client.proxy.create_model(
            primary,
            LiteLLMParamsBody(model=ANTHROPIC_MODEL, api_key="os.environ/ANTHROPIC_API_KEY"),
            provider_live=True,
        )
        resources.defer(lambda: client.proxy.delete_model(anthropic_id))
        bedrock_id: Final = client.proxy.create_model(
            fallback,
            LiteLLMParamsBody(
                model=BEDROCK_MODEL,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
            provider_live=True,
        )
        resources.defer(lambda: client.proxy.delete_model(bedrock_id))
        previous: Final = _forwarding(client)
        resources.defer(lambda: _set_forwarding(client, previous))
        _set_forwarding(client, True)

        response: Final = _first_call_after_forwarding_applies(
            client, scoped_key, f"sk-ant-api03-e2e-invalid-{unique_marker()}", primary, fallback
        )
        assert response.status_code == 200, (
            f"the Bedrock fallback should answer with the proxy's IAM credentials, got {response.status_code}: "
            f"{response.body[:300]}"
        )
        assert _fell_back(response), (
            f"the caller's invalid Anthropic key never made Anthropic fail within {POLL_TIMEOUT}s, so the "
            f"forwarding setting did not reach the proxy: {response.headers}"
        )
        assert content_of(response), f"the Bedrock fallback returned no content: {response.body[:300]}"

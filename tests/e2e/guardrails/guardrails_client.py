"""Client for the guardrails e2e suite: register global (default-on) guardrails
and chat through them on the shared ProxyClient so resources.defer cleans up.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, settle_propagation, unique_marker
from e2e_http import NoBody, Result, StreamingResponse, Success, unwrap
from lifecycle import ResourceManager
from models import (
    AnthropicMessagesBody,
    AnthropicMessagesResponse,
    ChatBody,
    ChatMessage,
    ChatResponse,
    ChatTool,
    KeyGenerateBody,
    LiteLLMParamsBody,
    TeamDeleteBody,
    TeamInfoParams,
    TeamInfoResponse,
    TeamMetadata,
    TeamNewBody,
    TeamNewResponse,
)
from proxy_client import ProxyClient
from pydantic import BaseModel

GuardrailMode = Literal["pre_call", "post_call", "during_call", "logging_only"]
BlockedWordAction = Literal["BLOCK", "MASK"]


class BlockedWordBody(BaseModel):
    keyword: str
    action: BlockedWordAction


class GuardrailParamsBase(BaseModel):
    mode: GuardrailMode
    default_on: bool


class ContentFilterParamsBody(GuardrailParamsBase):
    guardrail: Literal["litellm_content_filter"] = "litellm_content_filter"
    blocked_words: list[BlockedWordBody]


class BedrockGuardrailParamsBody(GuardrailParamsBase):
    guardrail: Literal["bedrock"] = "bedrock"
    guardrailIdentifier: str
    guardrailVersion: str
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region_name: str | None = None


class OpenAIModerationParamsBody(GuardrailParamsBase):
    guardrail: Literal["openai_moderation"] = "openai_moderation"
    api_key: str | None = None
    model: str | None = None


class BlockCodeExecutionParamsBody(GuardrailParamsBase):
    guardrail: Literal["block_code_execution"] = "block_code_execution"


class PresidioParamsBody(GuardrailParamsBase):
    """Presidio PII guardrail params. `presidio_filter_scope="input"` keeps the
    registration to a single callback on the configured mode; the default
    ("both") also registers a second post_call output-masking callback, which a
    pre_call- or logging_only-scoped test must not drag in. `output_parse_pii`
    stays unset/False: True would unmask the response back to the caller."""

    guardrail: Literal["presidio"] = "presidio"
    presidio_analyzer_api_base: str
    presidio_anonymizer_api_base: str
    presidio_filter_scope: Literal["input", "output", "both"] | None = None
    presidio_language: str | None = None
    output_parse_pii: bool | None = None


class ToolPermissionRuleBody(BaseModel):
    """One tool_permission rule: a decision for the tool named by `tool_name`."""

    id: str
    tool_name: str
    decision: Literal["allow", "deny"]


class ToolPermissionParamsBody(GuardrailParamsBase):
    """Tool-permission guardrail params. `default_action="deny"` makes the rules
    an allow-list, and `on_disallowed_action="block"` turns a disallowed tool into
    a 400 instead of rewriting the request; "rewrite" is a different product
    promise and belongs to its own scenario."""

    guardrail: Literal["tool_permission"] = "tool_permission"
    rules: list[ToolPermissionRuleBody]
    default_action: Literal["allow", "deny"] = "deny"
    on_disallowed_action: Literal["block", "rewrite"] = "block"


GuardrailParamsBody = (
    ContentFilterParamsBody
    | BedrockGuardrailParamsBody
    | OpenAIModerationParamsBody
    | BlockCodeExecutionParamsBody
    | PresidioParamsBody
    | ToolPermissionParamsBody
)


class GuardrailSpecBody(BaseModel):
    guardrail_name: str
    litellm_params: GuardrailParamsBody


class GuardrailCreateBody(BaseModel):
    guardrail: GuardrailSpecBody


class GuardrailCreateResponse(BaseModel):
    guardrail_id: str


class ApplyGuardrailRequest(BaseModel):
    guardrail_name: str
    text: str
    language: str | None = None
    input_type: str = "request"


class ApplyGuardrailResponse(BaseModel):
    response_text: str


class _ResponsesGuardrailBody(BaseModel):
    model: str
    input: str
    guardrails: list[str] | None = None


@dataclass(frozen=True, slots=True)
class GuardrailsClient:
    proxy: ProxyClient

    def create_content_filter_guardrail(self, name: str, blocked_keyword: str) -> str:
        return self.register(
            name,
            ContentFilterParamsBody(
                mode="pre_call",
                default_on=True,
                blocked_words=[BlockedWordBody(keyword=blocked_keyword, action="BLOCK")],
            ),
        )

    def create_bedrock_guardrail(
        self,
        name: str,
        *,
        identifier: str,
        version: str,
        default_on: bool = False,
    ) -> str:
        """Register a Bedrock guardrail, opted out of `default_on` by default.

        `default_on=True` applies the guardrail to every request the proxy serves,
        not just this test's. When the upstream ApplyGuardrail call fails (a missing
        bedrock:ApplyGuardrail permission answers 403), that failure is returned to
        unrelated traffic as `403 Bedrock guardrail request failed`, so one guardrail
        test takes out whatever else is running. Callers select the guardrail
        per-request instead, which keeps the blast radius to the test that wants it.
        """
        return self.register(
            name,
            BedrockGuardrailParamsBody(
                mode="pre_call",
                default_on=default_on,
                guardrailIdentifier=identifier,
                guardrailVersion=version,
            ),
        )

    def create_backend_model(
        self,
        resources: ResourceManager,
        prefix: str = "e2e-guard-backend",
        *,
        backend: str = "gemini/gemini-2.5-flash",
        api_key: str = "os.environ/GEMINI_API_KEY",
    ) -> str:
        """Register a chat deployment for a guardrail test to run against
        (deleted on teardown). The guardrails under test here gate on prompt/output
        content, not the backend, so a cheap deployment stands in for the model the
        customer would call. Messages/responses suites pass an Anthropic/OpenAI backend."""
        model_name = f"{prefix}-{unique_marker()}"
        model_id = self.proxy.create_model(
            model_name,
            LiteLLMParamsBody(model=backend, api_key=api_key),
        )
        resources.defer(lambda: self.proxy.delete_model(model_id))
        return model_name

    def register(self, name: str, params: GuardrailParamsBody) -> str:
        """Register any guardrail via POST /guardrails and return its id, once every
        replica can be expected to serve it. New built-ins register with
        default_on=False and are opted into per request via the chat body's
        `guardrails` list, so one guardrail under test never intercepts unrelated
        traffic on the shared proxy.

        /guardrails is a control-plane route and guardrails reach the data plane on
        the config reload, so a request naming this guardrail the instant the POST
        returns can 404 with "Guardrail not found" on a replica that has not
        reloaded. There is no data-plane read that lists guardrails, so unlike
        ProxyClient.create_model this settles on the propagation budget alone with
        nothing to poll first."""
        guardrail_id = unwrap(
            self.proxy.transport.post(
                "/guardrails",
                headers=self.proxy.transport.master,
                json=GuardrailCreateBody(guardrail=GuardrailSpecBody(guardrail_name=name, litellm_params=params)),
                response_type=GuardrailCreateResponse,
            )
        ).guardrail_id
        settle_propagation(time.monotonic())
        return guardrail_id

    def delete_guardrail(self, guardrail_id: str) -> None:
        _ = self.proxy.transport.delete(
            f"/guardrails/{guardrail_id}",
            headers=self.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def create_team_opted_out_of_global_guardrails(self, alias: str) -> str:
        team_id = unwrap(
            self.proxy.transport.post(
                "/team/new",
                headers=self.proxy.transport.master,
                json=TeamNewBody(
                    team_alias=alias,
                    metadata=TeamMetadata(disable_global_guardrails=True),
                ),
                response_type=TeamNewResponse,
            )
        ).team_id
        self._await_team(team_id)
        return team_id

    def delete_team(self, team_id: str) -> None:
        _ = self.proxy.transport.post(
            "/team/delete",
            headers=self.proxy.transport.master,
            json=TeamDeleteBody(team_ids=[team_id]),
            response_type=NoBody,
        )

    def create_key_in_team(self, team_id: str) -> str:
        return self.proxy.generate_key(KeyGenerateBody(team_id=team_id, user_id="e2e-guardrails-user"))

    def chat(
        self,
        key: str,
        model: str,
        text: str,
        *,
        guardrails: list[str] | None = None,
        max_tokens: int = 16,
        tools: list[ChatTool] | None = None,
    ) -> Result[ChatResponse]:
        """Drive a chat call, optionally opting into named guardrails for this
        request only (the per-request `guardrails` selector). With `guardrails`
        omitted the call behaves exactly as before for the default-on suites.
        `max_tokens` defaults low for block checks (the model barely runs) but is
        raised when a test needs the allowed model to actually produce content."""
        return self.proxy.chat(
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=text)],
                max_tokens=max_tokens,
                guardrails=guardrails,
                tools=tools,
            ),
        )

    def chat_raw(
        self,
        key: str,
        model: str,
        text: str,
        *,
        guardrails: list[str] | None = None,
        max_tokens: int = 16,
        tools: list[ChatTool] | None = None,
        tool_choice: str | None = None,
    ) -> StreamingResponse:
        """Drive /chat/completions returning the raw HTTP outcome, for the
        assertions a typed body cannot carry: the `x-litellm-applied-guardrails`
        response header, which is how an ALLOW scenario proves the guardrail ran
        rather than being absent."""
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=text)],
                max_tokens=max_tokens,
                guardrails=guardrails,
                tools=tools,
                tool_choice=tool_choice,
            ),
        )

    def chat_stream_raw(
        self,
        key: str,
        model: str,
        text: str,
        *,
        guardrails: list[str] | None = None,
        max_tokens: int = 64,
    ) -> StreamingResponse:
        """Drive /chat/completions with stream=true, returning the raw HTTP
        outcome (status, headers, SSE events) via the shared ProxyClient stream
        sender - a streamed guardrail block is judged on status and stream
        shape, not a typed body."""
        return self.proxy.chat_stream(
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=text)],
                max_tokens=max_tokens,
                stream=True,
                guardrails=guardrails,
            ),
        )

    def messages(
        self,
        key: str,
        model: str,
        text: str,
        *,
        guardrails: list[str] | None = None,
        max_tokens: int = 16,
    ) -> Result[AnthropicMessagesResponse]:
        return self.proxy.messages(
            key,
            AnthropicMessagesBody(
                model=model,
                messages=[ChatMessage(role="user", content=text)],
                max_tokens=max_tokens,
                guardrails=guardrails,
            ),
        )

    def responses(
        self,
        key: str,
        model: str,
        text: str,
        *,
        guardrails: list[str] | None = None,
    ) -> StreamingResponse:
        return self.proxy.transport.send(
            "/v1/responses",
            headers=self.proxy.transport.bearer(key),
            json=_ResponsesGuardrailBody(model=model, input=text, guardrails=guardrails),
        )

    def apply_guardrail(self, key: str, *, name: str, text: str) -> Result[ApplyGuardrailResponse]:
        return self.proxy.transport.post(
            "/guardrails/apply_guardrail",
            headers=self.proxy.transport.bearer(key),
            json=ApplyGuardrailRequest(guardrail_name=name, text=text),
            response_type=ApplyGuardrailResponse,
        )

    def _await_team(self, team_id: str) -> None:
        deadline = time.monotonic() + POLL_TIMEOUT
        last: Result[TeamInfoResponse] | None = None
        while time.monotonic() < deadline:
            last = self.proxy.transport.get(
                "/team/info",
                headers=self.proxy.transport.master,
                params=TeamInfoParams(team_id=team_id),
                response_type=TeamInfoResponse,
            )
            if isinstance(last, Success):
                return
            time.sleep(POLL_INTERVAL)
        raise AssertionError(f"team {team_id!r} was created but /team/info never returned it: {last}")


def build_client(proxy: ProxyClient) -> GuardrailsClient:
    return GuardrailsClient(proxy=proxy)


def poll_until_blocked[R: BaseModel](call: Callable[[], Result[R]]) -> Result[R]:
    """Retry a call that a guardrail should reject until it is, returning the last result.

    Registering a guardrail is a control-plane write; the data-plane worker that
    serves /chat/completions picks it up only on its next periodic DB sync (~30s in
    proxy_server.py). A call issued right after the create therefore runs against a
    worker that has no guardrail yet and is allowed through, which is in-flight
    propagation rather than a guardrail that failed to block. Polling to the deadline
    waits that out so the assertions judge the synced state; a guardrail that never
    blocks still fails, on the last allowed result.
    """
    deadline = time.monotonic() + POLL_TIMEOUT
    last = call()
    while time.monotonic() < deadline:
        if not isinstance(last, Success):
            return last
        time.sleep(POLL_INTERVAL)
        last = call()
    return last


#: Statuses a stream poll keeps retrying through instead of returning as "the
#: block": network failures (-1), key propagation (401), rate limits (429) -
#: transient rig noise, not a guardrail verdict.
_TRANSIENT_STREAM_STATUSES = frozenset({-1, 401, 429})


def poll_until_blocked_stream(call: Callable[[], StreamingResponse]) -> StreamingResponse:
    """poll_until_blocked for raw/streamed sends, which return a StreamingResponse
    instead of a Result: retry while the call still succeeds (the data-plane worker
    has not picked the new guardrail up yet) or fails with a transient status,
    returning the first guardrail-shaped non-2xx outcome or the last result at
    the deadline."""
    deadline = time.monotonic() + POLL_TIMEOUT
    last = call()
    while time.monotonic() < deadline:
        if not last.ok and last.status_code not in _TRANSIENT_STREAM_STATUSES:
            return last
        time.sleep(POLL_INTERVAL)
        last = call()
    return last

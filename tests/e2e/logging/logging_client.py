"""Client for the logging e2e suite: team/key/org-scoped Langfuse OTEL callbacks,
chat (including tools), Prometheus scrape, and Langfuse observation read-back.

Holds the shared Gateway so the ``resources`` fixture cleans up keys, teams,
users, orgs, and models it creates. External Langfuse reads go through
``e2e_http`` (the only module allowed to call ``requests.*``).

Uses the ``langfuse_otel`` callback (OTLP to ``{host}/api/public/otel``), not
the classic ``langfuse`` SDK callback. OTEL generations land as name
``litellm_request``; correlate by unique prompt marker and ``user_api_key_alias``
in metadata. Spend is on ``calculatedTotalCost`` (StandardLogging response_cost).
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT
from e2e_gateway import Gateway, build_gateway
from e2e_http import (
    URL,
    AuthHeaders,
    NoBody,
    StreamingResponse,
    Success,
    get,
    unwrap,
)
from models import (
    AnthropicMessagesBody,
    ChatBody,
    ChatMessage,
    ChatResponse,
    ChatTool,
    ChatToolFunction,
    KeyGenerateBody,
    KeyLoggingCallback,
    KeyLoggingCallbackVars,
    KeyMetadata,
    LiteLLMParamsBody,
    OrgDeleteBody,
    OrgNewBody,
    OrgNewResponse,
    SpendLogRow,
    TeamDeleteBody,
    TeamNewBody,
    TeamNewResponse,
    UserDeleteBody,
    UserNewBody,
    UserNewResponse,
)

# Deliberately invalid *upstream provider* key for failure-path tests.
# Not a LiteLLM virtual key; OpenAI must reject it after the proxy accepts the call.
INVALID_UPSTREAM_API_KEY = "sk-upstream-invalid-for-langfuse-e2e-only"

WEATHER_TOOL = ChatTool(
    type="function",
    function=ChatToolFunction(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
)


class ResponsesRequestBody(BaseModel):
    """OpenAI Responses API /v1/responses request (non-streaming)."""

    model: str
    input: str
    max_output_tokens: int


class TeamCallbackBody(BaseModel):
    callback_name: Literal["langfuse_otel", "langfuse", "langsmith", "gcs"]
    callback_type: Literal["success", "failure", "success_and_failure"]
    callback_vars: dict[str, str]


class TeamCallbackResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str


class GuardrailLitellmParams(BaseModel):
    guardrail: str
    mode: str
    default_on: bool = False
    rules: list[dict[str, object]] | None = None
    default_action: str | None = None
    on_disallowed_action: str | None = None


class GuardrailSpec(BaseModel):
    guardrail_name: str
    litellm_params: GuardrailLitellmParams


class CreateGuardrailBody(BaseModel):
    guardrail: GuardrailSpec


class CreateGuardrailResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    guardrail_id: str | None = None
    guardrail_name: str | None = None


class LangfuseObservation(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    trace_id: str | None = Field(default=None, alias="traceId")
    name: str | None = None
    type: str | None = None
    calculated_total_cost: float | None = Field(default=None, alias="calculatedTotalCost")
    level: str | None = None
    input: object | None = None
    output: object | None = None
    metadata: object | None = None
    usage: object | None = None
    usage_details: object | None = Field(default=None, alias="usageDetails")
    model: str | None = None


class LangfuseObservationList(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[LangfuseObservation] = []


class LangfuseListParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    limit: int = 100
    trace_id: str | None = Field(default=None, alias="traceId")
    name: str | None = None
    from_start_time: str | None = Field(default=None, alias="fromStartTime")


@dataclass(frozen=True, slots=True)
class LangfuseCreds:
    public_key: str
    secret_key: str
    host: str

    @property
    def auth_headers(self) -> AuthHeaders:
        token = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode()
        return AuthHeaders(authorization=f"Basic {token}")

    def callback_vars(self) -> dict[str, str]:
        return {
            "langfuse_public_key": self.public_key,
            "langfuse_secret_key": self.secret_key,
            "langfuse_host": self.host,
        }

    def key_logging_metadata(self) -> KeyMetadata:
        return KeyMetadata(
            logging=[
                KeyLoggingCallback(
                    callback_name="langfuse_otel",
                    callback_type="success_and_failure",
                    callback_vars=KeyLoggingCallbackVars(
                        langfuse_public_key=self.public_key,
                        langfuse_secret_key=self.secret_key,
                        langfuse_host=self.host,
                    ),
                )
            ]
        )


def load_langfuse_creds() -> LangfuseCreds:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "").rstrip("/")
    if not (public_key and secret_key and host):
        pytest.fail(
            "Langfuse e2e requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and "
            "LANGFUSE_BASE_URL (or LANGFUSE_HOST); missing credentials is a hard failure, not a skip"
        )
    return LangfuseCreds(public_key=public_key, secret_key=secret_key, host=host)


def observation_spend(obs: LangfuseObservation) -> float | None:
    """Langfuse calculatedTotalCost is populated from StandardLogging response_cost."""
    return obs.calculated_total_cost


def costs_agree(expected: float, actual: float, *, rel_tol: float = 0.05) -> bool:
    """Costs agree within 5% relative (or 1e-9 absolute for near-zero)."""
    return abs(expected - actual) <= max(1e-9, abs(expected) * rel_tol)


_COMPLETION_BODY_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def completion_response_id(body: str) -> str | None:
    """SpendLogs.request_id is the chat completion body id, not x-litellm-call-id."""
    if not body or body == "<streamed>":
        return None
    try:
        parsed = _COMPLETION_BODY_ADAPTER.validate_json(body)
    except ValidationError:
        return None
    raw = parsed.get("id")
    return raw if isinstance(raw, str) and raw else None


def _matches_run(obs: LangfuseObservation, *, key_alias: str, prompt_marker: str) -> bool:
    """Match a Langfuse generation for this run.

    langfuse_otel names generations ``litellm_request`` (not ``litellm:{alias}``).
    Prefer the unique prompt marker in input; fall back to key alias in metadata
    (user_api_key_alias) or the classic SDK generation name.
    """
    if prompt_marker and prompt_marker in json.dumps(obs.input, default=str):
        return True
    meta_blob = json.dumps(obs.metadata, default=str) if obs.metadata is not None else ""
    if key_alias and key_alias in meta_blob:
        return True
    if obs.name == f"litellm:{key_alias}":
        return True
    return False


def observation_mentions_tool(obs: LangfuseObservation, tool_name: str) -> bool:
    blob = json.dumps(
        {"input": obs.input, "output": obs.output, "metadata": obs.metadata},
        default=str,
    )
    return tool_name in blob


def observation_has_guardrail(obs: LangfuseObservation, *, guardrail_name: str) -> bool:
    blob = json.dumps(obs.metadata, default=str) if obs.metadata is not None else ""
    if guardrail_name in blob or "guardrail" in blob.lower():
        return True
    if obs.name is not None and "guardrail" in obs.name.lower():
        return True
    return False


@dataclass(frozen=True, slots=True)
class LoggingClient:
    gateway: Gateway

    def key_with_alias(
        self,
        alias: str,
        *,
        models: list[str],
        team_id: str | None = None,
        user_id: str | None = None,
        organization_id: str | None = None,
        metadata: KeyMetadata | None = None,
    ) -> str:
        return self.gateway.generate_key(
            KeyGenerateBody(
                key_alias=alias,
                models=models,
                user_id=user_id or f"e2e-{alias}",
                team_id=team_id,
                organization_id=organization_id,
                metadata=metadata,
            )
        )

    def delete_key(self, key: str) -> None:
        self.gateway.delete_key(key)

    def create_team(
        self,
        alias: str,
        *,
        models: list[str],
        organization_id: str | None = None,
    ) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/team/new",
                headers=self.gateway.transport.master,
                json=TeamNewBody(
                    team_alias=alias,
                    models=models,
                    organization_id=organization_id,
                ),
                response_type=TeamNewResponse,
            )
        ).team_id

    def delete_team(self, team_id: str) -> None:
        _ = self.gateway.transport.post(
            "/team/delete",
            headers=self.gateway.transport.master,
            json=TeamDeleteBody(team_ids=[team_id]),
            response_type=NoBody,
        )

    def create_user(self, *, user_email: str, user_id: str | None = None) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/user/new",
                headers=self.gateway.transport.master,
                json=UserNewBody(
                    user_email=user_email,
                    user_role="internal_user",
                    user_id=user_id,
                ),
                response_type=UserNewResponse,
            )
        ).user_id

    def delete_user(self, user_id: str) -> None:
        _ = self.gateway.transport.post(
            "/user/delete",
            headers=self.gateway.transport.master,
            json=UserDeleteBody(user_ids=[user_id]),
            response_type=NoBody,
        )

    def create_org(self, alias: str, *, models: list[str]) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/organization/new",
                headers=self.gateway.transport.master,
                json=OrgNewBody(organization_alias=alias, models=models),
                response_type=OrgNewResponse,
            )
        ).organization_id

    def delete_org(self, organization_id: str) -> None:
        _ = self.gateway.transport.delete(
            "/organization/delete",
            headers=self.gateway.transport.master,
            json=OrgDeleteBody(organization_ids=[organization_id]),
            response_type=NoBody,
        )

    def add_team_langfuse_callback(
        self,
        team_id: str,
        creds: LangfuseCreds,
        *,
        callback_type: Literal["success", "failure", "success_and_failure"] = "success_and_failure",
    ) -> None:
        response = unwrap(
            self.gateway.transport.post(
                f"/team/{team_id}/callback",
                headers=self.gateway.transport.master,
                json=TeamCallbackBody(
                    callback_name="langfuse_otel",
                    callback_type=callback_type,
                    callback_vars=creds.callback_vars(),
                ),
                response_type=TeamCallbackResponse,
            )
        )
        assert response.status == "success", (
            f"POST /team/{team_id}/callback must return status=success; got {response.status!r}"
        )

    def create_tool_permission_guardrail(self, name: str, *, allowed_tool: str) -> str:
        """Register a tool_permission guardrail that allows one tool and denies the rest."""
        response = unwrap(
            self.gateway.transport.post(
                "/guardrails",
                headers=self.gateway.transport.master,
                json=CreateGuardrailBody(
                    guardrail=GuardrailSpec(
                        guardrail_name=name,
                        litellm_params=GuardrailLitellmParams(
                            guardrail="tool_permission",
                            mode="post_call",
                            default_on=False,
                            default_action="deny",
                            on_disallowed_action="block",
                            rules=[
                                {
                                    "id": "allow-named-tool",
                                    "tool_name": allowed_tool,
                                    "decision": "allow",
                                }
                            ],
                        ),
                    )
                ),
                response_type=CreateGuardrailResponse,
            )
        )
        guardrail_id = response.guardrail_id
        assert guardrail_id, f"create guardrail returned no id: {response!r}"
        return guardrail_id

    def delete_guardrail(self, guardrail_id: str) -> None:
        _ = self.gateway.transport.delete(
            f"/guardrails/{guardrail_id}",
            headers=self.gateway.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def create_model(self, model_name: str, litellm_params: LiteLLMParamsBody) -> str:
        return self.gateway.create_model(model_name, litellm_params)

    def delete_model(self, model_id: str) -> None:
        self.gateway.delete_model(model_id)

    def chat(self, key: str, model: str, text: str) -> ChatResponse:
        return unwrap(
            self.gateway.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=text)],
                    max_tokens=64,
                ),
            )
        )

    def chat_raw(
        self,
        key: str,
        model: str,
        text: str,
        *,
        stream: bool = False,
        tools: list[ChatTool] | None = None,
        tool_choice: str | None = None,
        guardrails: list[str] | None = None,
        max_tokens: int = 64,
    ) -> StreamingResponse:
        body = ChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=text)],
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            guardrails=guardrails,
        )
        if stream:
            return self.gateway.chat_stream(key, body)
        return self.gateway.transport.send(
            "/chat/completions",
            headers=self.gateway.transport.bearer(key),
            json=body,
        )

    def messages_raw(self, key: str, model: str, text: str, *, max_tokens: int = 16) -> StreamingResponse:
        """Non-streaming POST /v1/messages (Anthropic-native body): raw outcome
        judged by status/body/headers, for tests that need x-litellm-call-id."""
        return self.gateway.transport.send(
            "/v1/messages",
            headers=self.gateway.transport.bearer(key),
            json=AnthropicMessagesBody(
                model=model,
                max_tokens=max_tokens,
                messages=[ChatMessage(role="user", content=text)],
            ),
        )

    def responses_raw(
        self, key: str, model: str, text: str, *, max_output_tokens: int = 64
    ) -> StreamingResponse:
        """Non-streaming POST /v1/responses (OpenAI Responses API): raw outcome
        judged by status/body/headers, for tests that need x-litellm-call-id.
        max_output_tokens caps reasoning-model output cost; a capped response is
        still a 200 and still exports the trace."""
        return self.gateway.transport.send(
            "/v1/responses",
            headers=self.gateway.transport.bearer(key),
            json=ResponsesRequestBody(model=model, input=text, max_output_tokens=max_output_tokens),
        )

    def scrape_metrics(self) -> str:
        return self.gateway.probe("/metrics", params=NoBody()).body

    def poll_proxy_spend_for_key(
        self,
        key: str,
        *,
        response_id: str | None = None,
        require_positive_spend: bool = True,
    ) -> SpendLogRow | None:
        """Poll /spend/logs by virtual key.

        When ``response_id`` is set, only that SpendLogs.request_id may match.
        When unset, any positive-spend row for the key is accepted. Never falls
        back to an unmatched row; missing match returns None.
        """

        def _matches(row: SpendLogRow) -> bool:
            if response_id is not None and row.request_id != response_id:
                return False
            if require_positive_spend and not (row.spend is not None and row.spend > 0):
                return False
            return True

        rows = self.gateway.poll_logs_for_key(
            key, min_rows=1, predicate=lambda rs: any(_matches(r) for r in rs)
        )
        for row in rows:
            if _matches(row):
                return row
        return None

    def list_langfuse_observations(
        self,
        creds: LangfuseCreds,
        *,
        trace_id: str | None = None,
        name: str | None = None,
        from_start_time: str | None = None,
    ) -> list[LangfuseObservation]:
        result = get(
            URL(f"{creds.host}/api/public/observations"),
            headers=creds.auth_headers,
            params=LangfuseListParams(
                limit=100,
                traceId=trace_id,
                name=name,
                fromStartTime=from_start_time,
            ),
            response_type=LangfuseObservationList,
            timeout=30.0,
        )
        match result:
            case Success(data=page):
                return page.data
            case _:
                return []

    def find_langfuse_observation(
        self,
        creds: LangfuseCreds,
        *,
        key_alias: str,
        prompt_marker: str,
    ) -> LangfuseObservation | None:
        # langfuse_otel generations are named litellm_request; classic SDK used
        # litellm:{key_alias}. Search both, then a recent unfiltered page.
        for name in ("litellm_request", f"litellm:{key_alias}"):
            for obs in self.list_langfuse_observations(creds, name=name):
                if _matches_run(obs, key_alias=key_alias, prompt_marker=prompt_marker):
                    return obs
        for obs in self.list_langfuse_observations(creds):
            if _matches_run(obs, key_alias=key_alias, prompt_marker=prompt_marker):
                return obs
        return None

    def poll_langfuse_observation(
        self,
        creds: LangfuseCreds,
        *,
        key_alias: str,
        prompt_marker: str,
        require_positive_cost: bool = False,
    ) -> LangfuseObservation | None:
        deadline = time.monotonic() + POLL_TIMEOUT
        last: LangfuseObservation | None = None
        while time.monotonic() < deadline:
            last = self.find_langfuse_observation(
                creds, key_alias=key_alias, prompt_marker=prompt_marker
            )
            if last is not None:
                cost = observation_spend(last)
                if not require_positive_cost or (cost is not None and cost > 0):
                    return last
            time.sleep(POLL_INTERVAL)
        return last

    def poll_langfuse_trace_observations(
        self,
        creds: LangfuseCreds,
        *,
        key_alias: str,
        prompt_marker: str,
    ) -> list[LangfuseObservation]:
        """Generation plus any sibling/child observations (guardrail spans, etc.)."""
        gen = self.poll_langfuse_observation(
            creds, key_alias=key_alias, prompt_marker=prompt_marker
        )
        if gen is None or not gen.trace_id:
            return [] if gen is None else [gen]
        return self.list_langfuse_observations(creds, trace_id=gen.trace_id) or [gen]


def build_logging_client() -> LoggingClient:
    return LoggingClient(gateway=build_gateway())

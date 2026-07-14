"""Client for the logging e2e suite.

The suite drives the otel_v2 logging pipeline; providers covered are langfuse,
arize_phoenix, datadog, and prometheus.

Holds the shared Gateway so the ResourceManager cleans up keys, teams, users,
orgs, and models it creates.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, TypeAdapter, ValidationError

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
    AnthropicTool,
    AnthropicToolChoice,
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

_PHOENIX_MAX_PAGES = 20

CLAUDE_CODE_TOOLS = [
    AnthropicTool(
        name="Read",
        description="Read a file from the local filesystem",
        input_schema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    ),
    AnthropicTool(
        name="Bash",
        description="Execute a bash command and return its output",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    ),
]


class TeamCallbackBody(BaseModel):
    callback_name: Literal["langfuse_otel", "langfuse", "langsmith", "gcs"]
    callback_type: Literal["success", "failure", "success_and_failure"]
    callback_vars: dict[str, str]


class TeamCallbackResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str


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


@dataclass(frozen=True, slots=True)
class PhoenixCreds:
    base_url: str
    project: str
    api_key: str | None

    @property
    def auth_headers(self) -> AuthHeaders:
        if self.api_key is None:
            return AuthHeaders()
        return AuthHeaders(authorization=f"Bearer {self.api_key}")


def load_phoenix_creds() -> PhoenixCreds:
    """Reads the exact env vars the proxy's arize_phoenix integration reads, so
    the read-back always targets the same Phoenix host and project the proxy
    ships traces to (locally and on the EKS cluster)."""
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        pytest.fail(
            "Arize Phoenix e2e requires PHOENIX_COLLECTOR_ENDPOINT; "
            "missing credentials is a hard failure, not a skip"
        )
    return PhoenixCreds(
        base_url=endpoint.rstrip("/").removesuffix("/v1/traces"),
        project=os.getenv("PHOENIX_PROJECT_NAME") or "default",
        api_key=os.getenv("PHOENIX_API_KEY"),
    )


def phoenix_span_blob(span: dict[str, JsonValue]) -> str:
    return json.dumps(span.get("attributes"), default=str)


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
        max_tokens: int = 64,
    ) -> StreamingResponse:
        body = ChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=text)],
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
        )
        if stream:
            return self.gateway.chat_stream(key, body)
        return self.gateway.transport.send(
            "/chat/completions",
            headers=self.gateway.transport.bearer(key),
            json=body,
        )

    def messages_raw(
        self,
        key: str,
        model: str,
        text: str,
        *,
        stream: bool = False,
        tools: list[AnthropicTool] | None = None,
        tool_choice: AnthropicToolChoice | None = None,
        max_tokens: int = 64,
    ) -> StreamingResponse:
        body = AnthropicMessagesBody(
            model=model,
            messages=[ChatMessage(role="user", content=text)],
            max_tokens=max_tokens,
            stream=True if stream else None,
            tools=tools,
            tool_choice=tool_choice,
        )
        if stream:
            return self.gateway.transport.stream(
                "/v1/messages", headers=self.gateway.transport.bearer(key), json=body
            )
        return self.gateway.transport.send(
            "/v1/messages", headers=self.gateway.transport.bearer(key), json=body
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

    def list_phoenix_spans(
        self,
        creds: PhoenixCreds,
        *,
        since: str | None = None,
    ) -> list[dict[str, JsonValue]]:
        """Every span Phoenix holds for the project (paged read of the
        /v1/projects/{project}/spans REST route), newest window bounded by ``since``."""
        spans: list[dict[str, JsonValue]] = []
        cursor: str | None = None
        for _ in range(_PHOENIX_MAX_PAGES):
            query = {"limit": "100"} | ({"cursor": cursor} if cursor else {}) | ({"start_time": since} if since else {})
            result = get(
                URL(f"{creds.base_url}/v1/projects/{creds.project}/spans"),
                headers=creds.auth_headers,
                params=RootModel[dict[str, str]].model_validate(query),
                response_type=RootModel[JsonValue],
                timeout=30.0,
            )
            if not isinstance(result, Success):
                break
            page = result.data.root
            if not isinstance(page, dict):
                break
            data = page.get("data")
            if isinstance(data, list):
                spans.extend(span for span in data if isinstance(span, dict))
            next_cursor = page.get("next_cursor")
            if not isinstance(next_cursor, str):
                break
            cursor = next_cursor
        return spans

    def find_phoenix_spans(
        self,
        creds: PhoenixCreds,
        *,
        marker: str,
        since: str | None = None,
    ) -> list[dict[str, JsonValue]]:
        """LLM generation spans carrying ``marker``. Only generation spans hold
        message content, so the marker match excludes the request's child spans
        (db writes, cache reads) that share the LLM span kind."""
        return [
            span
            for span in self.list_phoenix_spans(creds, since=since)
            if span.get("span_kind") == "LLM" and marker in phoenix_span_blob(span)
        ]

    def poll_phoenix_spans(
        self,
        creds: PhoenixCreds,
        *,
        marker: str,
        min_count: int = 1,
        since: str | None = None,
    ) -> list[dict[str, JsonValue]]:
        """Poll until at least ``min_count`` generation spans carry ``marker``
        in their attributes, or the poll budget runs out."""
        deadline = time.monotonic() + POLL_TIMEOUT
        matched: list[dict[str, JsonValue]] = []
        while time.monotonic() < deadline:
            matched = self.find_phoenix_spans(creds, marker=marker, since=since)
            if len(matched) >= min_count:
                return matched
            time.sleep(POLL_INTERVAL)
        return matched

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
        """Generation plus any sibling/child observations."""
        gen = self.poll_langfuse_observation(
            creds, key_alias=key_alias, prompt_marker=prompt_marker
        )
        if gen is None or not gen.trace_id:
            return [] if gen is None else [gen]
        return self.list_langfuse_observations(creds, trace_id=gen.trace_id) or [gen]


def build_logging_client() -> LoggingClient:
    client = LoggingClient(gateway=build_gateway())
    client.create_model(
        "bedrock/us.anthropic.claude-sonnet-5",
        LiteLLMParamsBody(model="bedrock/us.anthropic.claude-sonnet-5"),
    )
    client.create_model(
        "bedrock/us.anthropic.claude-opus-4-8",
        LiteLLMParamsBody(model="bedrock/us.anthropic.claude-opus-4-8"),
    )
    client.create_model(
        "anthropic/claude-sonnet-5",
        LiteLLMParamsBody(model="anthropic/claude-sonnet-5", api_key="os.environ/ANTHROPIC_API_KEY"),
    )
    return client

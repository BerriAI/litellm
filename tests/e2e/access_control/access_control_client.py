"""Client for the access-control e2e suite."""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from proxy_client import ProxyClient
from e2e_http import NoBody, StreamingResponse, is_ok, unwrap
from models import (
    ChatBody,
    ChatMessage,
    ChatMetadata,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ModelInfoBody,
    ModelNewBody,
    TagDeleteBody,
    TagInfoBody,
    TagInfoResponse,
    TagNewBody,
    TagUpdateBody,
    TeamDeleteBody,
    TeamInfoParams,
    TeamInfoResponse,
    TeamNewBody,
    TeamNewResponse,
    TeamUpdateBody,
)

MODEL_ACCESS_DENIED_MARKER = "key_model_access_denied"
TEAM_MODEL_ACCESS_DENIED_MARKER = "team_model_access_denied"
ROUTE_NOT_ALLOWED_MARKER = "not allowed to call this route"
TAG_OWNER_DENIED_MARKER = "can only be sent by keys of that team"


class ApiErrorDetail(BaseModel):
    message: str | None = None
    type: str | None = None
    code: str | int | None = None


class ApiErrorEnvelope(BaseModel):
    error: ApiErrorDetail


class AccessGroupInfoResponse(BaseModel):
    """GET /access_group/{name}/info: the deployments a model access group grants."""

    access_group: str
    model_names: list[str]
    deployment_count: int


def error_envelope(body: str) -> ApiErrorEnvelope | None:
    """The OpenAI-shaped `{"error": {...}}` a client parses, or None if absent."""
    try:
        return ApiErrorEnvelope.model_validate_json(body)
    except ValidationError:
        return None


@dataclass(frozen=True, slots=True)
class AccessControlClient:
    proxy: ProxyClient

    def llm_only_key(self) -> str:
        return self.proxy.generate_key(KeyGenerateBody(models=[], allowed_routes=["llm_api_routes"]))

    def delete_key(self, key: str) -> None:
        self.proxy.delete_key(key)

    def chat_status(
        self, key: str, model: str, content: str, max_completion_tokens: int | None = None
    ) -> StreamingResponse:
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_completion_tokens=max_completion_tokens,
            ),
        )

    def create_team(self, team_alias: str, models: list[str]) -> str:
        team_id = unwrap(
            self.proxy.transport.post(
                "/team/new",
                headers=self.proxy.transport.master,
                json=TeamNewBody(team_alias=team_alias, models=models),
                response_type=TeamNewResponse,
            )
        ).team_id
        self._await_team(team_id)
        return team_id

    def set_team_models(self, team_id: str, team_alias: str, models: list[str]) -> None:
        """Replace the team's allow-list. /model/new appends a team-scoped deployment's
        public name to it, so a test that means to grant only an access group has to
        put the allow-list back afterwards."""
        _ = unwrap(
            self.proxy.transport.post(
                "/team/update",
                headers=self.proxy.transport.master,
                json=TeamUpdateBody(team_id=team_id, team_alias=team_alias, models=models),
                response_type=NoBody,
            )
        )

    def delete_team(self, team_id: str) -> None:
        _ = self.proxy.transport.post(
            "/team/delete",
            headers=self.proxy.transport.master,
            json=TeamDeleteBody(team_ids=[team_id]),
            response_type=NoBody,
        )

    def access_group_info(self, access_group: str) -> AccessGroupInfoResponse | None:
        result = self.proxy.transport.get(
            f"/access_group/{access_group}/info",
            headers=self.proxy.transport.master,
            params=NoBody(),
            response_type=AccessGroupInfoResponse,
        )
        return unwrap(result) if is_ok(result) else None

    def team_models(self, team_id: str) -> list[str] | None:
        result = self.proxy.transport.get(
            "/team/info",
            headers=self.proxy.transport.master,
            params=TeamInfoParams(team_id=team_id),
            response_type=TeamInfoResponse,
        )
        return unwrap(result).team_info.models if is_ok(result) else None

    def _await_team(self, team_id: str) -> None:
        deadline = time.monotonic() + self.proxy.poll_timeout
        while time.monotonic() < deadline:
            if self.team_models(team_id) is not None:
                return
            time.sleep(self.proxy.poll_interval)
        raise AssertionError(f"/team/info never resolved team {team_id!r} created by /team/new")

    def team_key(self, team_id: str) -> str:
        return self.proxy.generate_key(KeyGenerateBody(models=[], team_id=team_id))

    def tagged_chat_status(self, key: str, model: str, content: str, tags: list[str]) -> StreamingResponse:
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                metadata=ChatMetadata(tags=tags),
                max_completion_tokens=16,
            ),
        )

    def create_owned_tag(self, name: str, team_id: str) -> None:
        resp = self.proxy.transport.send(
            "/tag/new", headers=self.proxy.transport.master, json=TagNewBody(name=name, team_id=team_id)
        )
        assert resp.ok, resp.body

    def update_tag_owner_status(self, key: str | None, name: str, team_id: str | None) -> StreamingResponse:
        """`key=None` sends the update as the proxy admin (master key). `team_id=None`
        is sent as an explicit null, which releases the owner."""
        headers = self.proxy.transport.master if key is None else self.proxy.transport.bearer(key)
        return self.proxy.transport.send("/tag/update", headers=headers, json=TagUpdateBody(name=name, team_id=team_id))

    def tag_owner(self, name: str) -> str | None:
        info = unwrap(
            self.proxy.transport.post(
                "/tag/info",
                headers=self.proxy.transport.master,
                json=TagInfoBody(names=[name]),
                response_type=TagInfoResponse,
            )
        )
        return info.root[name].team_id

    def delete_tag(self, name: str) -> None:
        _ = self.proxy.transport.post(
            "/tag/delete",
            headers=self.proxy.transport.master,
            json=TagDeleteBody(name=name),
            response_type=NoBody,
        )

    def create_model_status(self, key: str, model_name: str) -> StreamingResponse:
        return self.proxy.transport.send(
            "/model/new",
            headers=self.proxy.transport.bearer(key),
            json=ModelNewBody(
                model_name=model_name,
                litellm_params=LiteLLMParamsBody(model="openai/gpt-4o-mini"),
                model_info=ModelInfoBody(id=model_name),
            ),
        )


def build_client(proxy: ProxyClient) -> AccessControlClient:
    return AccessControlClient(proxy=proxy)

"""Client for the management-routes e2e suite: the shared Gateway plus the
key/team/user/organization writes, the info/list read-backs the tests assert,
and the raw-status calls judged by HTTP outcome (chat under a scoped key, an
llm-only key hitting a management route).
"""

from __future__ import annotations

from dataclasses import dataclass

from e2e_gateway import Gateway, build_gateway
from e2e_http import NoBody, ProbeResult, StreamingResponse, unwrap
from models import (
    ChatBody,
    ChatMessage,
    KeyDeleteBody,
    KeyGenerateBody,
    KeyListParams,
    KeyListResponse,
    KeyUpdateBody,
    OrgDeleteBody,
    OrgInfoParams,
    OrgInfoResponse,
    OrgNewBody,
    OrgNewResponse,
    TeamData,
    TeamDeleteBody,
    TeamInfoParams,
    TeamInfoResponse,
    TeamMemberAddBody,
    TeamMemberDeleteBody,
    TeamMemberEntry,
    TeamNewBody,
    TeamNewResponse,
    UserDeleteBody,
    UserInfoParams,
    UserInfoResponse,
    UserListParams,
    UserListResponse,
    UserNewBody,
    UserNewResponse,
)

MODEL_ACCESS_DENIED_MARKER = "key_model_access_denied"
ROUTE_NOT_ALLOWED_MARKER = "not allowed to call this route"


@dataclass(frozen=True, slots=True)
class ManagementClient:
    gateway: Gateway

    def llm_only_key(self) -> str:
        return self.gateway.generate_key(KeyGenerateBody(models=[], allowed_routes=["llm_api_routes"]))

    def update_key_models(self, key: str, models: list[str]) -> None:
        _ = unwrap(
            self.gateway.transport.post(
                "/key/update",
                headers=self.gateway.transport.master,
                json=KeyUpdateBody(key=key, models=models),
                response_type=NoBody,
            )
        )

    def delete_key_strict(self, key: str) -> None:
        """Strict delete for the act phase of a test: a failed delete is a hard
        failure, unlike the warn-only Gateway.delete_key used at teardown."""
        _ = unwrap(
            self.gateway.transport.post(
                "/key/delete",
                headers=self.gateway.transport.master,
                json=KeyDeleteBody(keys=[key]),
                response_type=NoBody,
            )
        )

    def key_alias_count(self, key_alias: str) -> int:
        return unwrap(
            self.gateway.transport.get(
                "/key/list",
                headers=self.gateway.transport.master,
                params=KeyListParams(key_alias=key_alias),
                response_type=KeyListResponse,
            )
        ).total_count

    def create_team(self, body: TeamNewBody) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/team/new",
                headers=self.gateway.transport.master,
                json=body,
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

    def team_info(self, team_id: str) -> TeamData:
        return unwrap(
            self.gateway.transport.get(
                "/team/info",
                headers=self.gateway.transport.master,
                params=TeamInfoParams(team_id=team_id),
                response_type=TeamInfoResponse,
            )
        ).team_info

    def team_info_status(self, team_id: str) -> ProbeResult:
        return self.gateway.transport.probe("/team/info", params=TeamInfoParams(team_id=team_id))

    def add_team_member(self, team_id: str, user_id: str) -> None:
        _ = unwrap(
            self.gateway.transport.post(
                "/team/member_add",
                headers=self.gateway.transport.master,
                json=TeamMemberAddBody(team_id=team_id, member=TeamMemberEntry(role="user", user_id=user_id)),
                response_type=NoBody,
            )
        )

    def delete_team_member(self, team_id: str, user_id: str) -> None:
        _ = unwrap(
            self.gateway.transport.post(
                "/team/member_delete",
                headers=self.gateway.transport.master,
                json=TeamMemberDeleteBody(team_id=team_id, user_id=user_id),
                response_type=NoBody,
            )
        )

    def create_user(self, body: UserNewBody) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/user/new",
                headers=self.gateway.transport.master,
                json=body,
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

    def user_info(self, user_id: str) -> UserInfoResponse:
        return unwrap(
            self.gateway.transport.get(
                "/user/info",
                headers=self.gateway.transport.master,
                params=UserInfoParams(user_id=user_id),
                response_type=UserInfoResponse,
            )
        )

    def user_count(self, user_id: str) -> int:
        return unwrap(
            self.gateway.transport.get(
                "/user/list",
                headers=self.gateway.transport.master,
                params=UserListParams(user_ids=user_id),
                response_type=UserListResponse,
            )
        ).total

    def create_org(self, body: OrgNewBody) -> str:
        return unwrap(
            self.gateway.transport.post(
                "/organization/new",
                headers=self.gateway.transport.master,
                json=body,
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

    def org_info(self, organization_id: str) -> OrgInfoResponse:
        return unwrap(
            self.gateway.transport.get(
                "/organization/info",
                headers=self.gateway.transport.master,
                params=OrgInfoParams(organization_id=organization_id),
                response_type=OrgInfoResponse,
            )
        )

    def chat_status(self, key: str, model: str, content: str) -> StreamingResponse:
        return self.gateway.transport.send(
            "/chat/completions",
            headers=self.gateway.transport.bearer(key),
            json=ChatBody(model=model, messages=[ChatMessage(role="user", content=content)], max_tokens=16),
        )

    def key_generate_status(self, key: str, body: KeyGenerateBody) -> StreamingResponse:
        return self.gateway.transport.send("/key/generate", headers=self.gateway.transport.bearer(key), json=body)

    def team_new_status(self, key: str, body: TeamNewBody) -> StreamingResponse:
        return self.gateway.transport.send("/team/new", headers=self.gateway.transport.bearer(key), json=body)

    def user_new_status(self, key: str, body: UserNewBody) -> StreamingResponse:
        return self.gateway.transport.send("/user/new", headers=self.gateway.transport.bearer(key), json=body)


def build_client() -> ManagementClient:
    return ManagementClient(gateway=build_gateway())

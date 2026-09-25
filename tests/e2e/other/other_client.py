"""Client for the `other` holding-pen suite: the auth gate (master key vs an
invalid key on an admin route), JWT auth against the suite's Keycloak realm
(idp.py), and the process-lifecycle health probes (liveness, public readiness,
authenticated readiness diagnostics).

Holds the shared ProxyClient so `resources` / `scoped_key` still clean up, and
adds only the routes these behaviors need. The health probes deliberately send
no auth header (public routes), so they go through the transport with an empty
headers model rather than a bearer. JWT tests reach the identity provider
through `idp`, which provisions identities and mints tokens through Keycloak's
own endpoints, so no test ever holds a signing key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from e2e_http import AnthropicHeaders, AuthHeaders, NoBody, ProbeResult, Result
from idp import Keycloak, keycloak_from_env
from models import (
    ChatBody,
    ChatResponse,
    JwtKeyMappingDeleteBody,
    JwtKeyMappingDeleteResponse,
    JwtKeyMappingListParams,
    JwtKeyMappingListResponse,
    ModelsListParams,
    ModelsListResponse,
    ReadinessDetailsResponse,
    ReadinessResponse,
    UserInfoParams,
    UserInfoWithKeysResponse,
    UserListParams,
    UserListResponse,
    UserNewBody,
    UserNewResponse,
)
from proxy_client import ProxyClient
from pydantic import Field


class TeamHeaders(AuthHeaders):
    """Bearer auth plus ``x-litellm-team-id``, the header a JWT caller sends to
    pick one of the teams it belongs to."""

    x_litellm_team_id: str = Field(serialization_alias="x-litellm-team-id")


@dataclass(frozen=True, slots=True)
class OtherClient:
    proxy: ProxyClient

    @property
    def idp(self) -> Keycloak:
        """Resolved per use, so the suite's non-JWT tests never need the IdP env."""
        return keycloak_from_env()

    def liveness(self) -> ProbeResult:
        """GET /health/liveliness. Unauthenticated; the probe returns status +
        raw body so the test can assert the worker reports itself alive."""
        return self.proxy.transport.probe("/health/liveliness", params=NoBody())

    def readiness_public(self) -> Result[ReadinessResponse]:
        """GET /health/readiness with no credential at all, proving the probe is
        safe to expose to an unauthenticated load balancer."""
        return self.proxy.transport.get(
            "/health/readiness",
            headers=NoBody(),
            params=NoBody(),
            response_type=ReadinessResponse,
        )

    def readiness_details(self, key: str) -> Result[ReadinessDetailsResponse]:
        return self.proxy.transport.get(
            "/health/readiness/details",
            headers=self.proxy.transport.bearer(key),
            params=NoBody(),
            response_type=ReadinessDetailsResponse,
        )

    def readiness_details_unauthenticated(self) -> Result[ReadinessDetailsResponse]:
        return self.proxy.transport.get(
            "/health/readiness/details",
            headers=NoBody(),
            params=NoBody(),
            response_type=ReadinessDetailsResponse,
        )

    def user_new(self, body: UserNewBody) -> Result[UserNewResponse]:
        """POST /user/new under the master key: seed the litellm user a JWT
        `sub` claim resolves to, before that token ever reaches the proxy."""
        return self.proxy.transport.post(
            "/user/new",
            headers=self.proxy.transport.master,
            json=body,
            response_type=UserNewResponse,
        )

    def user_info(self, user_id: str) -> Result[UserInfoWithKeysResponse]:
        """GET /user/info under the master key. Only the user's key rows are
        modelled: `token` is the stored key hash, never the plaintext key."""
        return self.proxy.transport.get(
            "/user/info",
            headers=self.proxy.transport.master,
            params=UserInfoParams(user_id=user_id),
            response_type=UserInfoWithKeysResponse,
        )

    def jwt_mapping_list(self) -> Result[JwtKeyMappingListResponse]:
        """GET /jwt/key/mapping/list under the master key."""
        return self.proxy.transport.get(
            "/jwt/key/mapping/list",
            headers=self.proxy.transport.master,
            params=JwtKeyMappingListParams(size=100),
            response_type=JwtKeyMappingListResponse,
        )

    def jwt_mapping_delete(self, mapping_id: str) -> Result[JwtKeyMappingDeleteResponse]:
        """POST /jwt/key/mapping/delete under the master key."""
        return self.proxy.transport.post(
            "/jwt/key/mapping/delete",
            headers=self.proxy.transport.master,
            json=JwtKeyMappingDeleteBody(id=mapping_id),
            response_type=JwtKeyMappingDeleteResponse,
        )

    def chat_as_team(self, token: str, team: str, body: ChatBody) -> Result[ChatResponse]:
        """POST /chat/completions under `token` with `x-litellm-team-id: team`."""
        return self.proxy.transport.post(
            "/chat/completions",
            headers=TeamHeaders(
                authorization=self.proxy.transport.bearer(token).authorization,
                x_litellm_team_id=team,
            ),
            json=body,
            response_type=ChatResponse,
        )

    def list_models_as(self, token: str, *, anthropic: bool = False) -> Result[ModelsListResponse]:
        """GET /v1/models under `token`, in the OpenAI shape or, with `anthropic`, the
        Anthropic Models API shape Claude Code reads. Both carry `data[].id`."""
        bearer: Final = self.proxy.transport.bearer(token)
        return self.proxy.transport.get(
            "/v1/models",
            headers=AnthropicHeaders(authorization=bearer.authorization) if anthropic else bearer,
            params=ModelsListParams(return_wildcard_routes=False),
            response_type=ModelsListResponse,
        )

    def list_users_as(self, key: str) -> Result[UserListResponse]:
        """GET /user/list under `key`. Admin-only, so it doubles as the master
        key's authorization proof: the master key (proxy admin) reads it, a
        non-matching key is rejected before it ever reaches the handler."""
        return self.proxy.transport.get(
            "/user/list",
            headers=self.proxy.transport.bearer(key),
            params=UserListParams(user_ids="e2e-test-user"),
            response_type=UserListResponse,
        )


def build_client(proxy: ProxyClient) -> OtherClient:
    return OtherClient(proxy=proxy)

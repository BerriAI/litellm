"""Live e2e: authenticating to the gateway with a JWT from a real OIDC identity
provider (Keycloak), the way an enterprise fronts the proxy with its IdP instead
of virtual keys.

The proxy runs with enable_jwt_auth and a litellm_jwtauth.issuers entry that
trusts the Keycloak realm this suite provisions (see keycloak.py / the module
docstring below for the required config). Tokens are minted from real Keycloak
clients: an admin client whose access token carries the litellm_proxy_admin scope,
and a team client whose token carries a hardcoded team_id claim. A token signed by
a key outside the IdP's JWKS, and an expired one, exercise the rejection paths.

Proxy config this suite requires (general_settings), issuer/JWKS matching
KEYCLOAK_URL + KEYCLOAK_REALM:

    enable_jwt_auth: true
    litellm_jwtauth:
      admin_jwt_scope: "litellm_proxy_admin"
      team_id_jwt_field: "team_id"
      issuers:
        - issuer: "http://localhost:8080/realms/litellm-e2e"
          jwks_url: "http://localhost:8080/realms/litellm-e2e/protocol/openid-connect/certs"
          disable_audience_validation: true

Enterprise-gated: JWT auth needs a licensed proxy (LITELLM_LICENSE).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from e2e_config import unique_marker
from e2e_http import Success, UnauthorizedError
from jwt_auth_client import JWTAuthClient
from keycloak import JWT_TEAM_ID, KeycloakEnv, mint_untrusted_jwt
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import LiteLLMParamsBody, TeamNewBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

_TEAM_DENIAL_MARKER = "team_model_access_denied"


def _poll[T](proxy: ProxyClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + proxy.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(proxy.poll_interval)
    pytest.fail(failure)


def _setup_jwt_team(client: ManagementClient, resources: ResourceManager, marker: str) -> tuple[str, str]:
    """Register two mock deployments and bind a team (with the fixed id the Keycloak
    team token claims) to only the first, so the second proves the deny path. The
    team is deleted first to clear a leaked prior run, since its id is fixed."""
    team_model = f"jwt-team-model-{marker}"
    other_model = f"jwt-other-model-{marker}"
    team_model_id = client.proxy.create_model(
        team_model, LiteLLMParamsBody(model="openai/gpt-4o-mini", mock_response="jwt ok")
    )
    resources.defer(lambda: client.proxy.delete_model(team_model_id))
    other_model_id = client.proxy.create_model(
        other_model, LiteLLMParamsBody(model="openai/gpt-4o-mini", mock_response="jwt ok")
    )
    resources.defer(lambda: client.proxy.delete_model(other_model_id))

    client.delete_team(JWT_TEAM_ID)
    _ = client.create_team(TeamNewBody(team_alias=f"jwt-e2e-{marker}", team_id=JWT_TEAM_ID, models=[team_model]))
    resources.defer(lambda: client.delete_team(JWT_TEAM_ID))
    return team_model, other_model


class TestJWTValidToken:
    @pytest.mark.covers("other.auth.jwt.valid_token_allows")
    def test_admin_token_allows_management_route(
        self, jwt_client: JWTAuthClient, keycloak_env: KeycloakEnv
    ) -> None:
        token = keycloak_env.admin_token()
        outcome = jwt_client.get_route("/user/list", token)
        match outcome:
            case Success():
                return
            case _:
                pytest.fail(
                    f"a valid IdP admin JWT (litellm_proxy_admin scope) must be allowed on /user/list, got {outcome}"
                )

    @pytest.mark.covers("other.auth.jwt.team_model_allowed")
    def test_team_token_allows_its_model(
        self,
        client: ManagementClient,
        jwt_client: JWTAuthClient,
        keycloak_env: KeycloakEnv,
        resources: ResourceManager,
    ) -> None:
        team_model, _ = _setup_jwt_team(client, resources, unique_marker())
        token = keycloak_env.team_token()

        _ = _poll(
            client.proxy,
            lambda: True if jwt_client.chat(token, team_model, f"hi {unique_marker()}").ok else None,
            f"team JWT was never allowed to call its own model {team_model}",
        )

    @pytest.mark.covers("other.auth.jwt.team_model_denied")
    def test_team_token_denied_untrusted_model(
        self,
        client: ManagementClient,
        jwt_client: JWTAuthClient,
        keycloak_env: KeycloakEnv,
        resources: ResourceManager,
    ) -> None:
        _, other_model = _setup_jwt_team(client, resources, unique_marker())
        token = keycloak_env.team_token()

        outcome = jwt_client.chat(token, other_model, f"hi {unique_marker()}")
        assert outcome.status_code == 403, (
            f"team JWT calling a model outside its team must be denied 403, got {outcome.status_code}: "
            f"{outcome.body[:300]}"
        )
        assert _TEAM_DENIAL_MARKER in outcome.body, (
            f"the 403 must be a team model-access denial, got: {outcome.body[:300]}"
        )


class TestJWTRejection:
    @pytest.mark.covers("other.auth.jwt.invalid_signature_denied")
    def test_untrusted_signature_denied(self, jwt_client: JWTAuthClient, keycloak_env: KeycloakEnv) -> None:
        token = mint_untrusted_jwt(keycloak_env.issuer)
        outcome = jwt_client.get_route("/user/list", token)
        match outcome:
            case UnauthorizedError():
                return
            case _:
                pytest.fail(
                    f"a JWT signed by a key outside the IdP's JWKS must be rejected 401, got {outcome}"
                )

    @pytest.mark.covers("other.auth.jwt.expired_denied")
    def test_expired_token_denied(self, jwt_client: JWTAuthClient, keycloak_env: KeycloakEnv) -> None:
        """The short-lived client's token carries the admin scope, so it is accepted
        before expiry and rejected after - isolating expiry as the only thing that
        changed, with a valid signature throughout."""
        token = keycloak_env.shortlived_token()

        before = jwt_client.get_route("/user/list", token)
        match before:
            case Success():
                pass
            case _:
                pytest.fail(f"the short-lived admin JWT should be accepted before expiry, got {before}")

        _ = _poll(
            jwt_client.proxy,
            lambda: True if isinstance(jwt_client.get_route("/user/list", token), UnauthorizedError) else None,
            "the short-lived JWT was never rejected (401) after its expiry",
        )

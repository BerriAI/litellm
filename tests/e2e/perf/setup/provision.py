"""Builds the caller side of a `PerformanceTestCase` on the gateway: the teams, the Keycloak users
(JWT callers) and the virtual keys, each suffixed with the run marker and deferred to the
`ResourceManager`, so two runs never collide and a failed run still tears down. Model groups and
guardrails are not created here: they are written out in the gateway's config (the helm values in
the lane, `gateway/perf_ci_config.yml` locally), and setup only checks the deployed gateway has what
the case names."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from idp import TESTS_CLIENT_ID, Keycloak
from lifecycle import ResourceManager
from models import KeyGenerateBody, TeamNewBody, UserNewBody, UserNewResponse
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict
from setup.performance_test_case import MockProvider, PerformanceTestCase
from setup.plan import Caller
from setup.wire import TokenSource

MOCK_PROVIDERS: Final[tuple[MockProvider, ...]] = ("openai", "anthropic", "bedrock")

MOCK_URL_ENV: Final[dict[MockProvider, str]] = {
    "openai": "E2E_PERF_MOCK_OPENAI_URL",
    "anthropic": "E2E_PERF_MOCK_ANTHROPIC_URL",
    "bedrock": "E2E_PERF_MOCK_BEDROCK_URL",
}
MOCK_URL_DEFAULT: Final[dict[MockProvider, str]] = {
    "openai": "http://127.0.0.1:8090/v1",
    "anthropic": "http://127.0.0.1:8091",
    "bedrock": "http://127.0.0.1:8092",
}
GUARDRAIL_URL_ENV: Final = "E2E_PERF_MOCK_GUARDRAIL_URL"
GUARDRAIL_URL_DEFAULT: Final = "http://127.0.0.1:8093"


def mock_url(provider: MockProvider) -> str:
    """The mock's root, where its control routes live. The same variable feeds the gateway's
    `api_base`, which for OpenAI ends in `/v1`, so that suffix is dropped here."""
    url: Final = os.environ.get(MOCK_URL_ENV[provider], MOCK_URL_DEFAULT[provider]).rstrip("/")
    return url.removesuffix("/v1")


def guardrail_url() -> str:
    return os.environ.get(GUARDRAIL_URL_ENV, GUARDRAIL_URL_DEFAULT).rstrip("/")


class PerfTeamNewBody(TeamNewBody):
    """`/team/new` with the per-team settings a `Team` carries."""

    max_budget: float | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None


class ListedGuardrailParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str | list[str]


class ListedGuardrail(BaseModel):
    model_config = ConfigDict(frozen=True)

    guardrail_name: str
    litellm_params: ListedGuardrailParams


class GuardrailList(BaseModel):
    model_config = ConfigDict(frozen=True)

    guardrails: list[ListedGuardrail]


@dataclass(frozen=True, slots=True)
class CaseSetup:
    marker: str
    team_ids: dict[str, str]
    key_bearers: dict[str, str]
    token_sources: dict[str, TokenSource]
    user_ids: dict[str, str]

    def authorization(self, caller: Caller) -> str | None:
        return f"Bearer {self.key_bearers[caller.name]}" if caller.kind == "key" else None

    def token_source(self, caller: Caller) -> TokenSource | None:
        return self.token_sources[caller.name] if caller.kind == "jwt" else None


def check_deployed(proxy: ProxyClient, case: PerformanceTestCase) -> None:
    """The gateway was deployed with every model group and guardrail the case names, with the
    guardrail on the hook the case expects. A mismatch is a config drift, not a perf result."""
    deployed: Final = sorted({entry.model_name for entry in proxy.model_info()})
    missing: Final = [group for group in case.model_groups if group not in deployed]
    assert not missing, f"{case.name}: model groups {missing} are not in the gateway config (deployed: {deployed})"
    if not case.guardrails:
        return
    listed: Final = unwrap(
        proxy.transport.get(
            "/guardrails/list", headers=proxy.transport.master, params=NoBody(), response_type=GuardrailList
        )
    ).guardrails
    modes: Final = {guardrail.guardrail_name: guardrail.litellm_params.mode for guardrail in listed}
    for guardrail in case.guardrails:
        mode = modes.get(guardrail.name)
        hooks = mode if isinstance(mode, list) else [mode]
        assert guardrail.mode in hooks, (
            f"{case.name}: guardrail {guardrail.name} should be on {guardrail.mode}, gateway has {modes}"
        )


def build_setup(
    case: PerformanceTestCase, *, proxy: ProxyClient, idp: Keycloak, resources: ResourceManager
) -> CaseSetup:
    check_deployed(proxy, case)
    marker: Final = unique_marker()
    team_ids: dict[str, str] = {}
    keycloak_groups: dict[str, str] = {}
    for team in case.teams:
        team_id = f"perf-{team.name}-{marker}"
        team_ids[team.name] = proxy.create_team(
            PerfTeamNewBody(
                team_alias=team_id,
                team_id=team_id,
                models=list(team.models),
                max_budget=team.max_budget,
                rpm_limit=team.rpm_limit,
                tpm_limit=team.tpm_limit,
            )
        )
        resources.defer(lambda team_id=team_id: proxy.delete_team(team_id))
        group_id = idp.create_group(team_id)
        keycloak_groups[team_id] = group_id
        resources.defer(lambda group_id=group_id: idp.delete_group(group_id))

    user_ids: dict[str, str] = {}
    token_sources: dict[str, TokenSource] = {}
    for user in case.users:
        groups = (team_ids[user.team],) if user.team else ()
        identity = idp.provision_user(
            marker=f"{marker}-{user.name}",
            groups=groups,
            group_ids=tuple(keycloak_groups[group] for group in groups),
            defer=resources.defer,
        )
        user_ids[user.name] = identity.user_id
        token_sources[user.name] = TokenSource(
            token_url=idp.token_url(idp.realm),
            client_id=TESTS_CLIENT_ID,
            username=identity.username,
            password=identity.password,
        )
        created = unwrap(
            proxy.transport.post(
                "/user/new",
                headers=proxy.transport.master,
                json=UserNewBody(
                    user_email=f"{identity.username}@example.com",
                    user_role="internal_user",
                    user_id=identity.user_id,
                    auto_create_key=False,
                    max_budget=user.max_budget,
                ),
                response_type=UserNewResponse,
            )
        )
        resources.defer(lambda user_id=created.user_id: proxy.delete_user(user_id))

    key_bearers: dict[str, str] = {}
    for key in case.virtual_keys:
        generated = proxy.generate_key(
            KeyGenerateBody(
                models=list(key.models),
                team_id=team_ids[key.team] if key.team else None,
                user_id=user_ids[key.user] if key.user else None,
                rpm_limit=key.rpm_limit,
                max_budget=key.max_budget,
                key_alias=f"perf-{key.name}-{marker}",
            )
        )
        resources.defer(lambda generated=generated: proxy.delete_key(generated))
        key_bearers[key.name] = generated

    return CaseSetup(
        marker=marker, team_ids=team_ids, key_bearers=key_bearers, token_sources=token_sources, user_ids=user_ids
    )

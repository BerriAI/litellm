"""Builds the gateway side of a `PerformanceTestCase`: the teams, the Keycloak users (JWT callers),
the virtual keys, the model groups over the mock providers and the guardrail registrations.
Every object is suffixed with the run marker and deferred to the `ResourceManager`, so two runs
never collide and a failed run still tears down."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Final, Literal

from e2e_config import settle_propagation, unique_marker
from e2e_http import NoBody, unwrap
from idp import TESTS_CLIENT_ID, Keycloak
from lifecycle import ResourceManager
from models import KeyGenerateBody, LiteLLMParamsBody, TeamNewBody, UserNewBody, UserNewResponse
from perf_case import Guardrail, MockDeployment, PerformanceTestCase, Provider
from perf_plan import Caller
from perf_wire import TokenSource
from proxy_client import ProxyClient
from pydantic import BaseModel

MOCK_URL_ENV: Final[dict[Provider, str]] = {
    "openai": "E2E_PERF_MOCK_OPENAI_URL",
    "anthropic": "E2E_PERF_MOCK_ANTHROPIC_URL",
    "bedrock": "E2E_PERF_MOCK_BEDROCK_URL",
}
MOCK_URL_DEFAULT: Final[dict[Provider, str]] = {
    "openai": "http://127.0.0.1:8090",
    "anthropic": "http://127.0.0.1:8091",
    "bedrock": "http://127.0.0.1:8092",
}
GUARDRAIL_URL_ENV: Final = "E2E_PERF_MOCK_GUARDRAIL_URL"
GUARDRAIL_URL_DEFAULT: Final = "http://127.0.0.1:8093"
MOCK_CREDENTIAL: Final = "perf-mock-not-checked"
INPUT_COST_PER_TOKEN: Final = 1e-6
OUTPUT_COST_PER_TOKEN: Final = 2e-6


def mock_url(provider: Provider) -> str:
    return os.environ.get(MOCK_URL_ENV[provider], MOCK_URL_DEFAULT[provider]).rstrip("/")


def guardrail_url() -> str:
    return os.environ.get(GUARDRAIL_URL_ENV, GUARDRAIL_URL_DEFAULT).rstrip("/")


class PerfTeamNewBody(TeamNewBody):
    """`/team/new` with the per-team settings a `Team` carries."""

    max_budget: float | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None


class GenericGuardrailParams(BaseModel):
    """`litellm_params` of a `generic_guardrail_api` registration (POST /guardrails). `default_on`
    applies it to every request on this gateway, which the perf stack dedicates to the run;
    `streaming_end_of_stream_only` makes the post hook one call per stream instead of one per
    sampled chunk, so a stream and a non-stream request both cost exactly two guardrail calls."""

    guardrail: Literal["generic_guardrail_api"] = "generic_guardrail_api"
    mode: Literal["pre_call", "post_call"]
    api_base: str
    default_on: bool = True
    streaming_end_of_stream_only: bool = True
    additional_provider_specific_params: dict[str, float]


class GuardrailSpec(BaseModel):
    guardrail_name: str
    litellm_params: GenericGuardrailParams


class GuardrailCreate(BaseModel):
    guardrail: GuardrailSpec


class GuardrailCreated(BaseModel):
    guardrail_id: str


@dataclass(frozen=True, slots=True)
class CaseSetup:
    marker: str
    team_ids: dict[str, str]
    model_names: dict[str, str]
    key_bearers: dict[str, str]
    token_sources: dict[str, TokenSource]
    user_ids: dict[str, str]
    model_ids: tuple[str, ...]
    guardrail_ids: tuple[str, ...]

    def authorization(self, caller: Caller) -> str | None:
        return f"Bearer {self.key_bearers[caller.name]}" if caller.kind == "key" else None

    def token_source(self, caller: Caller) -> TokenSource | None:
        return self.token_sources[caller.name] if caller.kind == "jwt" else None


def deployment_params(deployment: MockDeployment) -> LiteLLMParamsBody:
    """The litellm_params that point one deployment at its mock provider, with its knobs in the
    model id and a fixed custom price so spend is deterministic."""
    suffix: Final = deployment.model_suffix
    base: Final = mock_url(deployment.provider)
    if deployment.provider == "openai":
        return LiteLLMParamsBody(
            model=f"openai/{suffix}",
            api_base=f"{base}/v1",
            api_key=MOCK_CREDENTIAL,
            input_cost_per_token=INPUT_COST_PER_TOKEN,
            output_cost_per_token=OUTPUT_COST_PER_TOKEN,
        )
    if deployment.provider == "anthropic":
        return LiteLLMParamsBody(
            model=f"anthropic/{suffix}",
            api_base=base,
            api_key=MOCK_CREDENTIAL,
            input_cost_per_token=INPUT_COST_PER_TOKEN,
            output_cost_per_token=OUTPUT_COST_PER_TOKEN,
        )
    return LiteLLMParamsBody(
        model=f"bedrock/{deployment.bedrock_route}/anthropic.claude-{suffix}",
        aws_bedrock_runtime_endpoint=base,
        aws_region_name="us-east-1",
        aws_access_key_id=MOCK_CREDENTIAL,
        aws_secret_access_key=MOCK_CREDENTIAL,
        input_cost_per_token=INPUT_COST_PER_TOKEN,
        output_cost_per_token=OUTPUT_COST_PER_TOKEN,
    )


def _register_guardrail(proxy: ProxyClient, resources: ResourceManager, guardrail: Guardrail, name: str) -> str:
    params: Final = GenericGuardrailParams(
        mode=guardrail.mode,
        api_base=guardrail_url(),
        additional_provider_specific_params={"latency_s": guardrail.latency_s},
    )
    created: Final = unwrap(
        proxy.transport.post(
            "/guardrails",
            headers=proxy.transport.master,
            json=GuardrailCreate(guardrail=GuardrailSpec(guardrail_name=name, litellm_params=params)),
            response_type=GuardrailCreated,
        )
    )
    resources.defer(
        lambda: proxy.transport.delete(
            f"/guardrails/{created.guardrail_id}", headers=proxy.transport.master, json=NoBody(), response_type=NoBody
        )
    )
    return created.guardrail_id


def build_setup(
    case: PerformanceTestCase, *, proxy: ProxyClient, idp: Keycloak, resources: ResourceManager
) -> CaseSetup:
    marker: Final = unique_marker()
    team_ids: dict[str, str] = {}
    keycloak_groups: dict[str, str] = {}
    for team in case.teams:
        team_id = f"perf-{team.name}-{marker}"
        team_ids[team.name] = proxy.create_team(
            PerfTeamNewBody(
                team_alias=team_id,
                team_id=team_id,
                models=[f"{model}-{marker}" for model in team.models],
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
                    user_role=user.role,
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
                models=[f"{model}-{marker}" for model in key.models],
                team_id=team_ids[key.team] if key.team else None,
                user_id=user_ids[key.user] if key.user else None,
                rpm_limit=key.rpm_limit,
                max_budget=key.max_budget,
                key_alias=f"perf-{key.name}-{marker}",
            )
        )
        resources.defer(lambda generated=generated: proxy.delete_key(generated))
        key_bearers[key.name] = generated

    model_names: Final = {group.name: f"{group.name}-{marker}" for group in case.model_groups}
    model_ids: list[str] = []
    for group in case.model_groups:
        for deployment in group.deployments:
            model_id = proxy.create_model(model_names[group.name], deployment_params(deployment))
            resources.defer(lambda model_id=model_id: proxy.delete_model(model_id))
            model_ids.append(model_id)

    guardrail_ids: Final = tuple(
        _register_guardrail(proxy, resources, guardrail, f"{guardrail.name}-{marker}") for guardrail in case.guardrails
    )
    if guardrail_ids:
        settle_propagation(time.monotonic())
    return CaseSetup(
        marker=marker,
        team_ids=team_ids,
        model_names=model_names,
        key_bearers=key_bearers,
        token_sources=token_sources,
        user_ids=user_ids,
        model_ids=tuple(model_ids),
        guardrail_ids=guardrail_ids,
    )

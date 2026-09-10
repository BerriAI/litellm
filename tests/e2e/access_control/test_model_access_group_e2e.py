"""Live e2e: a model access group as the grant on a key and on a team.

Whoever holds the group can call every deployment in it and nothing else, whether
the request names a deployment exactly, names a model that a wildcard deployment
in the group covers, or spells that model with its provider prefix. The bare-name
spelling is the LIT-5813 regression: the group-membership lookup skipped the
provider-prefix retry every other model-resolution path performs, so a group
holding `openai/gpt-5.4*` denied `gpt-5.4-nano` while allowing `openai/gpt-5.4-nano`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Final

import pytest

from access_control_client import (
    AccessControlClient,
    MODEL_ACCESS_DENIED_MARKER,
    TEAM_MODEL_ACCESS_DENIED_MARKER,
)
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import (
    ChatResponse,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ModelInfoBody,
    ModelNewBody,
)

pytestmark = pytest.mark.e2e

WILDCARD_PATTERN: Final = "openai/gpt-5.4*"
WILDCARD_BARE_MODEL: Final = "gpt-5.4-nano"
WILDCARD_PREFIXED_MODEL: Final = "openai/gpt-5.4-nano"
GROUP_BACKEND: Final = "openai/gpt-5.4-nano"
UNCOVERED_OPENAI_MODEL: Final = "gpt-5.2"

TEAM_WILDCARD_PATTERN: Final = "openai/gpt-5.6*"
TEAM_WILDCARD_BARE_MODEL: Final = "gpt-5.6-luna"

MAX_COMPLETION_TOKENS: Final = 16
PROMPT: Final = "Reply with exactly: OK"


@dataclass(frozen=True, slots=True)
class GroupedDeployments:
    """A wildcard deployment and an exactly-named one inside `access_group`, plus a
    deployment left out of it."""

    access_group: str
    member_model: str
    outsider_model: str


@dataclass(frozen=True, slots=True)
class TeamGrant:
    """A team whose whole allow-list is `access_group`, holding one team-scoped
    wildcard deployment, and a key that belongs to it."""

    access_group: str
    team_id: str
    key: str


ModelSelector = Callable[[GroupedDeployments], str]

ALLOWED: Final[tuple[tuple[str, ModelSelector], ...]] = (
    ("bare name the group's wildcard covers", lambda grouped: WILDCARD_BARE_MODEL),
    ("provider-prefixed name the group's wildcard covers", lambda grouped: WILDCARD_PREFIXED_MODEL),
    ("exactly-named deployment in the group", lambda grouped: grouped.member_model),
)

DENIED: Final[tuple[tuple[str, ModelSelector], ...]] = (
    ("deployment outside the group", lambda grouped: grouped.outsider_model),
    ("provider model outside the group's wildcard", lambda grouped: UNCOVERED_OPENAI_MODEL),
    ("name no provider claims", lambda grouped: f"e2e-ag-unknown-{unique_marker()}"),
)


def _provider_key(env_var: str) -> str:
    return os.environ.get(env_var) or f"os.environ/{env_var}"


def _grouped_model(model_name: str, backend: str, access_groups: list[str] | None) -> ModelNewBody:
    return ModelNewBody(
        model_name=model_name,
        litellm_params=LiteLLMParamsBody(model=backend, api_key=_provider_key("OPENAI_API_KEY")),
        model_info=ModelInfoBody(access_groups=access_groups),
    )


def _await_group_members(client: AccessControlClient, access_group: str, expected: frozenset[str]) -> None:
    """The grant under test is the group's membership, so prove the proxy recorded it
    before asserting on what the group lets through."""
    deadline = time.monotonic() + client.proxy.poll_timeout
    listed: list[str] = []
    while time.monotonic() < deadline:
        info = client.access_group_info(access_group)
        listed = info.model_names if info is not None else []
        if expected.issubset(listed):
            return
        time.sleep(client.proxy.poll_interval)
    pytest.fail(
        f"/access_group/{access_group}/info never listed {sorted(expected)} as members; last read {listed}"
    )


def _await_team_allowlist(client: AccessControlClient, grant_key: str, access_group: str) -> None:
    """Registering a team-scoped deployment appends its public name to the team's
    allow-list, and a wildcard sitting there directly would grant the model under test
    on its own. Poll a denial until the message enumerates the allow-list the test
    means to exercise: the group, and nothing else."""
    allowlist: Final = f"models=['{access_group}']"
    deadline = time.monotonic() + client.proxy.poll_timeout
    body = ""
    while time.monotonic() < deadline:
        body = client.chat_status(
            grant_key, UNCOVERED_OPENAI_MODEL, f"{PROMPT} {unique_marker()}", MAX_COMPLETION_TOKENS
        ).body
        if allowlist in body:
            return
        time.sleep(client.proxy.poll_interval)
    pytest.fail(f"the team's allow-list never settled to {allowlist}; last denial read {body[:300]}")


@pytest.fixture(scope="module")
def grouped(client: AccessControlClient) -> Iterator[GroupedDeployments]:
    marker: Final = unique_marker()
    deployments: Final = GroupedDeployments(
        access_group=f"e2e-ag-{marker}",
        member_model=f"e2e-ag-member-{marker}",
        outsider_model=f"e2e-ag-outsider-{marker}",
    )
    registrations: Final = (
        _grouped_model(WILDCARD_PATTERN, WILDCARD_PATTERN, [deployments.access_group]),
        _grouped_model(deployments.member_model, GROUP_BACKEND, [deployments.access_group]),
        _grouped_model(deployments.outsider_model, GROUP_BACKEND, None),
    )
    created: Final = tuple(client.proxy.register_model(body) for body in registrations)
    try:
        _await_group_members(
            client,
            deployments.access_group,
            frozenset({WILDCARD_PATTERN, deployments.member_model}),
        )
        yield deployments
    finally:
        for model_id in created:
            client.proxy.delete_model(model_id)


@pytest.fixture(scope="module")
def team_grant(client: AccessControlClient) -> Iterator[TeamGrant]:
    marker: Final = unique_marker()
    access_group: Final = f"e2e-agt-{marker}"
    team_alias: Final = f"e2e-ag-team-{marker}"
    team_id: Final = client.create_team(team_alias, [access_group])
    key: Final = client.proxy.generate_key(KeyGenerateBody(models=[], team_id=team_id))
    model_id: Final = client.proxy.register_model(
        ModelNewBody(
            model_name=TEAM_WILDCARD_PATTERN,
            litellm_params=LiteLLMParamsBody(
                model=TEAM_WILDCARD_PATTERN, api_key=_provider_key("OPENAI_API_KEY")
            ),
            model_info=ModelInfoBody(team_id=team_id, access_groups=[access_group]),
        ),
        listed_for=key,
    )
    client.set_team_models(team_id, team_alias, [access_group])
    try:
        _await_team_allowlist(client, key, access_group)
        yield TeamGrant(access_group=access_group, team_id=team_id, key=key)
    finally:
        client.proxy.delete_model(model_id)
        client.proxy.delete_key(key)
        client.delete_team(team_id)


class TestKeyScopedToAccessGroup:
    @pytest.mark.covers(
        "other.auth.model_access_group.wildcard_bare_name_allowed",
        "other.auth.model_access_group.member_allowed",
    )
    @pytest.mark.parametrize(("case", "select_model"), ALLOWED)
    def test_group_grants_every_deployment_in_it(
        self,
        case: str,
        select_model: ModelSelector,
        client: AccessControlClient,
        resources: ResourceManager,
        grouped: GroupedDeployments,
    ) -> None:
        key = resources.key(models=[grouped.access_group])
        model = select_model(grouped)

        result = client.chat_status(
            key, model, f"{PROMPT} {unique_marker()}", MAX_COMPLETION_TOKENS
        )

        assert result.status_code == 200, (
            f"a key holding access group {grouped.access_group!r} must be able to call "
            f"{model!r} ({case}), got {result.status_code}: {result.body[:300]}"
        )
        assert ChatResponse.model_validate_json(result.body).choices, (
            f"200 must carry a real completion, not an error envelope: {result.body[:300]}"
        )

    @pytest.mark.covers("other.auth.model_access_group.non_member_denied")
    @pytest.mark.parametrize(("case", "select_model"), DENIED)
    def test_group_grants_nothing_outside_it(
        self,
        case: str,
        select_model: ModelSelector,
        client: AccessControlClient,
        resources: ResourceManager,
        grouped: GroupedDeployments,
    ) -> None:
        key = resources.key(models=[grouped.access_group])
        model = select_model(grouped)

        result = client.chat_status(
            key, model, f"{PROMPT} {unique_marker()}", MAX_COMPLETION_TOKENS
        )

        assert result.status_code == 403, (
            f"a key holding only access group {grouped.access_group!r} must be denied 403 on "
            f"{model!r} ({case}), got {result.status_code}: {result.body[:300]}"
        )
        assert MODEL_ACCESS_DENIED_MARKER in result.body, (
            f"403 body must be a key model-access denial, got: {result.body[:300]}"
        )


class TestTeamScopedToAccessGroup:
    @pytest.mark.covers("other.auth.model_access_group.team_wildcard_bare_name_allowed")
    def test_group_grants_the_teams_own_wildcard(
        self, client: AccessControlClient, team_grant: TeamGrant
    ) -> None:
        result = client.chat_status(
            team_grant.key,
            TEAM_WILDCARD_BARE_MODEL,
            f"{PROMPT} {unique_marker()}",
            MAX_COMPLETION_TOKENS,
        )

        assert result.status_code == 200, (
            f"a team whose allow-list is access group {team_grant.access_group!r} must be able to "
            f"call {TEAM_WILDCARD_BARE_MODEL!r} through its team-scoped {TEAM_WILDCARD_PATTERN!r} "
            f"deployment, got {result.status_code}: {result.body[:300]}"
        )
        assert ChatResponse.model_validate_json(result.body).choices, (
            f"200 must carry a real completion, not an error envelope: {result.body[:300]}"
        )

    @pytest.mark.covers("other.auth.model_access_group.team_non_member_denied")
    def test_group_grants_the_team_nothing_outside_it(
        self, client: AccessControlClient, team_grant: TeamGrant
    ) -> None:
        model = f"e2e-ag-unknown-{unique_marker()}"

        result = client.chat_status(
            team_grant.key, model, f"{PROMPT} {unique_marker()}", MAX_COMPLETION_TOKENS
        )

        assert result.status_code == 403, (
            f"a team holding only access group {team_grant.access_group!r} must be denied 403 on "
            f"{model!r}, got {result.status_code}: {result.body[:300]}"
        )
        assert TEAM_MODEL_ACCESS_DENIED_MARKER in result.body, (
            f"403 body must be a team model-access denial, got: {result.body[:300]}"
        )

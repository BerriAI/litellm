import json
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.shares_worker_state

KEY_ID: Final = "integration-jwt-signing-key"
MEMBER_PERMISSIONS: Final = ["/key/generate", "/key/info"]

Signer = Callable[[Mapping[str, JsonValue]], str]


@contextmanager
def _upserting_jwt_proxy(
    gateway: Gateway,
    directory: Path,
    team_claim: Mapping[str, str],
    default_team_params: Mapping[str, JsonValue],
) -> Iterator[tuple[Gateway, Signer]]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key())

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=json.dumps({"keys": [{**json.loads(public_jwk), "kid": KEY_ID}]}).encode())

    def sign(claims: Mapping[str, JsonValue]) -> str:
        return jwt.encode(
            {**claims, "iat": int(time.time()), "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": KEY_ID},
        )

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {
        **config["general_settings"],
        "enable_jwt_auth": True,
        "litellm_jwtauth": {"user_id_jwt_field": "sub", **team_claim, "team_id_upsert": True},
    }
    config["litellm_settings"] = {**config["litellm_settings"], "default_team_params": dict(default_team_params)}
    path: Final = directory / "jwt_team_upsert.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        wire_server(respond) as jwks,
        owned_proxy(gateway, directory, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path) as candidate,
    ):
        yield candidate, sign


def _chat(candidate: Gateway, token: str, model: str, team_header: bool, team_id: str) -> int:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "team upsert control"}]},
        key=token,
        headers={"x-litellm-team-id": team_id} if team_header else None,
    ).status_code


def _team_rows(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT models, team_member_permissions FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))


def _delete_team_if_present(scenario: Scenario, team_id: str) -> None:
    if _team_rows(team_id):
        scenario.delete_team(team_id)


@pytest.mark.parametrize(
    ("team_claim", "claim_name", "team_header"),
    [({"team_id_jwt_field": "dept"}, "dept", False), ({"team_ids_jwt_field": "groups"}, "groups", True)],
    ids=["team-id-claim", "header-team-from-teams-claim"],
)
def test_jwt_upserted_team_starts_with_default_team_models_and_other_teams_keep_theirs(
    gateway: Gateway, tmp_path: Path, team_claim: Mapping[str, str], claim_name: str, team_header: bool
) -> None:
    with gateway.scenario() as scenario:
        approved: Final = scenario.model()
        unapproved: Final = scenario.model()
        existing_team: Final = scenario.team(models=[unapproved])
        upserted_team: Final = f"integration-dept-{uuid.uuid4().hex}"
        admin_team: Final = f"integration-admin-team-{uuid.uuid4().hex}"
        subject: Final = scenario.user(user_role="internal_user")
        defaults: Final = {"models": [approved], "team_member_permissions": MEMBER_PERMISSIONS}

        def claims(team_id: str) -> Mapping[str, JsonValue]:
            return {"sub": subject, claim_name: [team_id] if team_header else team_id}

        with _upserting_jwt_proxy(gateway, tmp_path, team_claim, defaults) as (candidate, sign):
            scenario.cleanups.callback(_delete_team_if_present, scenario, upserted_team)
            scenario.cleanups.callback(_delete_team_if_present, scenario, admin_team)
            upsert_token: Final = sign(claims(upserted_team))

            approved_status: Final = _chat(candidate, upsert_token, approved, team_header, upserted_team)
            unapproved_status: Final = _chat(candidate, upsert_token, unapproved, team_header, upserted_team)
            admin_created: Final = candidate.request("POST", "/team/new", {"team_id": admin_team})
            assert admin_created.status_code == 200, admin_created.text
            existing_status: Final = _chat(
                candidate, sign(claims(existing_team)), unapproved, team_header, existing_team
            )

            assert (approved_status, _team_rows(upserted_team), unapproved_status) == (
                200,
                [{"models": [approved], "team_member_permissions": MEMBER_PERMISSIONS}],
                403,
            )
            assert _team_rows(admin_team) == [{"models": [], "team_member_permissions": MEMBER_PERMISSIONS}]
            assert (existing_status, [row["models"] for row in _team_rows(existing_team)]) == (200, [[unapproved]])


@pytest.mark.parametrize(
    ("organization_allows_default", "expected_status"),
    [(False, 404), (True, 200)],
    ids=["organization-excludes-default-model", "organization-allows-default-model"],
)
def test_jwt_team_upsert_creates_the_team_only_when_the_default_organization_allows_the_default_models(
    gateway: Gateway, tmp_path: Path, organization_allows_default: bool, expected_status: int
) -> None:
    with gateway.scenario() as scenario:
        approved: Final = scenario.model()
        org_allowed: Final = scenario.model()
        organization: Final = scenario.organization(
            models=[org_allowed, approved] if organization_allows_default else [org_allowed]
        )
        upserted_team: Final = f"integration-dept-{uuid.uuid4().hex}"
        subject: Final = scenario.user(user_role="internal_user")
        defaults: Final = {"models": [approved], "organization_id": organization}

        with _upserting_jwt_proxy(gateway, tmp_path, {"team_id_jwt_field": "dept"}, defaults) as (candidate, sign):
            scenario.cleanups.callback(_delete_team_if_present, scenario, upserted_team)

            status: Final = _chat(
                candidate, sign({"sub": subject, "dept": upserted_team}), approved, False, upserted_team
            )

            teams: Final = read_rows(
                'SELECT models, organization_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (upserted_team,)
            )
            expected_teams: Final = (
                [{"models": [approved], "organization_id": organization}] if organization_allows_default else []
            )
            assert (status, teams) == (expected_status, expected_teams)

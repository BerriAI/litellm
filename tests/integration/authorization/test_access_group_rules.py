import json
import time
import uuid
from pathlib import Path
from typing import Final

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.proxy.mcp_server.idp_token_exchange import REJECTED_SUBJECT_TOKEN
from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import graceful_stop_seconds, owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.shares_worker_state


@pytest.mark.parametrize(
    "enabled,mapping,claimed", ((True, False, True), (False, False, True), (True, True, False), (True, True, True))
)
def test_rules_add_models_to_jwt_and_scoped_virtual_key_without_opening_other_models(
    gateway: Gateway, tmp_path: Path, enabled: bool, mapping: bool, claimed: bool
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    key_id: Final = "rule-signing-key"

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=json.dumps({"keys": [{**public_key, "kid": key_id}]}).encode())

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        explicit: Final = scenario.model()
        group_model: Final = scenario.model()
        outside: Final = scenario.model()
        group_name: Final = f"integration-rule-{uuid.uuid4().hex}"
        created: Final = gateway.request(
            "POST", "/v1/access_group", {"access_group_name": group_name, "access_model_names": [group_model]}
        )
        assert created.status_code == 201, created.text
        group_id: Final = string_value(created.json()["access_group_id"])
        scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/access_group/{group_id}")
        virtual_user: Final = scenario.user()
        virtual_key: Final = scenario.key(models=[explicit], user_id=virtual_user)
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"] = {
            **config["general_settings"],
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "user_id_jwt_field": "sub",
                "team_ids_jwt_field": "groups",
                "user_id_upsert": True,
                "enforce_team_based_model_access": True,
                **(
                    {"virtual_key_claim_field": "sub", "unregistered_jwt_client_behavior": "auto_register"}
                    if mapping
                    else {}
                ),
            },
            **(
                {
                    "access_group_rules": [
                        {"attribute": "jwt_claims.groups", "contains": "science", "access_group": group_name},
                        {"attribute": "user_id", "equals": virtual_user, "access_group": group_name},
                    ]
                }
                if enabled
                else {}
            ),
        }
        path: Final = tmp_path / "access_group_rules.yaml"
        path.write_text(yaml.safe_dump(config))
        subject: Final = f"integration-rule-user-{uuid.uuid4().hex}"
        gateway.post("/user/new", {"user_id": subject, "user_role": "internal_user"})
        scenario.cleanups.callback(scenario.delete_user, subject)
        team_id: Final = (
            scenario.team(models=[explicit], members_with_roles=[{"role": "user", "user_id": subject}])
            if claimed
            else None
        )
        team_claims: Final = [team_id] if team_id is not None else []
        token: Final = jwt.encode(
            {"sub": subject, "groups": ["science", *team_claims], "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": key_id},
        )
        unmatched: Final = jwt.encode(
            {"sub": subject, "groups": ["other", *team_claims], "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": key_id},
        )
        with owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path, workers=2) as candidate:
            fallback_response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": group_model,
                    "fallbacks": [outside],
                    "messages": [{"role": "user", "content": "ungranted fallback control"}],
                },
                key=token,
            )
            assert fallback_response.status_code == 403, fallback_response.text
            for key, model, allowed in (
                (token, group_model, enabled and claimed),
                *(((token, group_model, enabled and claimed),) if mapping else ()),
                (token, outside, False),
                (unmatched, group_model, False),
                (virtual_key, explicit, True),
                (virtual_key, group_model, enabled),
                (virtual_key, outside, False),
            ):
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "rule authorization control"}]},
                    key=key,
                )
                assert response.status_code == (200 if allowed else 403), response.text
                if allowed:
                    assert (
                        response.json()["choices"][0]["message"]["content"]
                        == "Hello! This is a mock response from the fake OpenAI endpoint."
                    ), response.text

            if mapping and claimed:
                rows: Final = read_rows(
                    'SELECT models, access_group_ids FROM "LiteLLM_VerificationToken" '
                    'WHERE user_id=%s AND team_id=%s',
                    (subject, team_id),
                )
                assert len(rows) == 1, rows
                assert group_model not in rows[0]["models"], rows
                assert not rows[0]["access_group_ids"], rows


@pytest.mark.parametrize("consumer", ("auto_register", "token_exchange"))
@pytest.mark.parametrize("matching", (True, False))
def test_teamless_jwt_cannot_provision_credentials_without_native_team_access(
    gateway: Gateway, tmp_path: Path, consumer: str, matching: bool
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    key_id: Final = "credential-rule-signing-key"

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=json.dumps({"keys": [{**public_key, "kid": key_id}]}).encode())

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = f"integration-credential-user-{uuid.uuid4().hex}"
        gateway.post("/user/new", {"user_id": subject, "user_role": "internal_user"})
        scenario.cleanups.callback(scenario.delete_user, subject)
        group_name: Final = f"credential-rule-{uuid.uuid4().hex}"
        created_group: Final = gateway.request(
            "POST", "/v1/access_group", {"access_group_name": group_name, "access_model_names": [model]}
        )
        assert created_group.status_code == 201, created_group.text
        group_id: Final = string_value(created_group.json()["access_group_id"])
        scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/access_group/{group_id}")
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"] = {
            **config["general_settings"],
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "user_id_jwt_field": "sub",
                "enforce_team_based_model_access": True,
                **(
                    {"virtual_key_claim_field": "sub", "unregistered_jwt_client_behavior": "auto_register"}
                    if consumer == "auto_register"
                    else {}
                ),
            },
            "access_group_rules": [
                {"attribute": "jwt_claims.groups", "contains": "science", "access_group": group_name}
            ],
        }
        path: Final = tmp_path / "credential_access_group_rules.yaml"
        path.write_text(yaml.safe_dump(config))
        token: Final = jwt.encode(
            {"sub": subject, "groups": ["science" if matching else "other"], "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": key_id},
        )
        with owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path, workers=2) as candidate:
            if consumer == "auto_register":
                for _ in range(2):
                    response: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": "credential rejection control"}]},
                        key=token,
                    )
                    assert response.status_code == 403, response.text
                    assert "No teams found in token" in response.text, response.text
            else:
                registered: Final = candidate.client.post(
                    "/register", json={"redirect_uris": ["http://localhost/callback"], "client_name": "rule-control"}
                )
                assert registered.status_code in (200, 201), registered.text
                exchanged: Final = candidate.client.post(
                    "/token",
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                        "client_id": registered.json()["client_id"],
                        "subject_token": token,
                        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    },
                )
                assert exchanged.status_code == 400, exchanged.text
                assert exchanged.json() == {"error": "invalid_request", "error_description": REJECTED_SUBJECT_TOKEN}, (
                    exchanged.text
                )


def test_restricted_callers_reach_a_deployment_by_its_litellm_model_only_when_every_namesake_is_granted(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        granted_group: Final = f"granted-{suffix}"
        other_group: Final = f"other-{suffix}"
        granted_raw: Final = f"openai/granted-{suffix}"
        shared_raw: Final = f"openai/shared-{suffix}"
        sibling_raw: Final = f"openai/sibling-{suffix}"
        granted_name: Final = scenario.model(model=granted_raw, model_info={"access_groups": [granted_group]})
        shared_granted_name: Final = scenario.model(model=shared_raw, model_info={"access_groups": [granted_group]})
        scenario.model(model=shared_raw, model_info={"access_groups": [other_group]})
        split_name: Final = f"integration-split-{suffix}"
        scenario.model(name=split_name, model=f"openai/split-{suffix}", model_info={"access_groups": [granted_group]})
        scenario.model(name=split_name, model=sibling_raw, model_info={"access_groups": [other_group]})
        group_key: Final = scenario.key(models=[granted_group])
        explicit_key: Final = scenario.key(models=[granted_name])
        shared_explicit_key: Final = scenario.key(models=[shared_granted_name])
        raw_key: Final = scenario.key(models=[granted_raw])
        team_key: Final = scenario.key(team_id=scenario.team(models=[granted_name]))
        rows: Final = (
            ("access group key, granted raw name", group_key, granted_raw, 200),
            ("explicit key, granted raw name", explicit_key, granted_raw, 200),
            ("access group key, raw name shared with an ungranted deployment", group_key, shared_raw, 403),
            ("explicit key, raw name shared with an ungranted deployment", shared_explicit_key, shared_raw, 403),
            ("access group key, raw name of an ungranted sibling in a granted group", group_key, sibling_raw, 403),
            ("team restricted key, granted raw name", team_key, granted_raw, 200),
            ("key listing the raw name itself", raw_key, granted_raw, 200),
        )
        observed: Final = tuple(
            (
                label,
                gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "raw deployment name control"}]},
                    key=key,
                ),
            )
            for label, key, model, _ in rows
        )
        assert tuple((label, response.status_code) for label, response in observed) == tuple(
            (label, status) for label, _, _, status in rows
        ), [(label, response.status_code, response.text) for label, response in observed]
        assert all(
            "model_access_denied" in response.text
            for (_, response), (*_, status) in zip(observed, rows)
            if status == 403
        ), [response.text for _, response in observed]


def _chat(gateway: Gateway, model: str, key: str, text: str = "raw deployment name control") -> httpx.Response:
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}]}, key=key
    )


@pytest.mark.parametrize("first_call_by", ("model_name", "litellm_model"))
def test_key_model_rpm_limit_on_the_public_name_counts_requests_by_the_litellm_model(
    gateway: Gateway, first_call_by: str
) -> None:
    with gateway.scenario() as scenario:
        litellm_model: Final = f"openai/raw-{uuid.uuid4().hex}"
        public_name: Final = scenario.model(model=litellm_model)
        key: Final = scenario.key(models=[public_name], model_rpm_limit={public_name: 1})
        first_model, second_model = (
            (public_name, litellm_model) if first_call_by == "model_name" else (litellm_model, public_name)
        )
        served: Final = _chat(gateway, first_model, key)
        assert served.status_code == 200, served.text
        refused: Final = _chat(gateway, second_model, key)
        assert refused.status_code == 429, refused.text
        assert "model_per_key" in refused.text, refused.text


def test_model_level_during_call_guardrail_runs_for_a_request_by_the_deployment_litellm_model(gateway: Gateway) -> None:
    forbidden: Final = f"forbidden{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": f"raw-name-filter-{uuid.uuid4().hex}",
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "during_call",
                    "default_on": False,
                    "blocked_words": [{"keyword": forbidden, "action": "BLOCK"}],
                },
            }
        },
    )
    try:
        with gateway.scenario() as scenario:
            litellm_model: Final = f"openai/raw-{uuid.uuid4().hex}"
            public_name: Final = scenario.model(
                model=litellm_model, guardrails=[string_value(created["guardrail_name"])]
            )
            key: Final = scenario.key(models=[public_name])
            outcomes: Final = tuple(
                eventually(
                    lambda model=model: _chat(gateway, model, key, f"say {forbidden}"),
                    lambda response: response.status_code != 200,
                    seconds=30,
                    return_last_on_timeout=True,
                )
                for model in (public_name, litellm_model)
            )
            assert [(response.status_code, forbidden in response.text) for response in outcomes] == [
                (400, True),
                (400, True),
            ], [response.text for response in outcomes]
    finally:
        deleted: Final = gateway.request("DELETE", f"/guardrails/{string_value(created['guardrail_id'])}")
        assert deleted.status_code == 200, deleted.text


def _completion(request: Request) -> Reply:
    if (request.method, request.target) == ("GET", "/v1/models"):
        return Reply(body=b'{"object":"list","data":[]}')
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "served"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 1},
            }
        ).encode()
    )


@pytest.mark.parametrize("caller", ("teamless", "team-1"))
def test_another_teams_ungranted_deployment_neither_blocks_nor_serves_a_request_by_the_litellm_model(
    gateway: Gateway, caller: str
) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        group: Final = f"granted-{suffix}"
        litellm_model: Final = f"openai/two-teams-{suffix}"
        scenario.model(
            model=litellm_model,
            api_base=f"{wire.url}/v1",
            api_key="granted-deployment",
            model_info={"access_groups": [group]},
        )
        scenario.model(
            model=litellm_model,
            api_base=f"{wire.url}/v1",
            api_key="other-team-deployment",
            model_info={"team_id": scenario.team(), "team_public_model_name": f"other-team-{suffix}"},
        )
        key: Final = (
            scenario.key(models=[group])
            if caller == "teamless"
            else scenario.key(team_id=scenario.team(models=[group]))
        )
        statuses: Final = tuple(_chat(gateway, litellm_model, key).status_code for _ in range(6))
        served_by: Final = {
            entry.headers.get("authorization") for entry in wire.drain() if entry.target == "/v1/chat/completions"
        }
        assert statuses == (200,) * 6, statuses
        assert served_by == {"Bearer granted-deployment"}, served_by


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_wildcard_route_blocks_a_litellm_model_grant_only_for_the_names_it_matches(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        group: Final = f"granted-{suffix}"
        unmatched_raw: Final = f"openai/unmatched-{suffix}"
        matched_raw: Final = f"openai/matched-{suffix}-raw"
        for raw in (unmatched_raw, matched_raw):
            scenario.model(
                model=raw,
                api_base=f"{wire.url}/v1",
                api_key="granted-deployment",
                model_info={"access_groups": [group]},
            )
        key: Final = scenario.key(models=[group])
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            *config["model_list"],
            *(
                {
                    "model_name": pattern,
                    "litellm_params": {"model": "openai/*", "api_base": f"{wire.url}/v1", "api_key": "wildcard"},
                }
                for pattern in (f"unrelated-{suffix}/*", f"openai/matched-{suffix}-*")
            ),
        ]
        path: Final = tmp_path / "wildcard_routes.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path, workers=2) as candidate:
            wire.drain()
            statuses: Final = tuple(
                (raw, _chat(candidate, raw, key).status_code) for raw in (unmatched_raw, matched_raw)
            )
            granted_served_by: Final = [
                entry.headers.get("authorization") for entry in wire.drain() if entry.target == "/v1/chat/completions"
            ]
            admin: Final = _chat(candidate, matched_raw, candidate.key)
            admin_served_by: Final = [
                entry.headers.get("authorization") for entry in wire.drain() if entry.target == "/v1/chat/completions"
            ]
        assert statuses == ((unmatched_raw, 200), (matched_raw, 403)), statuses
        assert granted_served_by == ["Bearer granted-deployment"], granted_served_by
        assert admin.status_code == 200, admin.text
        assert admin_served_by == ["Bearer wildcard"], admin_served_by

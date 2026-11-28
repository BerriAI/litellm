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
from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import graceful_stop_seconds, owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

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


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"uncached {uuid.uuid4().hex}"}]},
        key=key,
    )


def _error_type(response: httpx.Response) -> str | None:
    return response.json()["error"]["type"] if response.status_code != 200 else None


def _outcome(gateway: Gateway, wire: Wire, model: str, key: str, calls: int) -> tuple[set[int], set[str | None]]:
    wire.drain()
    statuses: Final = {_chat(gateway, model, key).status_code for _ in range(calls)}
    return statuses, {
        entry.headers.get("authorization") for entry in wire.drain() if entry.target == "/v1/chat/completions"
    }


def test_a_deployment_named_by_its_litellm_model_is_served_only_by_the_deployments_the_caller_is_granted(
    gateway: Gateway,
) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        group_a: Final = f"group-a-{suffix}"
        group_b: Final = f"group-b-{suffix}"
        shared: Final = f"openai/shared-{suffix}"
        lone: Final = f"openai/lone-{suffix}"
        upstream: Final = f"{wire.url}/v1"
        name_a: Final = scenario.model(
            model=shared, api_base=upstream, api_key="group-a", model_info={"access_groups": [group_a]}
        )
        name_b: Final = scenario.model(
            model=shared, api_base=upstream, api_key="group-b", model_info={"access_groups": [group_b]}
        )
        scenario.model(model=lone, api_base=upstream, api_key="lone", model_info={"access_groups": [group_a]})
        group_key: Final = scenario.key(models=[group_a])
        name_key: Final = scenario.key(models=[name_a])
        ungranted_key: Final = scenario.key(models=[f"group-c-{suffix}"])
        literal_key: Final = scenario.key(models=[shared])
        team_key: Final = scenario.key(team_id=scenario.team(models=[group_a]))
        other_team_key: Final = scenario.key(models=[group_a], team_id=scenario.team(models=[group_b]))
        rows: Final = (
            ("group key, litellm model of its one deployment", group_key, lone, 1, {200}, {"Bearer lone"}),
            ("group key, litellm model shared with group b", group_key, shared, 12, {200}, {"Bearer group-a"}),
            ("public name key, litellm model shared with group b", name_key, shared, 12, {200}, {"Bearer group-a"}),
            ("team restricted key, litellm model shared with group b", team_key, shared, 12, {200}, {"Bearer group-a"}),
            ("key granted neither deployment", ungranted_key, shared, 1, {403}, set()),
            ("group a key in a group b team", other_team_key, shared, 1, {403}, set()),
            ("key listing the litellm model", literal_key, shared, 24, {200}, {"Bearer group-a", "Bearer group-b"}),
            ("group key, granted public name", group_key, name_a, 1, {200}, {"Bearer group-a"}),
            ("group key, ungranted public name", group_key, name_b, 1, {403}, set()),
        )
        observed: Final = tuple(
            (label, *_outcome(gateway, wire, model, key, calls)) for label, key, model, calls, _, _ in rows
        )
        assert observed == tuple((label, statuses, served_by) for label, _, _, _, statuses, served_by in rows)
        assert {_error_type(_chat(gateway, shared, key)) for key in (ungranted_key, other_team_key)} == {
            "key_model_access_denied"
        }


def _owned_config(tmp_path: Path, name: str, **sections: object) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / name
    path.write_text(yaml.safe_dump({**config, **sections}))
    return path


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_wildcard_deployment_outside_the_callers_groups_never_serves_a_litellm_model_it_matches(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        group: Final = f"group-a-{suffix}"
        unmatched: Final = f"openai/unmatched-{suffix}"
        matched: Final = f"openai/matched-{suffix}-model"
        upstream: Final = f"{wire.url}/v1"
        for litellm_model in (unmatched, matched):
            scenario.model(
                model=litellm_model, api_base=upstream, api_key="group-a", model_info={"access_groups": [group]}
            )
        key: Final = scenario.key(models=[group])
        base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        path: Final = _owned_config(
            tmp_path,
            "wildcard_route.yaml",
            model_list=[
                *base["model_list"],
                {
                    "model_name": f"openai/matched-{suffix}-*",
                    "litellm_params": {"model": "openai/*", "api_base": upstream, "api_key": "wildcard"},
                },
            ],
        )
        with owned_proxy(gateway, tmp_path, {}, config=path, workers=2) as candidate:
            observed: Final = tuple(
                _outcome(candidate, wire, model, caller, 4)
                for model, caller in ((unmatched, key), (matched, key), (matched, candidate.key))
            )
        assert observed == (({200}, {"Bearer group-a"}), ({403}, set()), ({200}, {"Bearer wildcard"}))


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_user_config_router_never_serves_a_litellm_model_the_caller_holds_only_through_a_deployment(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        group: Final = f"group-a-{suffix}"
        litellm_model: Final = f"openai/user-config-{suffix}"
        upstream: Final = f"{wire.url}/v1"
        scenario.model(model=litellm_model, api_base=upstream, api_key="group-a", model_info={"access_groups": [group]})
        key: Final = scenario.key(models=[group])
        base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        path: Final = _owned_config(
            tmp_path,
            "client_side_credentials.yaml",
            general_settings={**base["general_settings"], "allow_client_side_credentials": True},
        )
        user_config: Final = {
            "model_list": [
                {
                    "model_name": litellm_model,
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": upstream, "api_key": "user-config"},
                }
            ]
        }
        with owned_proxy(gateway, tmp_path, {}, config=path, workers=2) as candidate:
            wire.drain()
            responses: Final = tuple(
                candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": litellm_model,
                        "messages": [{"role": "user", "content": f"uncached {uuid.uuid4().hex}"}],
                        "user_config": user_config,
                    },
                    key=caller,
                )
                for caller in (key, candidate.key)
            )
            served_by: Final = [
                entry.headers.get("authorization") for entry in wire.drain() if entry.target == "/v1/chat/completions"
            ]
        assert [(response.status_code, _error_type(response)) for response in responses] == [
            (403, "key_model_access_denied"),
            (200, None),
        ], [response.text for response in responses]
        assert served_by == ["Bearer user-config"]


def test_a_failed_pick_never_hands_an_inaccessible_litellm_model_to_the_provider_under_its_own_name(
    gateway: Gateway,
) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        group_a: Final = f"group-a-{suffix}"
        group_b: Final = f"group-b-{suffix}"
        litellm_model: Final = f"openai/container-{suffix}"
        for group in (group_a, group_b):
            scenario.model(
                model=litellm_model,
                api_base=f"{wire.url}/v1",
                api_key=group,
                model_info={"access_groups": [group]},
            )
        key: Final = scenario.key(models=[group_a], team_id=scenario.team(models=[group_b]))
        wire.drain()
        refused: Final = gateway.request(
            "POST", "/v1/containers", {"name": f"container-{suffix}", "model": litellm_model}, key=key
        )
        assert (refused.status_code, _error_type(refused)) == (403, "key_model_access_denied"), refused.text
        assert wire.drain() == ()

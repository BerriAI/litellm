import uuid
from pathlib import Path
from typing import Final

import httpx
import yaml

from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import wire_server
from integration.authorization._guardrail_opt_out import (
    denying_guardrail,
    guardrail_config,
    non_admin_caller,
    stored_metadata,
)

_KEY_ROUTES: Final = ["/key/generate", "/key/update", "/key/regenerate", "/v1/chat/completions"]


def test_non_admin_cannot_opt_key_out_of_default_on_guardrail(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(denying_guardrail) as policy:
        config: Final = guardrail_config(policy.url, tmp_path / "default_on.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            member: Final = scenario.user(user_role="internal_user")
            team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
            caller: Final = scenario.key(user_id=member, models=[model], allowed_routes=_KEY_ROUTES)
            own: Final = scenario.key(team_id=team, models=[model])

            plain: Final = candidate.request("POST", "/key/generate", {"team_id": team, "models": [model]}, key=caller)
            assert plain.status_code == 200, plain.text
            scenario.cleanups.callback(scenario.delete_key, string_value(plain.json()["key"]))

            generated: Final = candidate.request(
                "POST",
                "/key/generate",
                {"team_id": team, "models": [model], "disable_global_guardrails": True},
                key=caller,
            )
            if generated.status_code == 200:
                scenario.cleanups.callback(scenario.delete_key, string_value(generated.json()["key"]))
            assert generated.status_code == 403, generated.text
            assert "disable_global_guardrails" in generated.text

            smuggled: Final = candidate.request(
                "POST",
                "/key/generate",
                {"team_id": team, "models": [model], "metadata": {"disable_global_guardrails": True}},
                key=caller,
            )
            if smuggled.status_code == 200:
                scenario.cleanups.callback(scenario.delete_key, string_value(smuggled.json()["key"]))
            assert smuggled.status_code == 403, smuggled.text

            updated: Final = candidate.request(
                "POST", "/key/update", {"key": own, "disable_global_guardrails": True}, key=caller
            )
            assert updated.status_code == 403, updated.text
            regenerated: Final = candidate.request(
                "POST", "/key/regenerate", {"key": own, "disable_global_guardrails": True}, key=caller
            )
            assert regenerated.status_code == 403, regenerated.text
            assert "disable_global_guardrails" not in stored_metadata(own)

            blocked: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "synthetic denied marker"}]},
                key=own,
            )
            assert blocked.status_code == 400 and "synthetic policy denial" in blocked.text, blocked.text

            exempt: Final = scenario.key(team_id=team, models=[model], disable_global_guardrails=True)
            assert stored_metadata(exempt)["disable_global_guardrails"] is True
            resaved: Final = candidate.request(
                "POST",
                "/key/update",
                {
                    "key": exempt,
                    "key_alias": "renamed" + uuid.uuid4().hex,
                    "metadata": {"disable_global_guardrails": True},
                },
                key=caller,
            )
            assert resaved.status_code == 200, resaved.text
            assert stored_metadata(exempt)["disable_global_guardrails"] is True
            served: Final = candidate.chat(model, key=exempt, text="synthetic denied marker")
            assert object_value(served["usage"])["total_tokens"] == 40
            assert len(policy.drain()) == 1


def _team_metadata(team_id: str) -> dict[str, object]:
    rows: Final = read_rows('SELECT metadata FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))
    assert len(rows) == 1, rows
    return rows[0]["metadata"]


def _drop_created_key(scenario: Scenario, response: httpx.Response) -> None:
    if response.status_code == 200:
        scenario.cleanups.callback(scenario.delete_key, string_value(response.json()["key"]))


def test_non_admin_flag_denied_on_every_key_write_route(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(denying_guardrail) as policy:
        config: Final = guardrail_config(policy.url, tmp_path / "denied-routes.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            member: Final = scenario.user(user_role="internal_user")
            team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
            caller: Final = non_admin_caller(scenario, member, team, model)
            own: Final = scenario.key(team_id=team, models=[model])

            attempts: Final = (
                ("POST", "/key/generate", {"team_id": team, "models": [model], "disable_global_guardrails": True}),
                (
                    "POST",
                    "/key/generate",
                    {"team_id": team, "models": [model], "metadata": {"disable_global_guardrails": True}},
                ),
                (
                    "POST",
                    "/key/generate",
                    {
                        "team_id": team,
                        "models": [model],
                        "disable_global_guardrails": False,
                        "metadata": {"disable_global_guardrails": True},
                    },
                ),
                ("POST", "/key/update", {"key": own, "disable_global_guardrails": True}),
                ("POST", "/key/update", {"key": own, "metadata": {"disable_global_guardrails": True}}),
                ("POST", "/key/regenerate", {"key": own, "disable_global_guardrails": True}),
                ("POST", f"/key/{own}/regenerate", {"disable_global_guardrails": True}),
                (
                    "POST",
                    "/key/service-account/generate",
                    {"team_id": team, "disable_global_guardrails": True},
                ),
            )
            for method, path, body in attempts:
                response: Final = candidate.request(method, path, body, key=caller)
                _drop_created_key(scenario, response)
                assert response.status_code == 403, f"{method} {path}: {response.text}"
                assert "disable_global_guardrails" in response.text, response.text
            assert "disable_global_guardrails" not in stored_metadata(own)

            service_alias: Final = "audit-sa-" + uuid.uuid4().hex
            service_denied: Final = candidate.request(
                "POST",
                "/key/service-account/generate",
                {"team_id": team, "key_alias": service_alias, "disable_global_guardrails": True},
                key=caller,
            )
            _drop_created_key(scenario, service_denied)
            assert (
                read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias = %s', (service_alias,)) == []
            ), service_denied.text


def test_non_admin_flag_denied_on_team_new(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(denying_guardrail) as policy:
        config: Final = guardrail_config(policy.url, tmp_path / "denied-team.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            member: Final = scenario.user(user_role="internal_user")
            team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
            caller: Final = non_admin_caller(scenario, member, team, model)

            alias: Final = "audit-team-" + uuid.uuid4().hex
            denied: Final = candidate.request(
                "POST",
                "/team/new",
                {"team_alias": alias, "models": [model], "disable_global_guardrails": True},
                key=caller,
            )
            created: Final = read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_alias = %s', (alias,))
            for row in created:
                scenario.cleanups.callback(scenario.delete_team, str(row["team_id"]))
            assert denied.status_code == 403, denied.text
            assert "disable_global_guardrails" in denied.text, denied.text


def test_admin_flag_writes_succeed_on_all_routes(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(denying_guardrail) as policy:
        config: Final = guardrail_config(policy.url, tmp_path / "admin-routes.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            team: Final = scenario.team(models=[model])

            generated: Final = candidate.post(
                "/key/generate", {"team_id": team, "models": [model], "disable_global_guardrails": True}
            )
            generated_key: Final = string_value(generated["key"])
            scenario.cleanups.callback(scenario.delete_key, generated_key)
            assert stored_metadata(generated_key)["disable_global_guardrails"] is True

            plain: Final = scenario.key(team_id=team, models=[model])
            candidate.post("/key/update", {"key": plain, "disable_global_guardrails": True})
            assert stored_metadata(plain)["disable_global_guardrails"] is True

            regen_source: Final = string_value(
                candidate.post("/key/generate", {"team_id": team, "models": [model]})["key"]
            )
            regenerated: Final = candidate.post(
                "/key/regenerate", {"key": regen_source, "disable_global_guardrails": True}
            )
            regenerated_key: Final = string_value(regenerated["key"])
            scenario.cleanups.callback(scenario.delete_key, regenerated_key)
            assert stored_metadata(regenerated_key)["disable_global_guardrails"] is True

            new_team: Final = candidate.post(
                "/team/new", {"team_alias": "audit-admin-" + uuid.uuid4().hex, "disable_global_guardrails": True}
            )
            new_team_id: Final = string_value(new_team["team_id"])
            scenario.cleanups.callback(scenario.delete_team, new_team_id)
            assert _team_metadata(new_team_id)["disable_global_guardrails"] is True

            candidate.post("/team/update", {"team_id": team, "disable_global_guardrails": True})
            assert _team_metadata(team)["disable_global_guardrails"] is True


def test_non_admin_resave_omit_and_revoke_sequences(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(denying_guardrail) as policy:
        config: Final = guardrail_config(policy.url, tmp_path / "resave.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            member: Final = scenario.user(user_role="internal_user")
            team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
            caller: Final = non_admin_caller(scenario, member, team, model)
            exempt: Final = scenario.key(team_id=team, models=[model], disable_global_guardrails=True)
            assert stored_metadata(exempt)["disable_global_guardrails"] is True

            resaved: Final = candidate.request(
                "POST",
                "/key/update",
                {
                    "key": exempt,
                    "key_alias": "audit-resave-" + uuid.uuid4().hex,
                    "metadata": {"disable_global_guardrails": True},
                },
                key=caller,
            )
            assert resaved.status_code == 200, resaved.text
            assert stored_metadata(exempt)["disable_global_guardrails"] is True

            omitted: Final = candidate.request(
                "POST",
                "/key/update",
                {"key": exempt, "key_alias": "audit-omit-" + uuid.uuid4().hex},
                key=caller,
            )
            assert omitted.status_code == 200, omitted.text

            candidate.post("/key/update", {"key": exempt, "disable_global_guardrails": False})
            assert stored_metadata(exempt)["disable_global_guardrails"] is False

            rejected: Final = candidate.request(
                "POST", "/key/update", {"key": exempt, "disable_global_guardrails": True}, key=caller
            )
            assert rejected.status_code == 403, rejected.text
            assert "disable_global_guardrails" in rejected.text, rejected.text
            assert stored_metadata(exempt)["disable_global_guardrails"] is False


def test_generate_ignores_server_default_metadata_flag(gateway: Gateway, tmp_path: Path) -> None:
    raw: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    raw.setdefault("litellm_settings", {})["default_key_generate_params"] = {
        "metadata": {"disable_global_guardrails": True}
    }
    path: Final = tmp_path / "server-defaults.yaml"
    path.write_text(yaml.safe_dump(raw))
    with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        member: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
        caller: Final = non_admin_caller(scenario, member, team, model)

        generated: Final = candidate.post("/key/generate", {"team_id": team, "models": [model]}, key=caller)
        generated_key: Final = string_value(generated["key"])
        scenario.cleanups.callback(scenario.delete_key, generated_key)
        assert stored_metadata(generated_key)["disable_global_guardrails"] is True

        explicit: Final = candidate.request(
            "POST",
            "/key/generate",
            {"team_id": team, "models": [model], "metadata": {"disable_global_guardrails": True}},
            key=caller,
        )
        _drop_created_key(scenario, explicit)
        assert explicit.status_code == 403, explicit.text
        assert "disable_global_guardrails" in explicit.text, explicit.text


def test_sad_flag_inputs_on_key_generate(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {}) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        member: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
        caller: Final = non_admin_caller(scenario, member, team, model)

        denied_bodies: Final = (
            {"team_id": team, "models": [model], "disable_global_guardrails": "true"},
            {"team_id": team, "models": [model], "disable_global_guardrails": 1},
            {"team_id": team, "models": [model], "metadata": {"disable_global_guardrails": "true"}},
            {"team_id": team, "models": [model], "metadata": {"disable_global_guardrails": 1}},
            {"team_id": team, "models": [model], "metadata": {"disable_global_guardrails": "x" * 5120}},
        )
        for body in denied_bodies:
            response: Final = candidate.request("POST", "/key/generate", body, key=caller)
            _drop_created_key(scenario, response)
            assert response.status_code == 403, response.text
            assert "disable_global_guardrails" in response.text, response.text

        invalid_bodies: Final = (
            {"team_id": team, "models": [model], "disable_global_guardrails": []},
            {"team_id": team, "models": [model], "disable_global_guardrails": {}},
        )
        for body in invalid_bodies:
            rejected: Final = candidate.request("POST", "/key/generate", body, key=caller)
            assert rejected.status_code == 422, rejected.text

        unauthenticated: Final = candidate.request(
            "POST",
            "/key/generate",
            {"team_id": team, "models": [model], "disable_global_guardrails": True},
            key="sk-not-a-real-key-" + uuid.uuid4().hex,
        )
        assert unauthenticated.status_code == 401, unauthenticated.text

        repeat_alias: Final = "audit-repeat-" + uuid.uuid4().hex
        for _ in range(2):
            repeated: Final = candidate.request(
                "POST",
                "/key/generate",
                {"team_id": team, "models": [model], "key_alias": repeat_alias, "disable_global_guardrails": True},
                key=caller,
            )
            _drop_created_key(scenario, repeated)
            assert repeated.status_code == 403, repeated.text
        assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias = %s', (repeat_alias,)) == []


def test_falsy_metadata_flag_shapes_stay_stored_and_guarded(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(denying_guardrail) as policy:
        config: Final = guardrail_config(policy.url, tmp_path / "falsy.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            member: Final = scenario.user(user_role="internal_user")
            team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
            caller: Final = non_admin_caller(scenario, member, team, model)

            for shape in ([], {}):
                created: Final = candidate.request(
                    "POST",
                    "/key/generate",
                    {"team_id": team, "models": [model], "metadata": {"disable_global_guardrails": shape}},
                    key=caller,
                )
                _drop_created_key(scenario, created)
                assert created.status_code == 200, created.text
                token: Final = string_value(created.json()["key"])
                assert stored_metadata(token)["disable_global_guardrails"] == shape
                blocked: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "synthetic denied marker"}]},
                    key=token,
                )
                assert blocked.status_code == 400 and "synthetic policy denial" in blocked.text, blocked.text

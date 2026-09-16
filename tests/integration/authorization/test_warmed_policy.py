from contextlib import ExitStack
from hashlib import sha256
from typing import Final
import os

import psycopg
import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test

from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.generation import LIFECYCLE_SETTINGS, bounded_http_requests


def assert_serving(gateway: Gateway, model: str, key: str, status: int, error_type: str = "auth_error") -> None:
    response: Final = eventually(
        lambda: gateway.request(
            "POST", "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "warmed policy control"}]}, key=key,
        ),
        lambda value: value.status_code == status,
        seconds=3,
    )
    if status == 200:
        assert response.json()["usage"]["total_tokens"] == 40
        assert response.json()["choices"][0]["message"]["content"] == (
            "Hello! This is a mock response from the fake OpenAI endpoint."
        )
    else:
        assert response.json()["error"]["type"] == error_type


@pytest.mark.covers("mgmt.key.update.two_workers_enforce_warmed_policy")
def test_generated_policy_changes_reach_both_warmed_workers(gateway: Gateway, peer: Gateway) -> None:
    class Policies(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            self.resources = ExitStack()
            try:
                scenario = self.resources.enter_context(gateway.scenario())
                self.models = (scenario.model(), scenario.model())
                self.allowed = 0
                self.blocked = False
                self.key = scenario.key(models=[self.models[0]], blocked=False)
                self.control = scenario.key(models=list(self.models))
                for worker in (gateway, peer):
                    assert_serving(worker, self.models[0], self.key, 200)
                    assert_serving(worker, self.models[1], self.control, 200)
            except BaseException:
                with budget.cleanup():
                    self.resources.close()
                raise

        @rule(index=st.integers(min_value=0, max_value=1))
        def model_grant(self, index: int) -> None:
            gateway.post("/key/update", {"key": self.key, "models": [self.models[index]]})
            self.allowed = index

        @rule(blocked=st.booleans())
        def block(self, blocked: bool) -> None:
            gateway.post("/key/update", {"key": self.key, "blocked": blocked})
            self.blocked = blocked

        @invariant()
        def both_workers_enforce_policy(self) -> None:
            rows: Final = read_rows(
                'SELECT models, blocked FROM "LiteLLM_VerificationToken" WHERE token = %s',
                (sha256(self.key.encode()).hexdigest(),),
            )
            assert rows == [{"models": [self.models[self.allowed]], "blocked": self.blocked}]
            for worker in (gateway, peer):
                for index, model in enumerate(self.models):
                    status: Final = 401 if self.blocked else 200 if index == self.allowed else 403
                    kind: Final = "auth_error" if self.blocked else "key_model_access_denied"
                    assert_serving(worker, model, self.key, status, kind)
                assert_serving(worker, self.models[1], self.control, 200)

        def teardown(self) -> None:
            with budget.cleanup():
                self.resources.close()

    with bounded_http_requests((gateway, peer), limit=3000) as budget:
        run_state_machine_as_test(Policies, settings=LIFECYCLE_SETTINGS)


@pytest.mark.covers("mgmt.user.scim.deactivation_includes_nullable_blocked_keys")
def test_scim_deactivation_blocks_null_and_false_keys_but_preserves_other_owners(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        other: Final = scenario.user(user_role="internal_user")
        null_key: Final = scenario.key(user_id=user, models=[model])
        false_key: Final = scenario.key(user_id=user, models=[model], blocked=False)
        manual: Final = scenario.key(user_id=user, models=[model], blocked=True)
        control: Final = scenario.key(user_id=other, models=[model])
        team: Final = scenario.team(models=[model])
        service: Final = gateway.post("/key/service-account/generate", {"team_id": team, "models": [model]})
        service_key: Final = service["key"]
        assert isinstance(service_key, str)
        scenario.cleanups.callback(scenario.delete_key, service_key)
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute(
                'UPDATE "LiteLLM_VerificationToken" SET blocked = NULL WHERE token = %s',
                (sha256(null_key.encode()).hexdigest(),),
            )
        assert read_rows(
            'SELECT user_id, blocked FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(null_key.encode()).hexdigest(),),
        ) == [{"user_id": user, "blocked": None}]
        assert read_rows(
            'SELECT user_id FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(service_key.encode()).hexdigest(),),
        ) == [{"user_id": None}]
        for token in (null_key, false_key, control, service_key):
            assert_serving(gateway, model, token, 200)
        for active in (False, True):
            response: Final = gateway.request(
                "PATCH", f"/scim/v2/Users/{user}",
                {"schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                 "Operations": [{"op": "replace", "path": "active", "value": active}]},
            )
            assert response.status_code == 200, response.text
            for token in (null_key, false_key):
                rows: Final = read_rows(
                    'SELECT blocked, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
                    (sha256(token.encode()).hexdigest(),),
                )
                assert rows[0]["blocked"] is not active
                assert object_value(rows[0]["metadata"]).get("scim_blocked") is (None if active else True)
                assert_serving(gateway, model, token, 200 if active else 401)
            assert_serving(gateway, model, manual, 401)
            for token in (control, service_key):
                assert_serving(gateway, model, token, 200)


@pytest.mark.covers("mgmt.team.member_update.demoted_role_cannot_write")
def test_warmed_team_role_demotion_prevents_later_management_writes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(models=[model], members_with_roles=[{"user_id": user, "role": "admin"}])
        control_team: Final = scenario.team(models=[model])
        caller: Final = scenario.key(
            user_id=user, team_id=team, models=[model], allowed_routes=["/team/update", "/v1/chat/completions"]
        )
        gateway.chat(model, key=caller)
        changed: Final = gateway.request("POST", "/team/update", {"team_id": team, "team_alias": "permitted"}, key=caller)
        assert changed.status_code == 200, changed.text
        unrelated_before: Final = read_rows(
            'SELECT team_alias FROM "LiteLLM_TeamTable" WHERE team_id = %s', (control_team,)
        )
        unrelated: Final = gateway.request(
            "POST", "/team/update", {"team_id": control_team, "team_alias": "must-not-persist"}, key=caller
        )
        assert unrelated.status_code == 403, unrelated.text
        assert read_rows(
            'SELECT team_alias FROM "LiteLLM_TeamTable" WHERE team_id = %s', (control_team,)
        ) == unrelated_before
        gateway.post("/team/member_update", {"team_id": team, "user_id": user, "role": "user"})
        for target in (team, control_team):
            before: Final = read_rows('SELECT team_alias FROM "LiteLLM_TeamTable" WHERE team_id = %s', (target,))
            denied: Final = gateway.request(
                "POST", "/team/update", {"team_id": target, "team_alias": "must-not-persist"}, key=caller
            )
            assert denied.status_code == 403, denied.text
            assert read_rows('SELECT team_alias FROM "LiteLLM_TeamTable" WHERE team_id = %s', (target,)) == before
        roster: Final = read_rows('SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
        members: Final = roster[0]["members_with_roles"]
        assert isinstance(members, list)
        assert next(object_value(member)["role"] for member in members if object_value(member)["user_id"] == user) == "user"
        assert_serving(gateway, model, caller, 200)


@pytest.mark.covers("mgmt.key.update.expiry_changes_reach_warmed_workers")
def test_expiry_and_explicit_clear_reach_both_warmed_workers(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], duration="1h")
        control: Final = scenario.key(models=[model])
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 200)
        gateway.post("/key/update", {"key": key, "duration": "0s"})
        assert read_rows(
            "SELECT expires <= timezone('UTC', now()) AS expired FROM \"LiteLLM_VerificationToken\" WHERE token = %s",
            (sha256(key.encode()).hexdigest(),),
        ) == [{"expired": True}]
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 401, "expired_key")
            assert_serving(worker, model, control, 200)
        gateway.post("/key/update", {"key": key, "duration": None})
        assert read_rows(
            'SELECT expires IS NULL AS cleared FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        ) == [{"cleared": True}]
        for worker in (gateway, peer):
            assert_serving(worker, model, key, 200)

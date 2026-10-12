from contextlib import ExitStack
from hashlib import sha256
from typing import Final

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test
from pydantic import JsonValue

from tests.integration._support.client import Gateway, object_value
from tests.integration._support.database import read_rows
from tests.integration._support.generation import LIFECYCLE_SETTINGS, bounded_http_requests


def _key_rows(digest: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT token, key_name, key_alias, models, aliases, config, router_settings, user_id, team_id, '
        'agent_id, project_id, permissions, max_parallel_requests, metadata, blocked, tpm_limit, rpm_limit, '
        'tpd_limit, max_budget, budget_duration, allowed_cache_controls, allowed_routes, key_type, policies, '
        'access_group_ids, model_spend, model_max_budget, budget_fallbacks, budget_id, organization_id, '
        'object_permission_id, budget_limits FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (digest,),
    )


@pytest.mark.covers("mgmt.key.update.generated_sequences_preserve_state")
def test_generated_partial_updates_preserve_persisted_and_effective_state(gateway: Gateway) -> None:
    class KeyUpdates(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            self.resources = ExitStack()
            try:
                scenario = self.resources.enter_context(gateway.scenario())
                self.models = (scenario.model(), scenario.model())
                self.key = scenario.key(models=[self.models[0]], key_alias="initial", metadata={"revision": "initial"})
                self.expected: dict[str, JsonValue] = {
                    "models": [self.models[0]], "key_alias": "initial", "metadata": {"revision": "initial"}
                }
                gateway.chat(self.models[0], key=self.key)
            except BaseException:
                with budget.cleanup():
                    self.resources.close()
                raise

        @rule(alias=st.sampled_from(("first", "second", "", "unicode-λ")))
        def alias(self, alias: str) -> None:
            gateway.post("/key/update", {"key": self.key, "key_alias": alias})
            self.expected["key_alias"] = alias

        @rule(index=st.integers(min_value=0, max_value=1), both=st.booleans())
        def grant(self, index: int, both: bool) -> None:
            models: Final = list(self.models) if both else [self.models[index]]
            gateway.post("/key/update", {"key": self.key, "models": models})
            self.expected["models"] = models

        @rule(value=st.sampled_from(("", "a", "different", "λ")))
        def metadata(self, value: str) -> None:
            gateway.post("/key/update", {"key": self.key, "metadata": {"revision": value}})
            self.expected["metadata"] = {"revision": value}

        @invariant()
        def persisted_state_and_serving_match(self) -> None:
            rows: Final = read_rows(
                'SELECT models, key_alias, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
                (sha256(self.key.encode()).hexdigest(),),
            )
            assert rows == [self.expected]
            info: Final = object_value(gateway.get("/key/info", {"key": self.key})["info"])
            assert {field: info[field] for field in self.expected} == self.expected
            for model in self.models:
                response: Final = gateway.request(
                    "POST", "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "generated update control"}]},
                    key=self.key,
                )
                if model in self.expected["models"]:
                    assert response.status_code == 200, response.text
                    assert response.json()["usage"]["total_tokens"] == 40
                else:
                    assert response.status_code == 403, response.text
                    assert response.json()["error"]["type"] == "key_model_access_denied"

        def teardown(self) -> None:
            with budget.cleanup():
                self.resources.close()

    with bounded_http_requests((gateway,), limit=2000) as budget:
        run_state_machine_as_test(KeyUpdates, settings=LIFECYCLE_SETTINGS)


@pytest.mark.covers("mgmt.key.update.false_zero_and_empty_values_affect_serving")
def test_zero_false_and_empty_values_are_not_treated_as_omission(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        models: Final = (scenario.model(), scenario.model())
        key: Final = scenario.key(models=[models[0]], max_budget=0, metadata={"ordinary": "value"})
        denied: Final = gateway.request(
            "POST", "/v1/chat/completions",
            {"model": models[0], "messages": [{"role": "user", "content": "zero budget"}]}, key=key,
        )
        assert denied.status_code == 422, denied.text
        assert denied.json()["error"]["type"] == "budget_exceeded"
        gateway.post("/key/update", {"key": key, "max_budget": 1, "models": [], "metadata": {}})
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert (info["models"], info["metadata"], info["max_budget"]) == ([], {}, 1)
        for model in models:
            assert object_value(gateway.chat(model, key=key)["usage"])["total_tokens"] == 40
        gateway.post("/key/update", {"key": key, "blocked": True})
        blocked: Final = gateway.request(
            "POST", "/v1/chat/completions",
            {"model": models[0], "messages": [{"role": "user", "content": "blocked control"}]}, key=key,
        )
        assert blocked.status_code == 401, blocked.text
        assert blocked.json()["error"]["type"] == "auth_error"
        gateway.post("/key/update", {"key": key, "blocked": False})
        assert object_value(gateway.chat(models[0], key=key)["usage"])["total_tokens"] == 40
        rows: Final = read_rows(
            'SELECT blocked, models, metadata, max_budget FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        )
        assert rows == [{"blocked": False, "models": [], "metadata": {}, "max_budget": 1.0}]
        gateway.post("/key/update", {"key": key, "max_budget": 0})
        assert read_rows(
            'SELECT max_budget FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        ) == [{"max_budget": 0.0}]
        zero_after_update: Final = gateway.request(
            "POST", "/v1/chat/completions",
            {"model": models[0], "messages": [{"role": "user", "content": "updated zero budget"}]}, key=key,
        )
        assert zero_after_update.status_code == 422, zero_after_update.text
        assert zero_after_update.json()["error"]["type"] == "budget_exceeded"
        gateway.post("/key/update", {"key": key, "max_budget": None})
        assert read_rows(
            'SELECT max_budget FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        ) == [{"max_budget": None}]
        assert object_value(gateway.chat(models[0], key=key)["usage"])["total_tokens"] == 40


@pytest.mark.covers("mgmt.key.update.project_clear_preserves_scope", "mgmt.key.update.invalid_batch_is_atomic")
def test_project_omission_clear_and_invalid_update_have_distinct_effects(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        outside: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model])
        other: Final = scenario.project(team, models=[model])
        key: Final = scenario.key(team_id=team, project_id=project, models=[model], key_alias="before", max_budget=5)
        gateway.chat(model, key=key)
        gateway.post("/key/update", {"key": key, "key_alias": "after"})
        digest: Final = sha256(key.encode()).hexdigest()

        def saved() -> list[dict[str, object]]:
            return read_rows(
                'SELECT key_alias, project_id, team_id, models, max_budget FROM "LiteLLM_VerificationToken" '
                'WHERE token = %s', (digest,),
            )

        before: Final = saved()
        assert before == [{"key_alias": "after", "project_id": project, "team_id": team, "models": [model], "max_budget": 5}]
        for invalid in (other, ""):
            rejected: Final = gateway.request(
                "POST", "/key/update", {"key": key, "project_id": invalid, "key_alias": "must-not-persist"}
            )
            assert rejected.status_code == 400, rejected.text
            assert saved() == before
            gateway.chat(model, key=key)
        gateway.post("/project/update", {"project_id": project, "blocked": True})
        denied: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "blocked project"}]},
            key=key,
        )
        assert denied.status_code == 401, denied.text
        assert denied.json()["error"]["type"] == "auth_error"
        for _ in range(2):
            gateway.post("/key/update", {"key": key, "project_id": None})
            assert saved() == [{**before[0], "project_id": None}]
            assert object_value(gateway.chat(model, key=key)["usage"])["total_tokens"] == 40
            outside_request: Final = gateway.request(
                "POST", "/v1/chat/completions",
                {"model": outside, "messages": [{"role": "user", "content": "detached scope control"}]}, key=key,
            )
            assert outside_request.status_code == 403, outside_request.text
            assert outside_request.json()["error"]["type"] == "key_model_access_denied"


@pytest.mark.covers("mgmt.key.update.denied_request_preserves_effective_state")
def test_denied_key_update_preserves_saved_grants_and_serving(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        outside: Final = scenario.model()
        owner: Final = scenario.user(user_role="internal_user")
        other: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=owner, models=[model], key_alias="unchanged", max_budget=2)
        caller: Final = scenario.key(user_id=other, models=[model], allowed_routes=["/key/update", "/v1/chat/completions"])
        gateway.chat(model, key=key)
        denied: Final = gateway.request(
            "POST", "/key/update", {"key": key, "key_alias": "wrong", "models": [outside], "max_budget": 0}, key=caller
        )
        assert denied.status_code == 403, denied.text
        assert read_rows(
            'SELECT user_id, models, key_alias, max_budget FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        ) == [{"user_id": owner, "models": [model], "key_alias": "unchanged", "max_budget": 2.0}]
        assert object_value(gateway.chat(model, key=key)["usage"])["total_tokens"] == 40
        rejected: Final = gateway.request(
            "POST", "/v1/chat/completions",
            {"model": outside, "messages": [{"role": "user", "content": "unchanged scope"}]}, key=key,
        )
        assert rejected.status_code == 403, rejected.text
        assert rejected.json()["error"]["type"] == "key_model_access_denied"


@pytest.mark.covers("mgmt.key.update.project_detach_denied_to_restricted_actor")
def test_restricted_actor_cannot_detach_key_from_project(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model], team_member_permissions=["/key/update"])
        project: Final = scenario.project(team, models=[model])
        member: Final = scenario.user(user_role="internal_user")
        gateway.post(
            "/team/member_add",
            {"team_id": team, "member": {"user_id": member, "role": "user"}},
        )
        target: Final = scenario.key(user_id=member, team_id=team, project_id=project, models=[model])
        caller: Final = scenario.key(
            user_id=member,
            team_id=team,
            models=[model],
            allowed_routes=["/key/update"],
        )
        digest: Final = sha256(target.encode()).hexdigest()
        before: Final = _key_rows(digest)
        assert len(before) == 1
        assert before[0]["project_id"] == project
        assert before[0]["team_id"] == team
        denied: Final = gateway.request(
            "POST", "/key/update", {"key": target, "project_id": None}, key=caller
        )
        assert denied.status_code == 403, denied.text
        assert _key_rows(digest) == before


@pytest.mark.covers(
    "mgmt.key.info.cross_tenant_key_is_denied",
    "mgmt.key.update.cross_tenant_key_is_denied",
    "mgmt.key.update.cross_tenant_project_detach_is_denied",
)
def test_cross_tenant_actor_cannot_read_update_or_detach_project_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        foreign_team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model])
        foreign_user: Final = scenario.user(user_role="internal_user")
        gateway.post(
            "/team/member_add",
            {"team_id": foreign_team, "member": {"user_id": foreign_user, "role": "user"}},
        )
        target: Final = scenario.key(team_id=team, project_id=project, models=[model])
        caller: Final = scenario.key(
            user_id=foreign_user,
            team_id=foreign_team,
            models=[model],
            allowed_routes=["/key/info", "/key/update"],
        )
        digest: Final = sha256(target.encode()).hexdigest()
        before: Final = _key_rows(digest)
        assert len(before) == 1
        assert before[0]["project_id"] == project
        assert before[0]["team_id"] == team
        info_denied: Final = gateway.request(
            "GET", "/key/info", params={"key": digest}, key=caller
        )
        assert info_denied.status_code == 403, info_denied.text
        assert target not in info_denied.text
        assert digest not in info_denied.text
        assert project not in info_denied.text
        assert team not in info_denied.text
        update_denied: Final = gateway.request(
            "POST", "/key/update", {"key": target, "key_alias": "foreign-update"}, key=caller
        )
        assert update_denied.status_code == 401, update_denied.text
        detach_denied: Final = gateway.request(
            "POST", "/key/update", {"key": target, "project_id": None}, key=caller
        )
        assert detach_denied.status_code == 401, detach_denied.text
        for response in (update_denied, detach_denied):
            assert target not in response.text
            assert digest not in response.text
            assert project not in response.text
        assert _key_rows(digest) == before

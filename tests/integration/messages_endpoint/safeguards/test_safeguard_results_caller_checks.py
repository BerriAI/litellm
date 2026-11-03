"""The classifier call runs as the caller: a key, team, member, user, end user, agent, or budget that cannot use the
classifier model turns the whole safeguard into ``unsupported`` instead of charging or bypassing the caller's limits,
the router keeps the call on the deployments the caller's access groups grant, and a limit that only holds for this
window (a full RPM or TPM window, or counters Redis cannot read) makes this turn's tool uses ``unavailable`` and leaves
the next turn on the server check, while the slot the turn itself took never denies its classifier call. A native call
that sends its own ``metadata`` is charged to the same per-model budgets as the classifier call."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Scenario, eventually, object_value, string_value
from integration._support.process import graceful_stop_seconds
from integration._support.redis_process import owned_redis
from integration._support.safeguards_rig import (
    CHAT_MODEL,
    CLASSIFIER_MODEL,
    NATIVE_MODEL,
    UNSUPPORTED,
    Rig,
    auto_mode_body,
    chat_backend,
    classifier_answering,
    deployment,
    flagged,
    json_body,
    messages_body,
    native_backend,
    native_message,
    nonce,
    safeguards_rig,
    text_block,
    tool_id,
    tool_use_ids,
    unavailable,
)
from integration._support.wire import Request
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(graceful_stop_seconds() + 240)

TINY_BUDGET: Final = {CLASSIFIER_MODEL: {"max_budget": 1e-7, "budget_duration": "1d"}}
TINY_NATIVE_BUDGET: Final = {NATIVE_MODEL: {"max_budget": 1e-7, "budget_duration": "1d"}}


def classify(rig: Rig, key: str) -> tuple[dict[str, JsonValue], str]:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    return json_body(rig.post(auto_mode_body(CHAT_MODEL, request_nonce), key=key)), request_nonce


def assert_unsupported_without_a_classifier_call(rig: Rig, key: str) -> None:
    body, request_nonce = classify(rig, key)
    assert body["safeguard_results"] == list(UNSUPPORTED), body
    assert len(tool_use_ids(body)) == 1, body
    received: Final = rig.received(request_nonce)
    assert rig.classifier_calls(received) == (), [request.target for request in received]
    assert len(rig.backend_calls(received)) == 1, [request.target for request in received]


def test_key_without_the_classifier_model_gets_unsupported(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        assert_unsupported_without_a_classifier_call(rig, scenario.key(models=[CHAT_MODEL]))


def test_team_without_the_classifier_model_gets_unsupported(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        team_id: Final = scenario.team(models=[CHAT_MODEL])
        assert_unsupported_without_a_classifier_call(rig, scenario.key(team_id=team_id))


def assert_unavailable_for_the_turn_without_a_classifier_call(rig: Rig, key: str) -> None:
    body, request_nonce = classify(rig, key)
    ids: Final = tool_use_ids(body)
    assert len(ids) == 1, body
    assert body["safeguard_results"] == unavailable(ids, "error"), body
    received: Final = rig.received(request_nonce)
    assert rig.classifier_calls(received) == (), [request.target for request in received]
    assert len(rig.backend_calls(received)) == 1, [request.target for request in received]


def spend_one_classifier_request(rig: Rig, key: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False), classifier_answering("pong"))
    reply: Final = rig.gateway.chat(CLASSIFIER_MODEL, key=key, text=f"spend the window {request_nonce}")
    assert "pong" in str(reply), reply
    assert len(rig.classifier_calls(rig.received(request_nonce))) == 1


def test_classifier_model_rpm_window_the_caller_spent_is_unavailable_for_the_turn(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={CLASSIFIER_MODEL: 1})
        spend_one_classifier_request(rig, key)
        assert_unavailable_for_the_turn_without_a_classifier_call(rig, key)


def test_classifier_call_takes_no_slot_of_the_callers_classifier_model_rpm_limit(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={CLASSIFIER_MODEL: 1})
        first, _ = classify(rig, key)
        assert first["safeguard_results"] == flagged(tool_use_ids(first)), first
        second, _ = classify(rig, key)
        assert second["safeguard_results"] == flagged(tool_use_ids(second)), second


def assert_flagged_after_one_classifier_call(rig: Rig, key: str) -> None:
    body, request_nonce = classify(rig, key)
    assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
    received: Final = rig.received(request_nonce)
    assert len(rig.classifier_calls(received)) == 1, [request.target for request in received]


def test_key_rpm_limit_that_admitted_the_turn_leaves_its_classifier_call_running(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        assert_flagged_after_one_classifier_call(rig, scenario.key(rpm_limit=1))


def test_user_rpm_limit_that_admitted_the_turn_leaves_its_classifier_call_running(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user", rpm_limit=1)
        assert_flagged_after_one_classifier_call(rig, scenario.key(user_id=user_id))


def test_team_rpm_limit_that_admitted_the_turn_leaves_its_classifier_call_running(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        team_id: Final = scenario.team(rpm_limit=1)
        assert_flagged_after_one_classifier_call(rig, scenario.key(team_id=team_id))


def test_classifier_model_tpm_window_spent_by_earlier_turns_is_unavailable_for_the_turn(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_tpm_limit={CLASSIFIER_MODEL: 100})
        first, _ = classify(rig, key)
        assert first["safeguard_results"] == flagged(tool_use_ids(first)), first
        spent: Final = eventually(
            lambda: classify(rig, key)[0],
            lambda body: body["safeguard_results"] == unavailable(tool_use_ids(body), "error"),
            seconds=60,
        )
        assert len(tool_use_ids(spent)) == 1, spent


@pytest.mark.timeout(2 * graceful_stop_seconds() + 240)
def test_unreadable_rate_limit_counters_make_the_turn_unavailable(tmp_path: Path) -> None:
    with owned_redis(tmp_path / "redis") as redis:

        def redis_environment(wire_url: str) -> Mapping[str, str]:
            return {"REDIS_HOST": redis.host, "REDIS_PORT": str(redis.port)}

        with safeguards_rig(
            tmp_path,
            environment=redis_environment,
            general_settings={"fail_closed_rate_limit_enforcement": True},
        ) as rig:
            with rig.gateway.scenario() as scenario:
                key: Final = scenario.key(model_rpm_limit={CLASSIFIER_MODEL: 5})
                first, _ = classify(rig, key)
                assert first["safeguard_results"] == flagged(tool_use_ids(first)), first
                redis.stop()
                assert_unavailable_for_the_turn_without_a_classifier_call(rig, key)
                redis.start()


def test_key_model_budget_for_the_classifier_is_enforced(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=TINY_BUDGET)
        first, _ = classify(rig, key)
        assert first["safeguard_results"] == flagged(tool_use_ids(first)), first
        exhausted: Final = eventually(
            lambda: classify(rig, key)[0], lambda body: body["safeguard_results"] == list(UNSUPPORTED), seconds=60
        )
        assert len(tool_use_ids(exhausted)) == 1, exhausted


def test_team_member_model_budget_for_the_classifier_is_enforced(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        team_id: Final = scenario.team(team_member_model_max_budget=TINY_BUDGET)
        user_id: Final = scenario.member(team_id)
        key: Final = scenario.key(team_id=team_id, user_id=user_id)
        first, _ = classify(rig, key)
        assert first["safeguard_results"] == flagged(tool_use_ids(first)), first
        exhausted: Final = eventually(
            lambda: classify(rig, key)[0], lambda body: body["safeguard_results"] == list(UNSUPPORTED), seconds=60
        )
        assert len(tool_use_ids(exhausted)) == 1, exhausted


def native_turn_with_client_metadata(rig: Rig, key: str) -> tuple[httpx.Response, str]:
    request_nonce: Final = nonce()
    message: Final = native_message(tool_id(request_nonce), content=[text_block("done")], stop_reason="end_turn")
    rig.register(request_nonce, native_backend(message, stream=False))
    body: Final = messages_body(NATIVE_MODEL, request_nonce, metadata={"user_id": f"session-{request_nonce}"})
    return rig.post(body, key=key, beta=None), request_nonce


def test_native_call_sending_its_own_metadata_is_charged_to_the_key_model_budget(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(user_id=scenario.user(), model_max_budget=TINY_NATIVE_BUDGET)
        first, first_nonce = native_turn_with_client_metadata(rig, key)
        assert first.status_code == 200, first.text
        assert len(rig.backend_calls(rig.received(first_nonce))) == 1
        exhausted, exhausted_nonce = eventually(
            lambda: native_turn_with_client_metadata(rig, key), lambda turn: turn[0].status_code != 200, seconds=60
        )
        assert f"exceeded budget for model={NATIVE_MODEL}" in exhausted.text, exhausted.text
        assert rig.backend_calls(rig.received(exhausted_nonce)) == ()


def test_key_update_dropping_the_classifier_takes_effect_on_the_next_request(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[CHAT_MODEL, CLASSIFIER_MODEL])
        first, _ = classify(rig, key)
        assert first["safeguard_results"] == flagged(tool_use_ids(first)), first
        rig.gateway.post("/key/update", {"key": key, "models": [CHAT_MODEL]})
        exhausted: Final = eventually(
            lambda: classify(rig, key)[0], lambda body: body["safeguard_results"] == list(UNSUPPORTED), seconds=30
        )
        assert len(tool_use_ids(exhausted)) == 1, exhausted
        rig.gateway.post("/key/update", {"key": key, "models": [CHAT_MODEL, CLASSIFIER_MODEL]})
        restored: Final = eventually(
            lambda: classify(rig, key)[0], lambda body: body["safeguard_results"] != list(UNSUPPORTED), seconds=30
        )
        assert restored["safeguard_results"] == flagged(tool_use_ids(restored)), restored


def direct_classifier_status(rig: Rig, key: str, **extra: JsonValue) -> int:
    body: Final = {"model": CLASSIFIER_MODEL, "messages": [{"role": "user", "content": f"direct {nonce()}"}], **extra}
    return rig.gateway.request("POST", "/v1/chat/completions", body, key=key).status_code


def customer(scenario: Scenario, **fields: JsonValue) -> str:
    identity: Final = f"integration-{uuid.uuid4().hex}"
    scenario.gateway.post("/customer/new", {"user_id": identity, **fields})
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [identity]})
    return identity


def test_end_user_without_the_classifier_model_gets_unsupported(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        end_user: Final = customer(scenario, models=[CHAT_MODEL])
        assert direct_classifier_status(rig, key, user=end_user) == 403
        request_nonce: Final = nonce()
        rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
        body: Final = json_body(
            rig.post(auto_mode_body(CHAT_MODEL, request_nonce, metadata={"user_id": end_user}), key=key)
        )
        assert body["safeguard_results"] == list(UNSUPPORTED), body
        received: Final = rig.received(request_nonce)
        assert rig.classifier_calls(received) == (), [request.target for request in received]
        assert len(rig.backend_calls(received)) == 1, [request.target for request in received]


def agent_key(scenario: Scenario, models: Sequence[str]) -> str:
    gateway: Final = scenario.gateway
    group: Final = gateway.request(
        "POST",
        "/v1/access_group",
        {"access_group_name": f"integration-{uuid.uuid4().hex}", "access_model_names": list(models)},
    )
    assert group.status_code == 201, group.text
    group_id: Final = string_value(object_value(group.json())["access_group_id"])
    scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/access_group/{group_id}")
    agent: Final = gateway.post(
        "/v1/agents",
        {
            "agent_name": f"integration-{uuid.uuid4().hex[:10]}",
            "agent_card_params": {
                "protocolVersion": "0.3.0",
                "name": "integration",
                "description": "integration agent",
                "url": "http://127.0.0.1:1/agent",
                "version": "1",
                "capabilities": {},
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
                "skills": [],
            },
            "access_group_ids": [group_id],
        },
    )
    agent_id: Final = string_value(agent["agent_id"])
    scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/agents/{agent_id}")
    return scenario.key(agent_id=agent_id)


def test_agent_whose_access_groups_omit_the_classifier_gets_unsupported(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = agent_key(scenario, [CHAT_MODEL])
        assert direct_classifier_status(rig, key) == 403
        assert_unsupported_without_a_classifier_call(rig, key)


def test_personal_key_of_a_user_without_the_classifier_model_gets_unsupported(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user", models=[CHAT_MODEL])
        key: Final = scenario.key(user_id=user_id)
        assert direct_classifier_status(rig, key) == 403
        assert_unsupported_without_a_classifier_call(rig, key)


def test_team_key_reaches_the_classifier_its_team_grants_beyond_its_users_personal_models(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        team_id: Final = scenario.team(models=[CHAT_MODEL, CLASSIFIER_MODEL])
        user_id: Final = scenario.user(user_role="internal_user", models=[CHAT_MODEL])
        rig.gateway.post("/team/member_add", {"team_id": team_id, "member": {"role": "user", "user_id": user_id}})
        key: Final = scenario.key(team_id=team_id, user_id=user_id)
        spend_one_classifier_request(rig, key)
        assert_flagged_after_one_classifier_call(rig, key)


GROUPED_CLASSIFIER: Final = "safeguards-classifier-grouped"
GRANTED_GROUP: Final = "safeguards-group-granted"
OTHER_GROUP: Final = "safeguards-group-other"
GRANTED_TARGET: Final = "/classifier/granted/v1/chat/completions"


def grouped_classifiers(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return (
        {
            **deployment(GROUPED_CLASSIFIER, "openai/gpt-5.4-mini", f"{wire_url}/classifier/granted/v1", weight=1),
            "model_info": {"access_groups": [GRANTED_GROUP]},
        },
        {
            **deployment(GROUPED_CLASSIFIER, "openai/gpt-5.4-mini", f"{wire_url}/classifier/other/v1", weight=1000),
            "model_info": {"access_groups": [OTHER_GROUP]},
        },
    )


def classifier_targets(requests: Sequence[Request]) -> tuple[str, ...]:
    return tuple(request.target for request in requests if request.target.startswith("/classifier/"))


@pytest.mark.timeout(graceful_stop_seconds() + 240)
def test_classifier_call_keeps_to_the_deployments_the_callers_access_group_grants(tmp_path: Path) -> None:
    with safeguards_rig(tmp_path, classifier_model=GROUPED_CLASSIFIER, extra_models=grouped_classifiers) as rig:
        with rig.gateway.scenario() as scenario:
            key: Final = scenario.key(models=[CHAT_MODEL, GRANTED_GROUP])
            direct_nonce: Final = nonce()
            rig.register(direct_nonce, chat_backend(tool_id(direct_nonce), stream=False), classifier_answering("pong"))
            assert "pong" in str(rig.gateway.chat(GROUPED_CLASSIFIER, key=key, text=f"direct {direct_nonce}"))
            assert classifier_targets(rig.received(direct_nonce)) == (GRANTED_TARGET,)
            for _ in range(3):
                body, request_nonce = classify(rig, key)
                assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
                assert classifier_targets(rig.received(request_nonce)) == (GRANTED_TARGET,)

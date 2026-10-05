import math
import socket
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows, write_rows
from integration._support.process import (
    UpstreamSlot,
    group_members,
    owned_proxy,
    owned_proxy_process,
    owned_upstream,
)
from integration._support.upstream import (
    InteractionState,
    clear_interaction_state,
    register_scenario,
    set_interaction_state,
)
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse
from pydantic import JsonValue

from litellm.proxy.spend_tracking.budget_reservation import DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK

pytestmark: Final = pytest.mark.timeout(900)

_MODEL: Final = "gemini/gemini-3.8-flash"
_INPUT_TOKENS: Final = 300
_OUTPUT_TOKENS: Final = 41
_USAGE: Final[dict[str, JsonValue]] = {
    "total_input_tokens": _INPUT_TOKENS,
    "total_output_tokens": _OUTPUT_TOKENS,
    "total_tool_use_tokens": 0,
    "total_reasoning_tokens": 0,
}
_CUSTOM_INPUT_RATE: Final = 2e-06
_CUSTOM_OUTPUT_RATE: Final = 4e-05
_ENV_KEY: Final = "integration-gemini-env-key"
_DEPLOYMENT_KEY: Final = "integration-gemini-deployment-key"
_CREATOR_POLL: Final = {"BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS": "300"}
_SETTLER_POLL: Final = {
    "BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS": "1",
    "BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS": "1",
    "BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS": "8",
}
_RESUMER_POLL: Final = {
    "BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS": "1",
    "BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS": "1",
    "BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS": "120",
}
_SPEND_QUERY: Final = (
    "SELECT request_id, spend, call_type, status, model, prompt_tokens, completion_tokens "
    'FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
)
_SETTLEMENT_QUERY: Final = (
    "SELECT interaction_id, claimed_by, outcome, claimed_at IS NOT NULL AS claimed, "
    'settled_at IS NOT NULL AS settled, create_context FROM "LiteLLM_BackgroundInteractionSettlement" '
    "WHERE interaction_id = %s"
)
_SETTLEMENT_TABLE_PRESENT_QUERY: Final = "SELECT to_regclass(%s) IS NOT NULL AS present"
_SETTLEMENT_TABLE: Final = '"LiteLLM_BackgroundInteractionSettlement"'
_SETTLEMENT_BY_CALL_QUERY: Final = (
    'SELECT interaction_id FROM "LiteLLM_BackgroundInteractionSettlement" WHERE create_context->>%s = %s'
)
_OUTAGE_RENAME: Final = (
    'ALTER TABLE IF EXISTS "LiteLLM_BackgroundInteractionSettlement" '
    'RENAME TO "LiteLLM_BackgroundInteractionSettlement_outage"'
)
_OUTAGE_RESTORE: Final = (
    'ALTER TABLE IF EXISTS "LiteLLM_BackgroundInteractionSettlement_outage" '
    'RENAME TO "LiteLLM_BackgroundInteractionSettlement"'
)


@dataclass(frozen=True, slots=True)
class Deployments:
    """Config deployments every replica boots with, so no worker ever misses a model added at run time."""

    in_progress: str
    completed_at_once: str
    failing_create: str
    custom_priced: str


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    upstream: UpstreamSlot
    config: Path
    models: Deployments
    creator: Gateway
    settler: Gateway
    settler_pid: int
    directory: Path

    def environment(self, **poll: str) -> dict[str, str]:
        return {"GEMINI_API_BASE": self.upstream.url, "GEMINI_API_KEY": _ENV_KEY, **poll}


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("settlement")
    with gateway_from_environment() as gateway, owned_upstream(directory) as upstream:
        models: Final = _register_deployments(upstream.url)
        config: Final = _write_config(directory, upstream.url, models)
        environment: Final = {"GEMINI_API_BASE": upstream.url, "GEMINI_API_KEY": _ENV_KEY}
        with (
            owned_proxy(gateway, directory, {**environment, **_CREATOR_POLL}, config=config, workers=1) as creator,
            owned_proxy_process(
                gateway, directory, {**environment, **_SETTLER_POLL}, config=config, workers=2
            ) as settler,
        ):
            yield Rig(gateway, upstream, config, models, creator, settler.gateway, settler.process.pid, directory)


def _register_deployments(upstream_url: str) -> Deployments:
    suffix: Final = uuid.uuid4().hex[:8]
    models: Final = Deployments(
        in_progress=f"settle-in-progress-{suffix}",
        completed_at_once=f"settle-completed-at-once-{suffix}",
        failing_create=f"settle-failing-create-{suffix}",
        custom_priced=f"settle-custom-priced-{suffix}",
    )
    _register_scenarios(upstream_url, models)
    return models


def _register_scenarios(upstream_url: str, models: Deployments) -> None:
    scripted: Final = {
        models.in_progress: _interaction("in_progress", None),
        models.completed_at_once: _interaction("completed", _USAGE),
        models.failing_create: JsonResponse(
            content_type="application/json", body={"error": {"message": "boom"}}, status=500
        ),
        models.custom_priced: _interaction("in_progress", None),
    }
    for name, response in scripted.items():
        register_scenario(
            name,
            RoutedResponse(content_type="application/x-routed", routes={"POST /v1beta/interactions": response}),
            control_url=upstream_url,
        )


def _write_config(directory: Path, upstream_url: str, models: Deployments) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    custom_pricing: Final = {"input_cost_per_token": _CUSTOM_INPUT_RATE, "output_cost_per_token": _CUSTOM_OUTPUT_RATE}
    model_list: Final = [
        {
            "model_name": name,
            "litellm_params": {
                "model": _MODEL,
                "api_base": f"{upstream_url}/{name}",
                "api_key": _DEPLOYMENT_KEY,
                **(custom_pricing if name == models.custom_priced else {}),
            },
        }
        for name in (models.in_progress, models.completed_at_once, models.failing_create, models.custom_priced)
    ]
    path: Final = directory / "settlement_config.yaml"
    path.write_text(yaml.safe_dump({**base, "model_list": model_list}))
    return path


def _interaction(status: str, usage: dict[str, JsonValue] | None, http_status: int = 200) -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": "$UNIQUE_ID",
            "object": "interaction",
            "model": "gemini-3.8-flash",
            "status": status,
            "steps": [],
            "usage": usage,
        },
        status=http_status,
    )


def _completed() -> InteractionState:
    return InteractionState(status="completed", usage=_USAGE)


def _create(
    replica: Gateway,
    model: str,
    key: str,
    *,
    path: str = "/v1beta/interactions",
    background: bool = True,
    text: str | None = None,
) -> str:
    response: Final = replica.request(
        "POST",
        path,
        {"model": model, "input": text or f"settle {uuid.uuid4().hex}", "background": background},
        key=key,
    )
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def _state(rig: Rig, interaction_id: str, state: InteractionState) -> None:
    set_interaction_state(rig.upstream.url, interaction_id, state)


def _delete(replica: Gateway, interaction_id: str, key: str, *, path: str = "/v1beta/interactions") -> httpx.Response:
    return replica.request("DELETE", f"{path}/{interaction_id}", key=key)


def _delete_ok(replica: Gateway, interaction_id: str, key: str) -> None:
    deleted: Final = _delete(replica, interaction_id, key)
    assert deleted.status_code == 200, deleted.text


def _delete_concurrently(replica: Gateway, interaction_ids: Sequence[str], key: str) -> tuple[int, ...]:
    def status(interaction_id: str) -> int:
        return _delete(replica, interaction_id, key).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        return tuple(pool.map(status, interaction_ids))


def _assert_unclaimed(interaction_id: str) -> None:
    row: Final = _settlement(interaction_id)
    assert row is not None and row["claimed"] is False and row["outcome"] is None, row


def _spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(_SPEND_QUERY, (request_id,))


def _settlement(interaction_id: str) -> dict[str, JsonValue] | None:
    rows: Final = read_rows(_SETTLEMENT_QUERY, (interaction_id,))
    return rows[0] if rows else None


def _settlement_table_present() -> bool:
    return read_rows(_SETTLEMENT_TABLE_PRESENT_QUERY, (_SETTLEMENT_TABLE,))[0]["present"] is True


def _settlement_if_stored(interaction_id: str) -> dict[str, JsonValue] | None:
    return _settlement(interaction_id) if _settlement_table_present() else None


def _settlements_by_call_if_stored(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(_SETTLEMENT_BY_CALL_QUERY, ("litellm_call_id", call_id)) if _settlement_table_present() else []


def _await_spend_row(interaction_id: str, seconds: float = 30) -> dict[str, JsonValue]:
    return eventually(lambda: _spend_rows(interaction_id), lambda rows: len(rows) == 1, seconds=seconds)[0]


def _await_outcome(interaction_id: str, outcome: str, seconds: float = 30) -> dict[str, JsonValue]:
    row: Final = eventually(
        lambda: _settlement(interaction_id),
        lambda value: value is not None and value["outcome"] == outcome,
        seconds=seconds,
    )
    assert row is not None
    return row


def _model_info(replica: Gateway, model: str) -> Mapping[str, JsonValue]:
    entries: Final = replica.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    return object_value(
        next(object_value(entry)["model_info"] for entry in entries if object_value(entry)["model_name"] == model)
    )


def _rates(replica: Gateway, model: str) -> tuple[float, float]:
    info: Final = _model_info(replica, model)
    input_rate: Final = info["input_cost_per_token"]
    output_rate: Final = info["output_cost_per_token"]
    assert isinstance(input_rate, float) and isinstance(output_rate, float), info
    return input_rate, output_rate


def _reservation_pin(replica: Gateway, model: str) -> float:
    """What one background create estimates before its usage is known: the output tokens the estimator assumes,
    at the deployment's output rate, with the prompt's few input tokens left as slack. A key budget below that
    is filled by the first create's reservation, so the next create is refused until a settlement releases it."""
    info: Final = _model_info(replica, model)
    max_output: Final = info["max_output_tokens"]
    output_rate: Final = info["output_cost_per_token"]
    assert isinstance(max_output, int) and isinstance(output_rate, float), info
    return min(max_output, DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK) * output_rate


def _assert_billed(row: Mapping[str, JsonValue], rates: tuple[float, float]) -> float:
    expected: Final = _INPUT_TOKENS * rates[0] + _OUTPUT_TOKENS * rates[1]
    spend: Final = row["spend"]
    assert isinstance(spend, float) and math.isclose(spend, expected, rel_tol=1e-9), (row, expected)
    assert row["call_type"] == "acreate_interaction", row
    assert row["status"] == "success", row
    assert row["prompt_tokens"] == _INPUT_TOKENS and row["completion_tokens"] == _OUTPUT_TOKENS, row
    return spend


def _key_spend(replica: Gateway, key: str) -> float:
    spend: Final = object_value(replica.get("/key/info", {"key": key})["info"])["spend"]
    assert isinstance(spend, float | int), spend
    return float(spend)


def _await_key_spend(replica: Gateway, key: str, expected: float) -> None:
    eventually(lambda: _key_spend(replica, key), lambda spend: math.isclose(spend, expected, rel_tol=1e-9), seconds=30)


def _drain(rig: Rig) -> list[JsonValue]:
    observed: Final = httpx.get(f"{rig.upstream.url}/__observations", trust_env=False, timeout=15)
    observed.raise_for_status()
    requests: Final = JSON_OBJECT.validate_json(observed.content)["requests"]
    assert isinstance(requests, list), requests
    return requests


def _calls(rig: Rig, interaction_id: str) -> tuple[tuple[str, str], ...]:
    suffix: Final = f"/v1beta/interactions/{interaction_id}"
    return tuple(
        (string_value(object_value(entry)["method"]), string_value(object_value(entry)["api_key"]))
        for entry in _drain(rig)
        if string_value(object_value(entry)["path"]).endswith(suffix)
    )


def _claimer_pid(row: Mapping[str, JsonValue]) -> int:
    claimed_by: Final = string_value(row["claimed_by"])
    host, _, pid = claimed_by.rpartition(":")
    assert host == socket.gethostname(), claimed_by
    return int(pid)


def _booted_after(pid: int, moment: float) -> bool:
    try:
        return psutil.Process(pid).create_time() > moment
    except psutil.NoSuchProcess:
        return False


def _worker_pids(root_pid: int) -> frozenset[int]:
    return frozenset(
        process.pid for process in group_members(root_pid) if process.pid != root_pid and _is_spawned_worker(process)
    )


def _is_spawned_worker(process: psutil.Process) -> bool:
    try:
        return process.name().lower().startswith("python") and "resource_tracker" not in " ".join(process.cmdline())
    except psutil.Error:
        return False


def _readiness(replica: Gateway) -> int:
    try:
        return replica.request("GET", "/health/readiness").status_code
    except httpx.TransportError:
        return 0


def test_creator_poll_bills_a_completed_background_interaction_once(rig: Rig) -> None:
    with rig.settler.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.settler, model, key)
        _state(rig, created, _completed())
        spend: Final = _assert_billed(_await_spend_row(created), _rates(rig.settler, model))
        _await_key_spend(rig.settler, key, spend)
        assert len(_spend_rows(created)) == 1


def test_creator_poll_records_its_settlement_durably(rig: Rig) -> None:
    with rig.settler.scenario() as scenario:
        model: Final = rig.models.in_progress
        created: Final = _create(rig.settler, model, scenario.key())
        _state(rig, created, _completed())
        _await_spend_row(created)
        row: Final = _await_outcome(created, "billed")
        assert row["claimed"] is True and row["settled"] is True, row
        assert row["create_context"] == {}, row
        _claimer_pid(row)


@pytest.mark.parametrize("path", ["/v1beta/interactions", "/interactions"])
def test_delete_on_another_replica_bills_the_creators_interaction_once(rig: Rig, path: str) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key, path=path)
        _state(rig, created, _completed())
        _drain(rig)
        deleted: Final = _delete(rig.settler, created, key, path=path)
        assert deleted.status_code == 200, deleted.text
        spend: Final = _assert_billed(_await_spend_row(created), _rates(rig.creator, model))
        row: Final = _await_outcome(created, "billed")
        assert _claimer_pid(row) in _worker_pids(rig.settler_pid), row
        assert _calls(rig, created) == (("GET", _ENV_KEY), ("DELETE", _ENV_KEY))
        _await_key_spend(rig.creator, key, spend)
        assert len(_spend_rows(created)) == 1


def test_delete_of_a_failed_interaction_releases_without_a_spend_row(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key)
        _state(rig, created, InteractionState(status="failed", usage=None))
        deleted: Final = _delete(rig.settler, created, key)
        assert deleted.status_code == 200, deleted.text
        _await_outcome(created, "released")
        assert _spend_rows(created) == []
        assert _key_spend(rig.creator, key) == 0


def test_delete_of_a_requires_action_interaction_bills_its_usage(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key)
        _state(rig, created, InteractionState(status="requires_action", usage=_USAGE))
        deleted: Final = _delete(rig.settler, created, key)
        assert deleted.status_code == 200, deleted.text
        _assert_billed(_await_spend_row(created), _rates(rig.creator, model))
        _await_outcome(created, "billed")


def test_a_replica_booting_later_resumes_and_bills_unclaimed_interactions(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created_after: Final = time.time()
        created: Final = tuple(_create(rig.creator, model, key) for _ in range(3))
        for item in created:
            _state(rig, item, _completed())
        rates: Final = _rates(rig.creator, model)
        with owned_proxy_process(
            rig.gateway, rig.directory, rig.environment(**_RESUMER_POLL), config=rig.config, workers=2
        ) as resumer:
            pids: Final = _worker_pids(resumer.process.pid)
            assert len(pids) == 2, pids
            for item in created:
                _assert_billed(_await_spend_row(item, seconds=90), rates)
                claimer: Final = _claimer_pid(_await_outcome(item, "billed"))
                assert claimer in pids or _booted_after(claimer, created_after), (claimer, pids)
        for item in created:
            assert len(_spend_rows(item)) == 1


def test_deletes_on_the_creating_proxy_bill_each_interaction_once(rig: Rig) -> None:
    with rig.settler.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = tuple(_create(rig.settler, model, key) for _ in range(8))
        for item in created:
            _state(rig, item, _completed())
        assert _delete_concurrently(rig.settler, created, key) == (200,) * 8
        rates: Final = _rates(rig.settler, model)
        for item in created:
            _assert_billed(_await_spend_row(item), rates)
            _await_outcome(item, "billed")
        _await_key_spend(rig.settler, key, 8 * (_INPUT_TOKENS * rates[0] + _OUTPUT_TOKENS * rates[1]))
        for item in created:
            assert len(_spend_rows(item)) == 1


def test_custom_deployment_pricing_bills_at_the_deployment_rate_on_another_replica(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.custom_priced
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key)
        _state(rig, created, _completed())
        deleted: Final = _delete(rig.settler, created, key)
        assert deleted.status_code == 200, deleted.text
        _assert_billed(_await_spend_row(created), (_CUSTOM_INPUT_RATE, _CUSTOM_OUTPUT_RATE))
        _await_outcome(created, "billed")


def test_cancel_then_delete_on_another_replica_releases_without_a_spend_row(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key)
        _state(rig, created, InteractionState(status="in_progress"))
        cancelled: Final = rig.settler.request("POST", f"/v1beta/interactions/{created}/cancel", {}, key=key)
        assert cancelled.status_code == 200, cancelled.text
        before_delete: Final = _settlement(created)
        assert before_delete is not None and before_delete["claimed"] is False, before_delete
        deleted: Final = _delete(rig.settler, created, key)
        assert deleted.status_code == 200, deleted.text
        _await_outcome(created, "released")
        assert _spend_rows(created) == []


def test_delete_fails_closed_when_the_settling_replica_cannot_fetch(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key)
        _state(rig, created, InteractionState(status="completed", usage=_USAGE, get_status=500))
        _drain(rig)
        refused: Final = _delete(rig.settler, created, key)
        assert refused.status_code >= 500, refused.text
        assert "Scripted interaction fetch failure" in refused.text, refused.text
        assert _calls(rig, created) == (("GET", _ENV_KEY),)
        _assert_unclaimed(created)
        assert _spend_rows(created) == []
        _state(rig, created, _completed())
        deleted: Final = _delete(rig.settler, created, key)
        assert deleted.status_code == 200, deleted.text
        _assert_billed(_await_spend_row(created), _rates(rig.creator, model))
        _await_outcome(created, "billed")


def test_delete_of_an_interaction_the_vendor_purged_sends_no_delete_and_keeps_the_row(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key)
        clear_interaction_state(rig.upstream.url, created)
        _drain(rig)
        deleted: Final = _delete(rig.settler, created, key)
        assert deleted.status_code == 404, deleted.text
        assert _calls(rig, created) == (("GET", _ENV_KEY),)
        row: Final = _settlement(created)
        assert row is not None and row["claimed"] is False, row
        assert _spend_rows(created) == []


def test_reading_an_interaction_never_bills_it(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key)
        _state(rig, created, InteractionState(status="in_progress"))
        read_ids: Final = tuple(str(uuid.uuid4()) for _ in range(2))
        first: Final = rig.settler.request(
            "GET", f"/v1beta/interactions/{created}", key=key, headers={"x-litellm-call-id": read_ids[0]}
        )
        assert first.status_code == 200 and JSON_OBJECT.validate_json(first.content)["status"] == "in_progress", (
            first.text
        )
        _state(rig, created, _completed())
        second: Final = rig.settler.request(
            "GET", f"/v1beta/interactions/{created}", key=key, headers={"x-litellm-call-id": read_ids[1]}
        )
        assert second.status_code == 200 and JSON_OBJECT.validate_json(second.content)["usage"] == _USAGE, second.text
        assert _key_spend(rig.creator, key) == 0
        assert _spend_rows(created) == []
        deleted: Final = _delete(rig.settler, created, key)
        assert deleted.status_code == 200, deleted.text
        spend: Final = _assert_billed(_await_spend_row(created), _rates(rig.creator, model))
        _await_key_spend(rig.creator, key, spend)
        for read_id in read_ids:
            assert all(row["spend"] == 0 for row in _spend_rows(read_id)), _spend_rows(read_id)


@pytest.mark.parametrize(
    "interaction_id",
    [f"missing-{uuid.uuid4().hex}", "x" * 5000, "a.b:c", "%2F..%2Fup"],
    ids=["unknown", "five-kilobytes", "punctuation", "encoded-traversal"],
)
def test_delete_of_an_odd_or_unknown_id_is_refused_and_the_proxy_keeps_serving(rig: Rig, interaction_id: str) -> None:
    with rig.settler.scenario() as scenario:
        key: Final = scenario.key()
        deleted: Final = rig.settler.request("DELETE", f"/v1beta/interactions/{interaction_id}", key=key)
        assert 400 <= deleted.status_code < 500, deleted.text
        assert _readiness(rig.settler) == 200
        assert _key_spend(rig.settler, key) == 0


def test_a_missing_settlement_table_leaves_in_process_billing_intact(rig: Rig) -> None:
    write_rows(_OUTAGE_RENAME, ())
    try:
        with rig.settler.scenario() as scenario:
            model: Final = rig.models.in_progress
            key: Final = scenario.key()
            created: Final = _create(rig.settler, model, key)
            _state(rig, created, _completed())
            spend: Final = _assert_billed(_await_spend_row(created), _rates(rig.settler, model))
            _await_key_spend(rig.settler, key, spend)
            deleted: Final = _delete(rig.creator, created, key)
            assert deleted.status_code == 200, deleted.text
            assert len(_spend_rows(created)) == 1
    finally:
        write_rows(_OUTAGE_RESTORE, ())


def test_a_failed_create_registers_nothing(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.failing_create
        key: Final = scenario.key()
        call_id: Final = str(uuid.uuid4())
        response: Final = rig.creator.request(
            "POST",
            "/v1beta/interactions",
            {"model": model, "input": f"settle {uuid.uuid4().hex}", "background": True},
            key=key,
            headers={"x-litellm-call-id": call_id},
        )
        assert response.status_code >= 500, response.text
        assert _key_spend(rig.creator, key) == 0
        assert all(row["spend"] == 0 for row in _spend_rows(call_id)), _spend_rows(call_id)
        assert _settlements_by_call_if_stored(call_id) == []


def test_polling_disabled_replica_registers_nothing_and_never_bills(rig: Rig) -> None:
    disabled: Final = rig.environment(BACKGROUND_INTERACTION_COST_POLLING_ENABLED="false")
    with (
        owned_proxy(rig.gateway, rig.directory, disabled, config=rig.config, workers=1) as quiet,
        quiet.scenario() as scenario,
    ):
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(quiet, model, key)
        _state(rig, created, _completed())
        deleted: Final = _delete(rig.settler, created, key)
        assert deleted.status_code == 200, deleted.text
        assert _settlement_if_stored(created) is None
        assert _key_spend(quiet, key) == 0
        assert _spend_rows(created) == []


@pytest.mark.parametrize("background", [False, True], ids=["synchronous", "background"])
def test_a_create_that_completes_at_once_is_billed_by_the_create_alone(rig: Rig, background: bool) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.completed_at_once
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key, background=background)
        spend: Final = _assert_billed(_await_spend_row(created), _rates(rig.creator, model))
        _await_key_spend(rig.creator, key, spend)
        assert _settlement_if_stored(created) is None
        _state(rig, created, _completed())
        deleted: Final = _delete(rig.settler, created, key)
        assert deleted.status_code == 200, deleted.text
        _await_key_spend(rig.creator, key, spend)
        assert len(_spend_rows(created)) == 1


def test_identical_creates_settle_as_separate_interactions(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        text: Final = f"settle {uuid.uuid4().hex}"
        created: Final = tuple(_create(rig.creator, model, key, text=text) for _ in range(3))
        assert len({item for item in created}) == 3, created
        for item in created:
            _state(rig, item, _completed())
            _delete_ok(rig.settler, item, key)
        rates: Final = _rates(rig.creator, model)
        for item in created:
            _assert_billed(_await_spend_row(item), rates)
            _await_outcome(item, "billed")
        _await_key_spend(rig.creator, key, 3 * (_INPUT_TOKENS * rates[0] + _OUTPUT_TOKENS * rates[1]))


def test_settlement_on_another_replica_releases_the_creators_budget_reservation(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key(max_budget=0.5 * _reservation_pin(rig.creator, model))
        janitor: Final = scenario.key()
        first: Final = _create(rig.creator, model, key)
        pinned: Final = rig.creator.request(
            "POST", "/v1beta/interactions", {"model": model, "input": "settle pinned", "background": True}, key=key
        )
        assert pinned.status_code == 422 and pinned.json()["error"]["type"] == "budget_exceeded", pinned.text
        _state(rig, first, _completed())
        still_pinned: Final = _delete(rig.settler, first, key)
        assert still_pinned.status_code == 422 and still_pinned.json()["error"]["type"] == "budget_exceeded", (
            still_pinned.text
        )
        deleted: Final = _delete(rig.settler, first, janitor)
        assert deleted.status_code == 200, deleted.text
        spend: Final = _assert_billed(_await_spend_row(first), _rates(rig.creator, model))
        _await_key_spend(rig.creator, key, spend)
        released: Final = eventually(
            lambda: (
                rig.creator.request(
                    "POST",
                    "/v1beta/interactions",
                    {"model": model, "input": "settle released", "background": True},
                    key=key,
                ).status_code
            ),
            lambda status: status == 200,
            seconds=20,
            return_last_on_timeout=True,
        )
        assert released == 200


def test_a_poll_that_never_sees_a_terminal_status_records_unsettled_and_releases(rig: Rig) -> None:
    with rig.settler.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.settler, model, key)
        _state(rig, created, InteractionState(status="in_progress"))
        row: Final = _await_outcome(created, "unsettled", seconds=40)
        assert row["create_context"] == {}, row
        assert _spend_rows(created) == []
        assert _key_spend(rig.settler, key) == 0
        deleted: Final = _delete(rig.creator, created, key)
        assert deleted.status_code == 200, deleted.text
        assert _spend_rows(created) == []


def test_an_upstream_outage_fails_deletes_closed_and_every_interaction_bills_once_after_recovery(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = tuple(_create(rig.creator, model, key) for _ in range(16))
        rates: Final = _rates(rig.creator, model)
        rig.upstream.stop()
        try:
            refused: Final = _delete_concurrently(rig.settler, created, key)
            assert all(status >= 500 for status in refused), refused
            for item in created:
                _assert_unclaimed(item)
            assert _readiness(rig.creator) == 200 and _readiness(rig.settler) == 200
        finally:
            rig.upstream.start()
            _register_scenarios(rig.upstream.url, rig.models)
        for item in created:
            _state(rig, item, _completed())
        assert _delete_concurrently(rig.settler, created, key) == (200,) * 16
        for item in created:
            _assert_billed(_await_spend_row(item), rates)
            _await_outcome(item, "billed")
        _await_key_spend(rig.creator, key, 16 * (_INPUT_TOKENS * rates[0] + _OUTPUT_TOKENS * rates[1]))
        for item in created:
            assert len(_spend_rows(item)) == 1


def test_killed_workers_leave_their_polls_to_the_respawned_workers(rig: Rig) -> None:
    with (
        owned_proxy_process(
            rig.gateway, rig.directory, rig.environment(**_RESUMER_POLL), config=rig.config, workers=2
        ) as resumer,
        resumer.gateway.scenario() as scenario,
    ):
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = tuple(_create(resumer.gateway, model, key) for _ in range(16))
        for item in created:
            _state(rig, item, InteractionState(status="in_progress"))
        rates: Final = _rates(resumer.gateway, model)
        killed: Final = _worker_pids(resumer.process.pid)
        assert len(killed) == 2, killed
        victims: Final = tuple(psutil.Process(pid) for pid in killed)
        for victim in victims:
            victim.kill()
        psutil.wait_procs(victims, timeout=15)
        for item in created:
            _state(rig, item, _completed())
        for item in created:
            _assert_billed(_await_spend_row(item, seconds=150), rates)
            assert _claimer_pid(_await_outcome(item, "billed")) not in killed
        assert eventually(lambda: _readiness(resumer.gateway), lambda status: status == 200, seconds=60) == 200
        for item in created:
            assert len(_spend_rows(item)) == 1


def test_concurrent_deletes_on_a_slow_upstream_settle_exactly_once(rig: Rig) -> None:
    with rig.creator.scenario() as scenario:
        model: Final = rig.models.in_progress
        key: Final = scenario.key()
        created: Final = _create(rig.creator, model, key)
        _state(rig, created, InteractionState(status="completed", usage=_USAGE, delay_seconds=1.5))
        statuses: Final = _delete_concurrently(rig.settler, (created, created), key)
        assert sorted(statuses) == [200, 404], statuses
        spend: Final = _assert_billed(_await_spend_row(created), _rates(rig.creator, model))
        _await_outcome(created, "billed")
        _await_key_spend(rig.creator, key, spend)
        assert len(_spend_rows(created)) == 1

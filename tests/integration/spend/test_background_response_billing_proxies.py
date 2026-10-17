import itertools
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, held, object_value, string_value
from integration._support.database import read_rows, scratch_database, write_rows
from integration._support.otlp_sink import Span, configure_sink, owned_sinks, recorded_spans
from integration._support.process import graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.prometheus_series import Sample, scrape
from integration._support.timing import assert_not_delayed_by, timed
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import RoutedResponse, SseResponse
from integration.spend._background_responses import (
    INPUT_COST_PER_TOKEN,
    MANAGED_READ_MARKER,
    ONE_POLL_SECONDS,
    OUTPUT_COST_PER_TOKEN,
    PRICE,
    PROMPT,
    PROVIDER_ID,
    READS,
    Read,
    assert_charged_exactly,
    background_routes,
    costs,
    create_background_response,
    free_reads,
    hashed,
    json_reply,
    key_spend,
    managed_rows,
    number,
    read_response,
    reported_cost,
    response_body,
    scripted_routes,
    settling_reads,
    streamed_reply,
)
from pydantic import JsonValue, TypeAdapter

CONFIGURATION: Final = TypeAdapter(dict[str, JsonValue])
OWNED_PROXY_SECONDS: Final = 2 * graceful_stop_seconds() + 240
NO_DATABASE: Final = ("DATABASE_URL", "DATABASE_URL_READ_REPLICA")
POLL_OFF: Final = {"PROXY_BATCH_POLLING_ENABLED": "false"}
SINK_DELAY_SECONDS: Final = 4.0
SPEND_RETRY_SECONDS: Final = 60
BURST_RESPONSES: Final = 6
RESTART_RESPONSES: Final = 3
COOLDOWN_SECONDS: Final = 300
CHAT_COMPLETION: Final[dict[str, JsonValue]] = {
    "id": "chatcmpl-$REQUEST_ID",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


@contextmanager
def _scripted(response: RoutedResponse | SseResponse, count: int = 1) -> Iterator[tuple[ScenarioHandle, ...]]:
    with ExitStack() as stack:
        handles: Final = tuple(register_scenario(f"bgonce-{uuid.uuid4().hex[:12]}", response) for _ in range(count))
        for handle in handles:
            stack.callback(delete_scenario, handle)
        yield handles


def _rescript(handles: Sequence[ScenarioHandle], response: RoutedResponse) -> None:
    for handle in handles:
        register_scenario(handle.scenario_id, response, control_url=handle.control_url)


def _with_chat(response: RoutedResponse) -> RoutedResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={**response.routes, "POST /chat/completions": json_reply(CHAT_COMPLETION)},
    )


def _deployment(handle: ScenarioHandle, model_info: Mapping[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    return {
        "model_name": handle.scenario_id,
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": "sk-scripted-provider",
            "api_base": handle.api_base(),
            "input_cost_per_token": INPUT_COST_PER_TOKEN,
            "output_cost_per_token": OUTPUT_COST_PER_TOKEN,
        },
        **({"model_info": dict(model_info)} if model_info else {}),
    }


def _write_config(directory: Path, config: Mapping[str, JsonValue]) -> Path:
    path: Final = directory / f"background-billing-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(dict(config)))
    return path


def _config(
    directory: Path,
    handles: Sequence[ScenarioHandle],
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    **sections: Mapping[str, JsonValue],
) -> Path:
    shared: Final = CONFIGURATION.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    settings: Final = {
        name: {**object_value(shared[name]), **sections.get(name, {})}
        for name in ("general_settings", "litellm_settings", "router_settings")
    }
    extra: Final = {name: dict(section) for name, section in sections.items() if name not in settings}
    return _write_config(
        directory,
        {"model_list": [_deployment(handle, model_info) for handle in handles], **settings, **extra},
    )


@contextmanager
def _scratch_proxy(
    gateway: Gateway,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: Path,
    overrides: Mapping[str, str] | None = None,
) -> Iterator[Gateway]:
    with scratch_database() as database_url:
        monkeypatch.setenv("DATABASE_URL", database_url)
        with owned_proxy(
            gateway, directory, {"DATABASE_URL": database_url, **(overrides or {})}, config=config
        ) as proxy:
            yield proxy


def _team_member_key(scenario: Scenario) -> str:
    team_id: Final = scenario.team()
    return scenario.key(team_id=team_id, user_id=scenario.member(team_id))


def _billed_rows_of(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND spend > 0', (hashed(key),)
    )


def _assert_each_billed_once_as_background_cost(key: str, count: int) -> None:
    rows: Final = _billed_rows_of(key)
    assert len(rows) == count, rows
    assert all(string_value(row["request_id"]).endswith("_background_cost") for row in rows), rows
    assert len({row["request_id"] for row in rows}) == count, rows
    assert [number(row["spend"]) for row in rows] == pytest.approx([PRICE] * count)


@pytest.mark.timeout(OWNED_PROXY_SECONDS)
def test_a_proxy_without_a_database_bills_every_read_of_a_finished_background_response(
    gateway: Gateway, tmp_path: Path
) -> None:
    forged: Final[dict[str, JsonValue]] = {
        "internal_call_origin": "background_response_status_poll",
        MANAGED_READ_MARKER: True,
    }
    with _scripted(background_routes("completed")) as (handle,):
        config: Final = _write_config(
            tmp_path,
            {
                "model_list": [_deployment(handle)],
                "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
            },
        )
        overrides: Final = {"STORE_MODEL_IN_DB": "False", "DISABLE_SCHEMA_UPDATE": "true"}
        with owned_proxy(
            gateway, tmp_path, overrides, config=config, database_setup=(), remove_environment=NO_DATABASE
        ) as proxy:
            created: Final = proxy.post(
                "/v1/responses", {"model": handle.scenario_id, "input": PROMPT, "background": True, "store": False}
            )
            response_id: Final = string_value(created["id"])
            reads: Final = (
                read_response(proxy, proxy.key, response_id),
                read_response(proxy, proxy.key, response_id),
                read_response(proxy, proxy.key, response_id, {"metadata": forged, "litellm_metadata": forged}),
            )
    assert [(read.status, read.text) for read in reads] == [("completed", "pong")] * 3
    assert [read.cost for read in reads] == pytest.approx([PRICE] * 3)


def _span_cost(span: Span) -> float:
    return number(span["attributes"].get("gen_ai.cost.total_cost", 0))


def _spans_of(sink: str, key: str) -> tuple[Span, ...]:
    _, spans = recorded_spans(sink)
    return tuple(span for span in spans if span["attributes"].get("metadata.user_api_key_hash") == hashed(key))


@pytest.mark.timeout(OWNED_PROXY_SECONDS + 120)
def test_with_the_cost_poll_off_the_first_finished_read_bills_the_creator_once_and_tells_the_destination_once(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _scripted(background_routes("queued")) as (handle,), owned_sinks(tmp_path / "sinks") as sinks:
        config: Final = _config(
            tmp_path,
            (handle,),
            litellm_settings={"callbacks": ["otel"]},
            callback_settings={
                "otel": {"exporter": "http/json", "endpoint": sinks.operator, "use_simple_processor": True}
            },
        )
        overrides: Final = {**POLL_OFF, "OTEL_EXPORTER": "http/json", "OTEL_ENDPOINT": sinks.operator}
        with (
            _scratch_proxy(gateway, tmp_path, monkeypatch, config, overrides) as proxy,
            proxy.scenario() as scenario,
        ):
            key: Final = _team_member_key(scenario)
            response_id: Final = create_background_response(scenario, key, handle.scenario_id)
            unfinished: Final = read_response(proxy, key, response_id)
            _rescript((handle,), background_routes("completed"))
            finished: Final = tuple(read_response(proxy, key, response_id) for _ in range(READS))
            assert (unfinished.status, unfinished.cost) == ("queued", 0.0)
            assert finished == settling_reads("completed", READS)
            assert_charged_exactly(lambda: key_spend(key), PRICE)
            _assert_each_billed_once_as_background_cost(key, 1)
            spans: Final = eventually(
                lambda: _spans_of(sinks.operator, key), lambda found: len(found) >= READS + 3, seconds=30
            )
            assert [_span_cost(span) for span in spans if _span_cost(span) > 0] == pytest.approx([PRICE])


@pytest.mark.timeout(OWNED_PROXY_SECONDS + 60)
def test_the_read_that_finishes_a_background_response_does_not_wait_on_a_slow_spend_destination(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _scripted(background_routes("queued")) as (handle,), owned_sinks(tmp_path / "sinks") as sinks:
        config: Final = _config(tmp_path, (handle,), litellm_settings={"callbacks": ["openmeter"]})
        overrides: Final = {
            **POLL_OFF,
            "OPENMETER_API_KEY": "om-scripted-key",
            "OPENMETER_API_ENDPOINT": sinks.tenant,
        }
        with (
            _scratch_proxy(gateway, tmp_path, monkeypatch, config, overrides) as proxy,
            proxy.scenario() as scenario,
        ):
            key: Final = scenario.key()
            response_id: Final = create_background_response(scenario, key, handle.scenario_id)
            configure_sink(sinks.tenant, delay_seconds=SINK_DELAY_SECONDS)
            unfinished: Final = timed(lambda: read_response(proxy, key, response_id))
            _rescript((handle,), background_routes("completed"))
            first_finished: Final = timed(lambda: read_response(proxy, key, response_id))
            later: Final = tuple(timed(lambda: read_response(proxy, key, response_id)) for _ in range(2))
            assert first_finished.value.status == "completed"
            assert_not_delayed_by(
                (first_finished.seconds,),
                control=(unfinished.seconds, *(read.seconds for read in later)),
                injected_delay=SINK_DELAY_SECONDS,
            )


def _provider_404() -> RoutedResponse:
    gone: Final = json_reply(
        {
            "error": {
                "message": f"Response with id '{PROVIDER_ID}' not found.",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
        404,
    )
    return scripted_routes(json_reply(response_body("queued")), gone)


def _create_status(proxy: Gateway, key: str, model: str) -> int:
    return proxy.request(
        "POST", "/v1/responses", {"model": model, "input": PROMPT, "background": True, "store": False}, key=key
    ).status_code


@pytest.mark.timeout(OWNED_PROXY_SECONDS + ONE_POLL_SECONDS)
def test_neither_a_poll_404_nor_a_client_read_404_takes_the_deployment_out_of_service(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ExitStack() as stack:
        (handle,) = stack.enter_context(_scripted(_provider_404()))
        config: Final = _config(
            tmp_path,
            (handle,),
            model_info={"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0, "NotFoundErrorAllowedFails": 0}},
            router_settings={"disable_cooldowns": False, "cooldown_time": COOLDOWN_SECONDS},
        )
        proxy: Final = stack.enter_context(_scratch_proxy(gateway, tmp_path, monkeypatch, config))
        scenario: Final = stack.enter_context(proxy.scenario())
        key: Final = scenario.key()
        polled: Final = create_background_response(scenario, key, handle.scenario_id)
        eventually(
            lambda: [row["status"] for row in managed_rows(polled)],
            lambda statuses: statuses == ["stale_expired"],
            seconds=ONE_POLL_SECONDS,
        )
        after_poll: Final = _create_status(proxy, key, handle.scenario_id)
        client_read: Final = proxy.request("GET", f"/v1/responses/{polled}", key=key).status_code
        after_client_read: Final = _create_status(proxy, key, handle.scenario_id)
    assert (after_poll, client_read, after_client_read) == (200, 404, 200)


def _burst_call(proxy: Gateway, call: tuple[str, str, str]) -> int | None:
    kind, key, target = call
    try:
        if kind == "read":
            return proxy.request("GET", f"/v1/responses/{target}", key=key).status_code
        return proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": target, "messages": [{"role": "user", "content": PROMPT}]},
            key=key,
        ).status_code
    except httpx.TransportError:
        return None


@pytest.mark.timeout(OWNED_PROXY_SECONDS + 2 * graceful_stop_seconds() + 60 + ONE_POLL_SECONDS + 100)
def test_background_responses_read_across_a_killed_worker_are_each_billed_once_to_their_creator(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        _scripted(_with_chat(background_routes("queued")), BURST_RESPONSES) as handles,
        scratch_database() as database_url,
    ):
        monkeypatch.setenv("DATABASE_URL", database_url)
        config: Final = _config(tmp_path, handles)
        with (
            owned_proxy_process(gateway, tmp_path, {"DATABASE_URL": database_url}, config=config, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            key: Final = _team_member_key(scenario)
            chat_key: Final = scenario.key()
            response_ids: Final = tuple(
                create_background_response(scenario, key, handle.scenario_id) for handle in handles
            )
            reads: Final = tuple(("read", key, response_id) for response_id in response_ids * 3)
            chats: Final = tuple(("chat", chat_key, handle.scenario_id) for handle in handles * 2)
            calls: Final = tuple(
                call for call in itertools.chain.from_iterable(itertools.zip_longest(reads, chats)) if call is not None
            )
            with owned.replacing_worker(min(owned.worker_pids())) as kill, ThreadPoolExecutor(max_workers=10) as pool:
                kill.freeze()
                futures: Final = tuple(pool.submit(_burst_call, owned.gateway, call) for call in calls)
                eventually(
                    lambda: sum(future.done() for future in futures),
                    lambda done: done >= len(futures) // 2,
                    seconds=30,
                )
                kill()
            outcomes: Final = tuple(future.result() for future in futures)
            assert set(outcomes) <= {200, None}, outcomes
            assert outcomes.count(200) >= len(outcomes) // 2, outcomes
            _rescript(handles, _with_chat(background_routes("completed")))
            finished: Final = tuple(read_response(owned.gateway, key, response_id) for response_id in response_ids * 2)
            assert [read.status for read in finished] == ["completed"] * 2 * BURST_RESPONSES
            assert [read.cost in (0.0, pytest.approx(PRICE)) for read in finished] == [True] * 2 * BURST_RESPONSES
            assert sum(read.cost or 0.0 for read in finished) <= BURST_RESPONSES * PRICE + 1e-9
            assert_charged_exactly(lambda: key_spend(key), BURST_RESPONSES * PRICE)
            _assert_each_billed_once_as_background_cost(key, BURST_RESPONSES)


def _statuses(response_ids: Sequence[str]) -> list[JsonValue]:
    rows: Final = itertools.chain.from_iterable(managed_rows(response_id) for response_id in response_ids)
    return [row["status"] for row in rows]


def _submit(proxy: Gateway, key: str, model: str) -> str:
    response: Final = proxy.request(
        "POST", "/v1/responses", {"model": model, "input": PROMPT, "background": True, "store": False}, key=key
    )
    assert response.status_code == 200, response.text
    return string_value(object_value(response.json())["id"])


@pytest.mark.timeout(2 * OWNED_PROXY_SECONDS + ONE_POLL_SECONDS + 100)
def test_background_responses_left_pending_across_a_proxy_restart_are_billed_once_to_their_creator(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _scripted(background_routes("queued"), RESTART_RESPONSES) as handles, scratch_database() as database_url:
        monkeypatch.setenv("DATABASE_URL", database_url)
        config: Final = _config(tmp_path, handles)
        with owned_proxy(
            gateway, tmp_path / "before", {"DATABASE_URL": database_url, **POLL_OFF}, config=config
        ) as first:
            key: Final = string_value(first.post("/key/generate", {})["key"])
            response_ids: Final = tuple(_submit(first, key, handle.scenario_id) for handle in handles)
        _rescript(handles, background_routes("completed"))
        with owned_proxy(gateway, tmp_path / "after", {"DATABASE_URL": database_url}, config=config):
            eventually(
                lambda: _statuses(response_ids),
                lambda statuses: statuses == ["completed"] * RESTART_RESPONSES,
                seconds=ONE_POLL_SECONDS,
            )
            assert_charged_exactly(lambda: key_spend(key), RESTART_RESPONSES * PRICE)
            _assert_each_billed_once_as_background_cost(key, RESTART_RESPONSES)


DELETED: Final[dict[str, JsonValue]] = {"id": PROVIDER_ID, "object": "response.deleted", "deleted": True}
REFUSAL_HOLD_SECONDS: Final = 10


def _deletable(read_status: str) -> RoutedResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={**background_routes(read_status).routes, f"DELETE /responses/{PROVIDER_ID}": json_reply(DELETED)},
    )


@pytest.mark.timeout(OWNED_PROXY_SECONDS + 120)
def test_a_delete_bills_a_finished_background_response_once_and_only_for_a_caller_that_may_address_it(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _scripted(_deletable("queued")) as (handle,):
        config: Final = _config(tmp_path, (handle,))
        with (
            _scratch_proxy(gateway, tmp_path, monkeypatch, config, POLL_OFF) as proxy,
            proxy.scenario() as scenario,
        ):
            creator: Final = _team_member_key(scenario)
            stranger: Final = _team_member_key(scenario)
            response_id: Final = create_background_response(scenario, creator, handle.scenario_id)
            _rescript((handle,), _deletable("completed"))
            refused: Final = proxy.request("DELETE", f"/v1/responses/{response_id}", key=stranger)
            assert refused.status_code == 403, refused.text
            held(lambda: key_spend(creator), costs(0.0), holding=REFUSAL_HOLD_SECONDS)
            assert [row["status"] for row in managed_rows(response_id)] == ["queued"]
            deleted: Final = proxy.request("DELETE", f"/v1/responses/{response_id}", key=creator)
            assert deleted.status_code == 200, deleted.text
            assert object_value(deleted.json())["deleted"] is True
            assert reported_cost(deleted.headers) == pytest.approx(PRICE)
            assert_charged_exactly(lambda: key_spend(creator), PRICE)
            _assert_each_billed_once_as_background_cost(creator, 1)
            assert key_spend(stranger) == 0
            assert [row["status"] for row in managed_rows(response_id)] == ["completed"]


@contextmanager
def _billing_claims_refused() -> Iterator[None]:
    write_rows(
        "CREATE FUNCTION audit_refuse_billing_claim() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
        "IF NEW.status = 'completed' AND OLD.status IS DISTINCT FROM 'completed' THEN "
        "RAISE EXCEPTION 'audit refuses the billing claim'; END IF; RETURN NEW; END $$",
        (),
    )
    write_rows(
        'CREATE TRIGGER audit_refuse_billing_claim BEFORE UPDATE ON "LiteLLM_ManagedObjectTable" '
        "FOR EACH ROW EXECUTE FUNCTION audit_refuse_billing_claim()",
        (),
    )
    try:
        yield
    finally:
        write_rows('DROP TRIGGER audit_refuse_billing_claim ON "LiteLLM_ManagedObjectTable"', ())
        write_rows("DROP FUNCTION audit_refuse_billing_claim()", ())


@pytest.mark.timeout(OWNED_PROXY_SECONDS + 180)
def test_a_finished_read_whose_bill_fails_to_settle_bills_the_creator_once_after_the_database_recovers(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _scripted(background_routes("queued")) as (handle,):
        config: Final = _config(tmp_path, (handle,))
        with (
            _scratch_proxy(gateway, tmp_path, monkeypatch, config, POLL_OFF) as proxy,
            proxy.scenario() as scenario,
        ):
            key: Final = _team_member_key(scenario)
            response_id: Final = create_background_response(scenario, key, handle.scenario_id)
            _rescript((handle,), background_routes("completed"))
            with _billing_claims_refused():
                unsettled: Final = read_response(proxy, key, response_id)
                statuses: Final = [row["status"] for row in managed_rows(response_id)]
            assert unsettled == Read("completed", "pong", 0.0)
            assert statuses == ["queued"]
            eventually(lambda: key_spend(key), costs(PRICE), seconds=SPEND_RETRY_SECONDS)
            assert tuple(read_response(proxy, key, response_id) for _ in range(READS)) == free_reads("completed", READS)
            assert_charged_exactly(lambda: key_spend(key), PRICE)
            _assert_each_billed_once_as_background_cost(key, 1)
            assert [row["status"] for row in managed_rows(response_id)] == ["completed"]


REMAINING_KEY_BUDGET: Final = "litellm_remaining_api_key_budget_metric"
KEY_BUDGET: Final = 10.0


def _remaining_key_budget(samples: Sequence[Sample], key: str) -> list[float]:
    return [
        sample.value
        for sample in samples
        if sample.name == REMAINING_KEY_BUDGET and sample.labels.get("hashed_api_key") == hashed(key)
    ]


@pytest.mark.timeout(OWNED_PROXY_SECONDS + ONE_POLL_SECONDS + 60)
def test_a_background_response_the_poll_bills_lowers_its_creating_keys_remaining_budget_by_its_price(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _scripted(background_routes("completed")) as (handle,):
        config: Final = _config(tmp_path, (handle,), litellm_settings={"callbacks": ["prometheus"]})
        with _scratch_proxy(gateway, tmp_path, monkeypatch, config) as proxy, proxy.scenario() as scenario:
            key: Final = scenario.key(max_budget=KEY_BUDGET)
            response_id: Final = create_background_response(scenario, key, handle.scenario_id)
            eventually(
                lambda: [row["status"] for row in managed_rows(response_id)],
                lambda statuses: statuses == ["completed"],
                seconds=ONE_POLL_SECONDS,
            )
            assert_charged_exactly(lambda: key_spend(key), PRICE)
            eventually(
                lambda: _remaining_key_budget(scrape(proxy), key),
                lambda values: values == [pytest.approx(KEY_BUDGET - PRICE)],
                seconds=30,
            )


CACHE_POLLING: Final[dict[str, JsonValue]] = {"responses": {"background_mode": {"polling_via_cache": "all"}}}
CACHE_POLLED_CREATES: Final[Mapping[str, tuple[dict[str, JsonValue], bool]]] = {
    "streamed_with_background_in_extra_body": ({"stream": True, "extra_body": {"background": True}}, True),
    "unstreamed": ({}, False),
}


def _cache_polled_create(proxy: Gateway, key: str, model: str, create: Mapping[str, JsonValue]) -> str:
    created: Final = proxy.request(
        "POST",
        "/v1/responses",
        {"model": model, "input": PROMPT, "background": True, "store": False, **create},
        key=key,
    )
    assert created.status_code == 200, created.text
    return string_value(object_value(created.json())["id"])


@pytest.mark.timeout(OWNED_PROXY_SECONDS + 120)
@pytest.mark.parametrize("create", CACHE_POLLED_CREATES.keys())
def test_a_background_response_polled_through_the_cache_reports_its_alias_and_bills_its_creator_once(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, create: str
) -> None:
    body, provider_sees_background = CACHE_POLLED_CREATES[create]
    with _scripted(streamed_reply(background=provider_sees_background)) as (handle,):
        config: Final = _config(tmp_path, (handle,), litellm_settings=CACHE_POLLING)
        with _scratch_proxy(gateway, tmp_path, monkeypatch, config, POLL_OFF) as proxy, proxy.scenario() as scenario:
            key: Final = _team_member_key(scenario)
            polling_id: Final = _cache_polled_create(proxy, key, handle.scenario_id, body)
            finished: Final = eventually(
                lambda: read_response(proxy, key, polling_id),
                lambda read: read.status == "completed",
                seconds=SPEND_RETRY_SECONDS,
            )
            assert finished.text == "pong"
            polled: Final = object_value(proxy.request("GET", f"/v1/responses/{polling_id}", key=key).json())
            assert polled["model"] == handle.scenario_id
            assert_charged_exactly(lambda: key_spend(key), PRICE)
            assert [read_response(proxy, key, polling_id).text for _ in range(READS)] == ["pong"] * READS
            assert_charged_exactly(lambda: key_spend(key), PRICE)

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
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.otlp_sink import Span, configure_sink, owned_sinks, recorded_spans
from integration._support.process import graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.timing import assert_not_delayed_by, timed
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import RoutedResponse
from integration.spend._background_responses import (
    INPUT_COST_PER_TOKEN,
    MANAGED_READ_MARKER,
    ONE_POLL_SECONDS,
    OUTPUT_COST_PER_TOKEN,
    PRICE,
    PROMPT,
    PROVIDER_ID,
    READS,
    assert_charged_exactly,
    background_routes,
    create_background_response,
    hashed,
    json_reply,
    key_spend,
    managed_rows,
    number,
    read_response,
    response_body,
    scripted_routes,
)
from pydantic import JsonValue, TypeAdapter

CONFIGURATION: Final = TypeAdapter(dict[str, JsonValue])
OWNED_PROXY_SECONDS: Final = 2 * graceful_stop_seconds() + 240
NO_DATABASE: Final = ("DATABASE_URL", "DATABASE_URL_READ_REPLICA")
POLL_OFF: Final = {"PROXY_BATCH_POLLING_ENABLED": "false"}
SINK_DELAY_SECONDS: Final = 4.0
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
def _scripted(response: RoutedResponse, count: int = 1) -> Iterator[tuple[ScenarioHandle, ...]]:
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
            assert [(read.status, read.cost) for read in finished] == [("completed", 0.0)] * READS
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
            assert [(read.status, read.cost) for read in finished] == [("completed", 0.0)] * 2 * BURST_RESPONSES
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

import json
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.spend.test_batch_completion_accounting import batch_input_file, batch_routes
from integration.streaming.test_stream_contracts import text_stream
from pydantic import JsonValue

CACHED_PROMPT_TOKENS: Final = 4


def _config(
    tmp_path: Path,
    spend_logs_metadata_fields: Mapping[str, JsonValue] | None,
    *,
    model_list: tuple[Mapping[str, JsonValue], ...] = (),
    litellm_settings: Mapping[str, JsonValue] | None = None,
    guardrails: tuple[Mapping[str, JsonValue], ...] = (),
) -> Path:
    retention: Final = (
        {} if spend_logs_metadata_fields is None else {"spend_logs_metadata_fields": dict(spend_logs_metadata_fields)}
    )
    config: Final = tmp_path / f"spend-logs-metadata-fields-{uuid4()}.json"
    config.write_text(
        json.dumps(
            {
                "model_list": [dict(model) for model in model_list],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    **retention,
                },
                "litellm_settings": dict(litellm_settings or {}),
                "guardrails": [dict(guardrail) for guardrail in guardrails],
            }
        )
    )
    return config


def _respond(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": CACHED_PROMPT_TOKENS},
                },
            }
        ).encode()
    )


def _stored_row_after_one_chat(
    gateway: Gateway, tmp_path: Path, spend_logs_metadata_fields: Mapping[str, JsonValue]
) -> tuple[dict[str, JsonValue], str]:
    with (
        wire_server(_respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_config(tmp_path, spend_logs_metadata_fields)) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic-openai-key")
        api_key: Final = scenario.key(key_alias=f"metadata-fields-{uuid4()}", models=[model])
        response_id: Final = string_value(isolated.chat(model, key=api_key)["id"])
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT metadata, proxy_server_request, response FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (response_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        return rows[0], sha256(api_key.encode()).hexdigest()


def test_excluded_metadata_fields_are_not_stored_but_still_reach_daily_spend(gateway: Gateway, tmp_path: Path) -> None:
    row, hashed_key = _stored_row_after_one_chat(
        gateway, tmp_path, {"exclude": ["model_map_information", "usage_object"]}
    )

    metadata: Final = object_value(row["metadata"])
    assert "model_map_information" not in metadata, metadata
    assert "usage_object" not in metadata, metadata
    assert {"status", "cold_storage_object_key"} <= set(metadata), metadata
    assert string_value(metadata["user_api_key_alias"]).startswith("metadata-fields-")
    assert row["proxy_server_request"] == {}
    assert row["response"] == {}
    daily: Final = eventually(
        lambda: read_rows(
            'SELECT prompt_tokens, cache_read_input_tokens FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s',
            (hashed_key,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert daily[0]["cache_read_input_tokens"] == CACHED_PROMPT_TOKENS, daily


def test_included_metadata_fields_are_the_only_ones_stored_besides_always_kept(
    gateway: Gateway, tmp_path: Path
) -> None:
    row, _ = _stored_row_after_one_chat(gateway, tmp_path, {"include": ["user_api_key_alias"]})

    assert set(object_value(row["metadata"])) == {"status", "cold_storage_object_key", "user_api_key_alias"}


def test_guardrail_usage_is_tracked_when_guardrail_information_is_not_stored(gateway: Gateway, tmp_path: Path) -> None:
    guardrail_name: Final = f"metadata-fields-guardrail-{uuid4()}"
    with (
        wire_server(lambda _: Reply(body=b'{"action":"NONE"}')) as policy,
        wire_server(_respond) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {},
            config=_config(
                tmp_path,
                {"exclude": ["guardrail_information"]},
                guardrails=(
                    {
                        "guardrail_name": guardrail_name,
                        "litellm_params": {
                            "guardrail": "generic_guardrail_api",
                            "mode": "pre_call",
                            "default_on": True,
                            "api_base": policy.url,
                            "api_key": "synthetic-guardrail-key",
                        },
                    },
                ),
            ),
        ) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic-openai-key")
        response_id: Final = string_value(isolated.chat(model, key=scenario.key(models=[model]))["id"])
        assert len(policy.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert "guardrail_information" not in object_value(rows[0]["metadata"]), rows[0]
        indexed: Final = eventually(
            lambda: read_rows(
                'SELECT guardrail_id FROM "LiteLLM_SpendLogGuardrailIndex" WHERE request_id=%s', (response_id,)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        metrics: Final = eventually(
            lambda: read_rows(
                'SELECT requests_evaluated, passed_count FROM "LiteLLM_DailyGuardrailMetrics" WHERE guardrail_id=%s',
                (string_value(indexed[0]["guardrail_id"]),),
            ),
            lambda values: values == [{"requests_evaluated": 1, "passed_count": 1}],
            seconds=70,
        )
        assert metrics == [{"requests_evaluated": 1, "passed_count": 1}], metrics


EXCLUDED: Final = ("model_map_information", "user_api_key_alias")


@dataclass(frozen=True, slots=True)
class _Isolated:
    proxy: Gateway
    scenario: Scenario
    key: str
    database_url: str | None = None

    @property
    def hashed_key(self) -> str:
        return sha256(self.key.encode()).hexdigest()

    def rows(self, count: int, where: str = "TRUE", seconds: int = 70) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            eventually(
                lambda: read_rows(
                    "SELECT request_id, call_type, status, spend, cache_hit, metadata "
                    f'FROM "LiteLLM_SpendLogs" WHERE api_key=%s AND {where} ORDER BY "startTime"',
                    (self.hashed_key,),
                    database_url=self.database_url,
                ),
                lambda values: len(values) == count,
                seconds=seconds,
            )
        )


@contextmanager
def _isolated(gateway: Gateway, config: Path, tmp_path: Path, database_url: str | None = None) -> Generator[_Isolated]:
    database: Final = {} if database_url is None else {"DATABASE_URL": database_url}
    replica: Final = () if database_url is None else ("DATABASE_URL_READ_REPLICA",)
    with (
        owned_proxy(gateway, tmp_path, database, config=config, remove_environment=replica) as proxy,
        proxy.scenario() as scenario,
    ):
        key: Final = scenario.key(key_alias=f"metadata-fields-{uuid4()}")
        yield _Isolated(proxy, scenario, key, database_url)


def _assert_filtered(row: Mapping[str, JsonValue], *kept: str) -> dict[str, JsonValue]:
    metadata: Final = object_value(row["metadata"])
    assert not set(EXCLUDED) & set(metadata), metadata
    assert {"status", "cold_storage_object_key", *kept} <= set(metadata), metadata
    return metadata


def _anthropic_event(event: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _anthropic_stream(model: JsonValue) -> tuple[bytes, ...]:
    message: Final = {**_anthropic_message(model), "content": [], "stop_reason": None}
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    )
    return tuple(_anthropic_event(event) for event in events)


def _respond_streaming_or_not(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    if request.target.startswith("/v1/messages"):
        return _anthropic(request)
    if JSON_OBJECT.validate_json(request.body).get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=text_stream(f"chatcmpl-{uuid4()}"))
    return _respond(request)


def _stream(proxy: Gateway, key: str, path: str, body: Mapping[str, JsonValue]) -> None:
    with proxy.client.stream("POST", path, json=dict(body), headers={"Authorization": f"Bearer {key}"}) as response:
        assert response.status_code == 200, response.read().decode()
        lines: Final = tuple(response.iter_lines())
    assert any(line.startswith("data:") for line in lines), lines


def _call(proxy: Gateway, key: str, path: str, body: Mapping[str, JsonValue]) -> None:
    response: Final = proxy.request("POST", path, body, key=key)
    assert response.status_code == 200, response.text


def test_every_inference_surface_and_stream_stores_filtered_metadata(gateway: Gateway, tmp_path: Path) -> None:
    with (
        wire_server(_respond_streaming_or_not) as wire,
        _isolated(gateway, _config(tmp_path, {"exclude": list(EXCLUDED)}), tmp_path) as isolated,
    ):
        model: Final = isolated.scenario.model(
            model="deepseek/gpt-4o-mini", api_base=wire.url + "/v1", api_key="synthetic-key"
        )
        anthropic: Final = isolated.scenario.model(
            model="anthropic/claude-sonnet-4-6", api_base=wire.url, api_key="synthetic-key"
        )
        prompt: Final[list[JsonValue]] = [{"role": "user", "content": f"surfaces {uuid4()}"}]
        calls: Final[tuple[Callable[[Gateway, str, str, Mapping[str, JsonValue]], None], ...]] = (
            _stream,
            _call,
            _stream,
            _call,
            _stream,
        )
        requests: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
            ("/v1/chat/completions", {"model": model, "messages": prompt, "stream": True}),
            ("/v1/messages", {"model": anthropic, "max_tokens": 16, "messages": prompt}),
            ("/v1/messages", {"model": anthropic, "max_tokens": 16, "messages": prompt, "stream": True}),
            ("/v1/responses", {"model": model, "input": f"surfaces {uuid4()}"}),
            ("/v1/responses", {"model": model, "input": f"surfaces {uuid4()}", "stream": True}),
        )
        for send, (path, body) in zip(calls, requests, strict=True):
            send(isolated.proxy, isolated.key, path, body)

        rows: Final = isolated.rows(len(requests))
        for row in rows:
            assert row["status"] == "success", row
            _assert_filtered(row, "usage_object")


def test_failed_request_keeps_its_error_and_status_but_drops_excluded_metadata(
    gateway: Gateway, tmp_path: Path
) -> None:
    def fail(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        return Reply(status=500, body=b'{"error":{"message":"scripted upstream failure","type":"server_error"}}')

    with (
        wire_server(fail) as wire,
        _isolated(gateway, _config(tmp_path, {"exclude": list(EXCLUDED)}), tmp_path) as isolated,
    ):
        model: Final = isolated.scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic")
        response: Final = isolated.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "fail"}]},
            key=isolated.key,
        )
        assert response.status_code >= 500, response.text

        row: Final = isolated.rows(1)[0]
        assert row["status"] == "failure", row
        metadata: Final = _assert_filtered(row, "error_information")
        assert metadata["status"] == "failure", metadata
        assert "scripted upstream failure" in json.dumps(metadata["error_information"]), metadata


def test_response_cache_hit_row_is_filtered_and_charged_nothing(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _config(
        tmp_path, {"exclude": list(EXCLUDED)}, litellm_settings={"cache": True, "cache_params": {"type": "local"}}
    )
    with wire_server(_respond) as wire, _isolated(gateway, config, tmp_path) as isolated:
        model: Final = isolated.scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic")
        text: Final = f"cache {uuid4()}"
        first: Final = string_value(isolated.proxy.chat(model, key=isolated.key, text=text)["id"])
        isolated.proxy.chat(model, key=isolated.key, text=text)

        rows: Final = isolated.rows(2)
        assert len([request for request in wire.drain() if request.method == "POST"]) == 1
        paid, hit = sorted(rows, key=lambda row: row["cache_hit"] == "True")
        assert paid["request_id"] == first and float(str(paid["spend"])) > 0, rows
        assert hit["cache_hit"] == "True" and string_value(hit["request_id"]).startswith(first + "_cache_hit"), rows
        assert float(str(hit["spend"])) == 0, rows
        for row in rows:
            _assert_filtered(row, "usage_object")


def test_batch_cost_row_is_filtered_and_charged_once(gateway: Gateway, tmp_path: Path) -> None:
    with _isolated(gateway, _config(tmp_path, {"exclude": list(EXCLUDED)}), tmp_path) as isolated:
        handle: Final = register_scenario(f"metadata-fields-batch-{uuid4().hex[:12]}", batch_routes("gpt-4o-mini"))
        isolated.scenario.cleanups.callback(delete_scenario, handle)
        model: Final = isolated.scenario.model(api_base=handle.api_base())
        uploaded: Final = isolated.proxy.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": model},
            {"file": ("in.jsonl", batch_input_file(model), "application/jsonl")},
            key=isolated.key,
        )
        assert uploaded.status_code == 200, uploaded.text
        batch: Final = isolated.proxy.post(
            "/v1/batches",
            {
                "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": model,
            },
            key=isolated.key,
        )
        batch_path: Final = f"/v1/batches/{string_value(batch['id'])}"
        retrievals: Final = tuple(isolated.proxy.request("GET", batch_path, key=isolated.key) for _ in range(2))
        assert all(r.status_code == 200 and r.json()["status"] == "completed" for r in retrievals), retrievals

        row: Final = isolated.rows(1, "call_type='aretrieve_batch'")[0]
        assert float(str(row["spend"])) > 0, row
        metadata: Final = _assert_filtered(row, "usage_object")
        assert (metadata["batch_successful_requests"], metadata["batch_failed_requests"]) == (2, 3), metadata


def _update_retention(proxy: Gateway, value: JsonValue) -> httpx.Response:
    return proxy.request(
        "POST",
        "/config/field/update",
        {"field_name": "spend_logs_metadata_fields", "field_value": value, "config_type": "general_settings"},
    )


def test_runtime_retention_update_rejects_invalid_values_and_applies_valid_ones_without_restart(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        scratch_database() as database_url,
        wire_server(_respond) as wire,
        _isolated(gateway, _config(tmp_path, None), tmp_path, database_url) as isolated,
    ):
        model: Final = isolated.scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic")
        invalid: Final[tuple[JsonValue, ...]] = (
            {"include": ["usage_object"], "exclude": ["model_map_information"]},
            {},
            {"exclude": ["not_a_metadata_field"]},
            {"exclude": ["status"]},
        )

        isolated.proxy.chat(model, key=isolated.key)
        assert set(EXCLUDED) <= set(object_value(isolated.rows(1)[0]["metadata"]))
        for value in invalid:
            assert _update_retention(isolated.proxy, value).status_code == 400, value
        isolated.proxy.chat(model, key=isolated.key)
        assert set(EXCLUDED) <= set(object_value(isolated.rows(2)[-1]["metadata"]))

        assert _update_retention(isolated.proxy, {"exclude": list(EXCLUDED)}).status_code == 200
        isolated.proxy.chat(model, key=isolated.key)
        _assert_filtered(isolated.rows(3)[-1])

        for value in invalid:
            assert _update_retention(isolated.proxy, value).status_code == 400, value
        isolated.proxy.chat(model, key=isolated.key)
        _assert_filtered(isolated.rows(4)[-1])


SAVINGS_FIELDS: Final = ("autorouter_savings_estimate", "autorouter_savings")


def _anthropic_message(model: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        },
    }


def _anthropic(request: Request) -> Reply:
    if request.target.startswith("/v1/messages/count_tokens"):
        return Reply(body=b'{"input_tokens":12}')
    body: Final = JSON_OBJECT.validate_json(request.body)
    if body.get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=_anthropic_stream(body["model"]))
    return Reply(body=json.dumps(_anthropic_message(body["model"])).encode())


def _router_models(wire: Wire, router: str) -> tuple[Mapping[str, JsonValue], ...]:
    tiers: Final = {"cheap": "anthropic/claude-sonnet-4-6", "frontier": "anthropic/claude-opus-4-8"}
    return (
        *(
            {"model_name": name, "litellm_params": {"model": model, "api_base": wire.url, "api_key": "synthetic"}}
            for name, model in tiers.items()
        ),
        {
            "model_name": router,
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_config": {
                    "tiers": {"SIMPLE": "cheap", "MEDIUM": "cheap", "COMPLEX": "frontier", "REASONING": "frontier"},
                },
            },
        },
    )


@pytest.mark.parametrize(
    ("retention", "stored"),
    [(None, SAVINGS_FIELDS), ({"exclude": list(SAVINGS_FIELDS)}, ())],
    ids=["unset-keeps-savings", "excluded-savings-stay-out"],
)
@pytest.mark.timeout(240)
def test_delayed_autorouter_savings_publication_respects_retention(
    gateway: Gateway, tmp_path: Path, retention: Mapping[str, JsonValue] | None, stored: tuple[str, ...]
) -> None:
    router: Final = f"router-{uuid4().hex[:8]}"
    with wire_server(_anthropic) as wire:
        config: Final = _config(tmp_path, retention, model_list=_router_models(wire, router))
        with _isolated(gateway, config, tmp_path) as isolated:
            response: Final = isolated.proxy.request(
                "POST",
                "/v1/messages",
                {"model": router, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
                key=isolated.key,
                headers={"x-litellm-session-id": f"session-{uuid4()}"},
            )
            assert response.status_code == 200, response.text

            published: Final = isolated.rows(
                1,
                "metadata::jsonb ? 'routing_decision' AND NOT metadata::jsonb ? 'autorouter_baseline_observation' "
                'AND request_id IN (SELECT request_id FROM "LiteLLM_AutoRouterBaselineObservation" '
                "WHERE publication IS NOT NULL)",
                seconds=150,
            )[0]
            metadata: Final = object_value(published["metadata"])
            assert {name for name in SAVINGS_FIELDS if name in metadata} == set(stored), metadata

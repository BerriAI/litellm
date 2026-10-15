import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.upstream import JsonResponse, SseResponse, delete_scenario, register_scenario
from pydantic import BaseModel, JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _AnthropicTextBlock(BaseModel):
    type: Literal["text"]
    text: str


class _AnthropicMessageUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class _AnthropicMessageResponse(BaseModel):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[_AnthropicTextBlock]
    stop_reason: str | None
    stop_sequence: str | None
    usage: _AnthropicMessageUsage


@pytest.mark.covers("spend.end_user.charged_when_key_has_no_user_id_and_auth_cache_is_redis")
def test_end_user_spend_lands_for_key_without_user_id_when_auth_cache_is_redis(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"end user spend {end_user}"}], "user": end_user},
            key=key,
        )
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["total_tokens"] == 40, response.text
        charged: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id=%s', (end_user,)),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(charged[0]["spend"]) == pytest.approx(20 * 0.001 + 20 * 0.002)


def _observed(upstream: httpx.Client) -> list[dict[str, JsonValue]]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = response.json()["requests"]
    assert isinstance(requests, list)
    return requests


def _observation(path: str, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "path": path,
        "authorization": "Bearer integration-provider-key",
        "body": dict(body),
        "method": "POST",
        "api_key": "",
    }


def _new_customer(gateway: Gateway, user_id: str, **fields: JsonValue) -> None:
    response: Final = gateway.request("POST", "/customer/new", {"user_id": user_id, **fields})
    assert response.status_code == 200, response.text


def _charged_spend(user_id: str) -> float:
    charged: Final = eventually(
        lambda: read_rows('SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id=%s', (user_id,)),
        lambda values: len(values) == 1 and float(str(values[0]["spend"])) >= 0.05,
        seconds=70,
    )
    return float(str(charged[0]["spend"]))


def _assert_daily_row(user_id: str, model: str, upstream_model: str = "openai/gpt-4o-mini") -> None:
    daily: Final = eventually(
        lambda: read_rows(
            'SELECT model, model_group, spend, api_requests FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id=%s',
            (user_id,),
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    assert daily[0]["model"] == upstream_model
    assert daily[0]["model_group"] == model
    assert float(str(daily[0]["spend"])) == pytest.approx(0.06)
    assert int(str(daily[0]["api_requests"])) == 1


def _assert_budget_refusal(response: httpx.Response, end_user: str) -> None:
    assert response.status_code == 422, response.text
    error: Final = object_value(response.json()["error"])
    assert error["type"] == "budget_exceeded"
    assert error["param"] is None
    assert error["code"] == "422"
    assert str(error["message"]).startswith(f"ExceededBudget: End User={end_user} over budget. Spend="), error[
        "message"
    ]


def _json_object(raw: bytes) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(raw)


_CHAT_SPELLINGS: Final = (
    pytest.param("x-litellm-end-user-id", id="header-x-litellm-end-user-id"),
    pytest.param("x-litellm-customer-id", id="header-x-litellm-customer-id"),
    pytest.param("litellm_metadata.user", id="body-litellm_metadata-user"),
    pytest.param("metadata.user_id", id="body-metadata-user_id"),
    pytest.param("litellm_metadata.user:json-string", id="body-litellm_metadata-json-string-user"),
    pytest.param("metadata.user_id:json-string", id="body-metadata-json-string-user_id"),
)


def _chat_request(
    gateway: Gateway,
    model: str,
    key: str,
    spelling: str,
    end_user: str,
    text: str,
    stream: bool = False,
) -> httpx.Response:
    headers: Final[dict[str, str]] = {}
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
    }
    if spelling.startswith("x-"):
        headers[spelling] = end_user
    elif spelling == "litellm_metadata.user":
        body["litellm_metadata"] = {"user": end_user}
    elif spelling == "litellm_metadata.user:json-string":
        body["litellm_metadata"] = json.dumps({"user": end_user})
    elif spelling == "metadata.user_id:json-string":
        body["metadata"] = json.dumps({"user_id": end_user})
    else:
        body["metadata"] = {"user_id": end_user}
    if stream:
        body["stream"] = True
    return gateway.request("POST", "/v1/chat/completions", body, key=key, headers=headers)


@pytest.mark.parametrize(
    "spelling",
    (
        pytest.param("x-litellm-end-user-id", id="header"),
        pytest.param("metadata.user_id", id="metadata"),
    ),
)
def test_streamed_end_user_spelling_is_charged_and_budget_enforced(gateway: Gateway, spelling: str) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, max_budget=0.05)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])

        text: Final = f"streamed end user spelling {uuid.uuid4().hex}"
        served: Final = _chat_request(gateway, model, key, spelling, end_user, text, stream=True)
        assert served.status_code == 200, served.text
        assert served.headers["content-type"].startswith("text/event-stream"), served.headers
        chunks: Final = tuple(
            _json_object(line.removeprefix("data: ").strip().encode())
            for line in served.text.splitlines()
            if line.startswith("data: ") and line.removeprefix("data: ").strip() != "[DONE]"
        )
        assert chunks
        assert object_value(chunks[-1]["choices"][0])["finish_reason"] == "stop"
        assert "data: [DONE]" in served.text
        observed: Final = _observed(upstream)
        assert observed == [
            _observation(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": text}],
                    "model": "gpt-4o-mini",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
        ]
        spend: Final = _charged_spend(end_user)
        assert spend == pytest.approx(20 * 0.001 + 20 * 0.002)
        _assert_daily_row(end_user, model)

        denied: Final = _chat_request(
            gateway,
            model,
            key,
            spelling,
            end_user,
            f"again {uuid.uuid4().hex}",
            stream=True,
        )
        assert denied.headers["content-type"].startswith("application/json"), denied.headers
        assert "data:" not in denied.text
        _assert_budget_refusal(denied, end_user)
        assert _observed(upstream) == []


@pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
def test_anthropic_messages_metadata_user_id_is_charged_and_budget_enforced(gateway: Gateway, stream: bool) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, max_budget=0.05)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        scenario_id: Final = f"sc-{uuid.uuid4().hex}"
        response: Final = (
            SseResponse(
                content_type="text/event-stream",
                frames=(
                    "event: message_start\ndata: "
                    '{"type":"message_start","message":{"id":"msg_$REQUEST_ID","type":"message",'
                    '"role":"assistant","model":"claude-sonnet-5-5","content":[],"stop_reason":null,'
                    '"stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":0}}}',
                    "event: content_block_start\ndata: "
                    '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                    "event: content_block_delta\ndata: "
                    '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"scripted answer"}}',
                    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
                    "event: message_delta\ndata: "
                    '{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
                    '"usage":{"output_tokens":20}}',
                    'event: message_stop\ndata: {"type":"message_stop"}',
                ),
            )
            if stream
            else JsonResponse(
                content_type="application/json",
                body={
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [{"type": "text", "text": "scripted answer"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 20, "output_tokens": 20},
                },
            )
        )
        handle: Final = register_scenario(scenario_id, response)
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-5-5",
            api_base=handle.api_base(),
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        key: Final = scenario.key(models=[model])
        text: Final = f"anthropic end user {uuid.uuid4().hex}"
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": text}],
            "metadata": {"user_id": end_user},
            "stream": stream,
        }
        served: Final = gateway.client.request(
            "POST",
            "/v1/messages",
            json=body,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        assert served.status_code == 200, served.text
        if stream:
            assert served.headers["content-type"].startswith("text/event-stream"), served.headers
            message_events: Final = tuple(
                _json_object(line.removeprefix("data: ").strip().encode())
                for line in served.text.splitlines()
                if line.startswith("data: ")
            )
            assert tuple(str(event["type"]) for event in message_events) == (
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            )
        else:
            message: Final = _AnthropicMessageResponse.model_validate_json(served.content)
            assert message.type == "message"
            assert message.role == "assistant"
            assert message.content == [_AnthropicTextBlock(type="text", text="scripted answer")]
            assert message.stop_reason == "end_turn"
            assert message.stop_sequence is None
            assert message.usage == _AnthropicMessageUsage(input_tokens=20, output_tokens=20)
        observed: Final = _observed(upstream)
        assert observed == [
            {
                "path": f"/{scenario_id}/v1/messages",
                "authorization": "",
                "body": {
                    "model": "claude-sonnet-5-5",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": text}],
                    "metadata": {"user_id": end_user},
                    "stream": stream,
                },
                "method": "POST",
                "api_key": "",
            }
        ]
        spend: Final = _charged_spend(end_user)
        assert spend == pytest.approx(20 * 0.001 + 20 * 0.002)
        _assert_daily_row(end_user, model, "anthropic/claude-sonnet-5-5")

        denied: Final = gateway.client.request(
            "POST",
            "/v1/messages",
            json={**body, "messages": [{"role": "user", "content": f"again {uuid.uuid4().hex}"}]},
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        _assert_budget_refusal(denied, end_user)
        assert _observed(upstream) == []


@pytest.mark.parametrize("spelling", _CHAT_SPELLINGS)
def test_end_user_id_spelling_is_charged_and_budget_enforced(gateway: Gateway, spelling: str) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, max_budget=0.05)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])

        text: Final = f"end user spelling {uuid.uuid4().hex}"
        served: Final = _chat_request(gateway, model, key, spelling, end_user, text)
        assert served.status_code == 200, served.text
        assert _observed(upstream) == [
            _observation(
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": text}], "model": "gpt-4o-mini"},
            )
        ]
        spend: Final = _charged_spend(end_user)
        assert spend == pytest.approx(20 * 0.001 + 20 * 0.002)
        _assert_daily_row(end_user, model)

        denied: Final = _chat_request(gateway, model, key, spelling, end_user, f"again {uuid.uuid4().hex}")
        _assert_budget_refusal(denied, end_user)
        assert _observed(upstream) == []


def test_safety_identifier_on_responses_is_charged_and_budget_enforced(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        scenario_id: Final = f"sc-{uuid.uuid4().hex}"
        handle: Final = register_scenario(
            scenario_id,
            JsonResponse(
                content_type="application/json",
                body={
                    "id": f"resp_{scenario_id}",
                    "object": "response",
                    "created_at": 1759700000,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "scripted answer", "annotations": []}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 20,
                        "total_tokens": 40,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            ),
        )
        scenario.cleanups.callback(
            lambda: httpx.delete(f"{handle.control_url}/__scenarios/{handle.scenario_id}", trust_env=False)
        )
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, max_budget=0.05)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(
            api_base=f"{gateway.upstream_url}/{handle.scenario_id}/v1",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        key: Final = scenario.key(models=[model])

        input_message: Final[dict[str, JsonValue]] = {
            "role": "user",
            "content": [{"type": "input_text", "text": f"responses spelling {uuid.uuid4().hex}"}],
        }

        def call() -> httpx.Response:
            return gateway.request(
                "POST",
                "/v1/responses",
                {"model": model, "input": [input_message], "safety_identifier": end_user},
                key=key,
            )

        served: Final = call()
        assert served.status_code == 200, served.text
        assert _observed(upstream) == [
            _observation(
                f"/{handle.scenario_id}/v1/responses",
                {"model": "gpt-4o-mini", "input": [input_message], "safety_identifier": end_user},
            )
        ]
        spend: Final = _charged_spend(end_user)
        assert spend == pytest.approx(0.06)
        _assert_daily_row(end_user, model)

        denied: Final = call()
        _assert_budget_refusal(denied, end_user)
        assert _observed(upstream) == []


def test_end_user_header_wins_over_body_user(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        header_user: Final = f"integration-end-user-a-{uuid.uuid4().hex}"
        body_user: Final = f"integration-end-user-b-{uuid.uuid4().hex}"
        _new_customer(gateway, header_user, max_budget=0.05)
        _new_customer(gateway, body_user, max_budget=1.0)
        for end_user in (header_user, body_user):
            scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])

        text: Final = f"header precedence {uuid.uuid4().hex}"
        served: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": text}], "user": body_user},
            key=key,
            headers={"x-litellm-end-user-id": header_user},
        )
        assert served.status_code == 200, served.text
        assert _observed(upstream) == [
            _observation(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": text}],
                    "model": "gpt-4o-mini",
                    "user": body_user,
                },
            )
        ]
        assert _charged_spend(header_user) == pytest.approx(0.06)
        _assert_daily_row(header_user, model)
        body_rows: Final = read_rows('SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id=%s', (body_user,))
        assert all(float(str(row["spend"])) == 0.0 for row in body_rows), body_rows
        assert (
            read_rows(
                'SELECT id FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id=%s',
                (body_user,),
            )
            == []
        )

        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"again {uuid.uuid4().hex}"}], "user": body_user},
            key=key,
            headers={"x-litellm-end-user-id": header_user},
        )
        _assert_budget_refusal(denied, header_user)
        assert _observed(upstream) == []


def _config_with_general_settings(tmp_path: Path, general_settings: Mapping[str, JsonValue]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {**object_value(config["general_settings"]), **dict(general_settings)}
    path: Final = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize(
    ("general_settings", "injects_user_body_field"),
    (
        pytest.param(
            {"user_header_mappings": [{"header_name": "x-integration-end-user", "litellm_user_role": "customer"}]},
            False,
            id="config-user_header_mappings",
        ),
        pytest.param({"user_header_name": "x-integration-end-user"}, True, id="config-user_header_name"),
    ),
)
def test_configured_end_user_header_is_charged_and_budget_enforced(
    gateway: Gateway, tmp_path: Path, general_settings: Mapping[str, JsonValue], injects_user_body_field: bool
) -> None:
    config: Final = _config_with_general_settings(tmp_path, general_settings)
    with (
        owned_proxy(gateway, tmp_path, {}, config=config) as candidate,
        candidate.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(candidate, end_user, max_budget=0.05)
        scenario.cleanups.callback(candidate.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        headers: Final = {"x-integration-end-user": end_user}

        text: Final = f"configured header {uuid.uuid4().hex}"
        served: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": text}]},
            key=key,
            headers=headers,
        )
        assert served.status_code == 200, served.text
        expected_body: Final[dict[str, JsonValue]] = {
            "messages": [{"role": "user", "content": text}],
            "model": "gpt-4o-mini",
        }
        if injects_user_body_field:
            expected_body["user"] = end_user
        assert _observed(upstream) == [_observation("/v1/chat/completions", expected_body)]
        assert _charged_spend(end_user) == pytest.approx(0.06)
        _assert_daily_row(end_user, model)

        denied: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"again {uuid.uuid4().hex}"}]},
            key=key,
            headers=headers,
        )
        _assert_budget_refusal(denied, end_user)
        assert _observed(upstream) == []

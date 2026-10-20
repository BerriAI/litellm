import json
import uuid
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, list_value
from integration._support.database import read_rows
from integration._support.deepseek_tokens import (
    CHINESE_TEXT,
    SHORT_TEXT,
    assert_counted_with_deepseek,
    chinese_minus_short,
)
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue


@pytest.mark.covers("spend.failed_dispatch.failure_row_records_estimated_input_tokens")
def test_provider_500_after_dispatch_records_estimated_prompt_tokens_on_failure_row(gateway: Gateway) -> None:
    prompt: Final = "failed dispatch accounting " + uuid.uuid4().hex
    system: Final = "You are a terse accounting assistant"

    def provider(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions"
        body: Final = json.loads(request.body)
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"] == [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return Reply(
            status=500,
            body=b'{"error":{"message":"synthetic provider outage","type":"server_error","code":"500"}}',
        )

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=0)
        key: Final = scenario.key(models=[model])
        failed: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
            key=key,
        )
        assert failed.status_code == 500 and "synthetic provider outage" in failed.text, failed.text
        call_id: Final = failed.headers["x-litellm-call-id"]
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, spend, prompt_tokens, completion_tokens, total_tokens FROM "LiteLLM_SpendLogs" '
                "WHERE request_id=%s",
                (call_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert row["status"] == "failure" and float(row["spend"]) == 0 and row["completion_tokens"] == 0, row
        assert row["prompt_tokens"] > 0, f"failure row lost the dispatched input tokens: {row}"
        assert row["total_tokens"] == row["prompt_tokens"], row


def _chat_body(model: str, text: str) -> Mapping[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": text}]}


def _responses_body(model: str, text: str) -> Mapping[str, JsonValue]:
    return {"model": model, "input": text}


def _messages_body(model: str, text: str) -> Mapping[str, JsonValue]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}]}


FAILURE_ROUTES: Final = pytest.mark.parametrize(
    ("route", "body"),
    (
        ("/v1/chat/completions", _chat_body),
        ("/v1/responses", _responses_body),
        ("/v1/messages", _messages_body),
    ),
    ids=("chat", "responses", "messages"),
)


def _provider_500_failure_row(
    gateway: Gateway,
    route: str,
    body: Callable[[str, str], Mapping[str, JsonValue]],
    headers: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, JsonValue]]:
    def provider(request: Request) -> Reply:
        assert request.method == "POST"
        return Reply(
            status=500,
            body=b'{"error":{"message":"synthetic provider outage","type":"server_error","code":"500"}}',
        )

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=0)
        key: Final = scenario.key(models=[model])
        failed: Final = gateway.request(
            "POST", route, body(model, f"failed dispatch group {uuid.uuid4().hex}"), key=key, headers=headers
        )
        assert failed.status_code == 500, failed.text
        assert wire.drain()
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, model_group, request_tags FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (failed.headers["x-litellm-call-id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["status"] == "failure", rows
        return model, rows[0]


@FAILURE_ROUTES
def test_a_provider_500_failure_row_names_the_model_group_the_caller_asked_for(
    gateway: Gateway, route: str, body: Callable[[str, str], Mapping[str, JsonValue]]
) -> None:
    model, row = _provider_500_failure_row(gateway, route, body)

    assert row["model_group"] == model, row


@FAILURE_ROUTES
def test_a_provider_500_failure_row_keeps_the_tags_the_caller_sent(
    gateway: Gateway, route: str, body: Callable[[str, str], Mapping[str, JsonValue]]
) -> None:
    tag: Final = f"failed-dispatch-{uuid.uuid4().hex}"

    _, row = _provider_500_failure_row(gateway, route, body, {"x-litellm-tags": tag})

    assert tag in list_value(row["request_tags"]), row


def _provider_500(request: Request) -> Reply:
    assert request.method == "POST"
    return Reply(
        status=500,
        body=b'{"error":{"message":"synthetic provider outage","type":"server_error","code":"500"}}',
    )


def _failure_row_prompt_tokens(gateway: Gateway, model: str, key: str, text: str) -> int:
    failed: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, text), key=key)
    assert failed.status_code == 500, failed.text
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT status, prompt_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (failed.headers["x-litellm-call-id"],),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert rows[0]["status"] == "failure", rows
    prompt_tokens: Final = rows[0]["prompt_tokens"]
    assert isinstance(prompt_tokens, int), rows
    return prompt_tokens


def test_provider_500_failure_rows_count_deepseek_v4_prompt_tokens_with_deepseeks_tokenizer(gateway: Gateway) -> None:
    with wire_server(_provider_500) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure_ai/DeepSeek-V4-Flash-0731", api_base=wire.url, num_retries=0)
        key: Final = scenario.key(models=[model])
        chinese: Final = _failure_row_prompt_tokens(gateway, model, key, CHINESE_TEXT)
        short: Final = _failure_row_prompt_tokens(gateway, model, key, SHORT_TEXT)
        assert wire.drain()
    assert_counted_with_deepseek(chinese - short, chinese_minus_short)

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, eventually
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

LIMIT: Final = 100
RESERVED_OUTPUT: Final = 80
ACTUAL_OUTPUT: Final = 5
ACTUAL_INPUT: Final = 90


def _provider(prompt_tokens: int, completion_tokens: int) -> Callable[[Request], Reply]:
    def reply(_: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-io-limits",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            ).encode()
        )

    return reply


@contextmanager
def _model(gateway: Gateway, prompt_tokens: int, completion_tokens: int) -> Iterator[tuple[Scenario, str]]:
    with wire_server(_provider(prompt_tokens, completion_tokens)) as wire, gateway.scenario() as scenario:
        yield scenario, scenario.model(api_base=f"{wire.url}/v1")


def _chat(gateway: Gateway, key: str, model: str, max_tokens: int) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": f"io limits {uuid.uuid4().hex}"}],
    }
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


def _limited_by(response: httpx.Response, descriptor: str) -> bool:
    return response.status_code == 429 and descriptor in response.text


def _admitted_after_credit_back(gateway: Gateway, key: str, model: str) -> httpx.Response:
    return eventually(lambda: _chat(gateway, key, model, RESERVED_OUTPUT), lambda response: response.status_code == 200)


def test_key_model_otpm_limit_reserves_output_credits_back_and_spares_other_keys(gateway: Gateway) -> None:
    with _model(gateway, prompt_tokens=10, completion_tokens=ACTUAL_OUTPUT) as (scenario, model):
        limited: Final = scenario.key(metadata={"model_otpm_limit": {model: LIMIT}})
        other: Final = scenario.key()

        first: Final = _chat(gateway, limited, model, RESERVED_OUTPUT)
        assert first.status_code == 200, first.text
        second: Final = _admitted_after_credit_back(gateway, limited, model)
        assert second.status_code == 200, second.text

        over: Final = eventually(
            lambda: _chat(gateway, limited, model, LIMIT - 2 * ACTUAL_OUTPUT + 1),
            lambda response: response.status_code == 429,
            return_last_on_timeout=True,
        )
        assert _limited_by(over, "model_per_key_otpm"), over.text
        unlimited: Final = _chat(gateway, other, model, LIMIT * 10)
        assert unlimited.status_code == 200, unlimited.text


def test_team_model_otpm_limit_is_shared_by_the_team_keys_and_spares_other_teams(gateway: Gateway) -> None:
    with _model(gateway, prompt_tokens=10, completion_tokens=ACTUAL_OUTPUT) as (scenario, model):
        team: Final = scenario.team(metadata={"model_otpm_limit": {model: LIMIT}})
        first_key: Final = scenario.key(team_id=team)
        second_key: Final = scenario.key(team_id=team)
        outsider: Final = scenario.key()

        first: Final = _chat(gateway, first_key, model, RESERVED_OUTPUT)
        assert first.status_code == 200, first.text
        second: Final = _admitted_after_credit_back(gateway, second_key, model)
        assert second.status_code == 200, second.text

        over: Final = eventually(
            lambda: _chat(gateway, second_key, model, LIMIT - 2 * ACTUAL_OUTPUT + 1),
            lambda response: response.status_code == 429,
            return_last_on_timeout=True,
        )
        assert _limited_by(over, "model_per_team_otpm"), over.text
        unlimited: Final = _chat(gateway, outsider, model, LIMIT * 10)
        assert unlimited.status_code == 200, unlimited.text


def test_key_model_itpm_limit_counts_the_input_the_provider_billed_and_ignores_output(gateway: Gateway) -> None:
    with _model(gateway, prompt_tokens=ACTUAL_INPUT, completion_tokens=LIMIT * 10) as (scenario, model):
        limited: Final = scenario.key(metadata={"model_itpm_limit": {model: LIMIT}})

        first: Final = _chat(gateway, limited, model, 1)
        assert first.status_code == 200, first.text

        over: Final = eventually(
            lambda: _chat(gateway, limited, model, 1),
            lambda response: response.status_code != 200,
            return_last_on_timeout=True,
        )
        assert _limited_by(over, "model_per_key_itpm"), over.text


BATCH_ROWS: Final = 3
BATCH_ROW_MAX_TOKENS: Final = 100
BATCH_ROW_WORDS: Final = 200


def _batch_lines(model: str) -> bytes:
    return "".join(
        json.dumps(
            {
                "custom_id": f"r{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "max_tokens": BATCH_ROW_MAX_TOKENS,
                    "messages": [{"role": "user", "content": "enqueued " * BATCH_ROW_WORDS}],
                },
            }
        )
        + "\n"
        for index in range(BATCH_ROWS)
    ).encode()


def _batch_provider(lines: bytes) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        path: Final = request.target.split("?", 1)[0]
        if request.method == "POST" and path == "/v1/files":
            return Reply(
                body=json.dumps(
                    {
                        "id": f"file-in-{uuid.uuid4().hex}",
                        "object": "file",
                        "purpose": "batch",
                        "bytes": len(lines),
                        "created_at": 1,
                        "filename": "in.jsonl",
                        "status": "processed",
                    }
                ).encode()
            )
        if request.method == "GET" and path.startswith("/v1/files/") and path.endswith("/content"):
            return Reply(body=lines, content_type="application/jsonl")
        if request.method == "POST" and path == "/v1/batches":
            return Reply(
                body=json.dumps(
                    {
                        "id": f"batch-{uuid.uuid4().hex}",
                        "object": "batch",
                        "endpoint": "/v1/chat/completions",
                        "input_file_id": "file-in",
                        "completion_window": "24h",
                        "status": "validating",
                        "created_at": 1,
                    }
                ).encode()
            )
        return Reply(status=404, body=json.dumps({"error": {"message": f"unscripted {request.target}"}}).encode())

    return reply


def _submit_batch(gateway: Gateway, key: str, model: str) -> httpx.Response:
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "model": model},
        {"file": ("in.jsonl", _batch_lines(model), "application/jsonl")},
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    return gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": uploaded.json()["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model,
        },
        key=key,
    )


def _batches_created(requests: tuple[Request, ...]) -> int:
    return sum(
        1 for request in requests if (request.method, request.target.split("?", 1)[0]) == ("POST", "/v1/batches")
    )


@contextmanager
def _batch_model(gateway: Gateway) -> Iterator[tuple[Scenario, str, Callable[[], tuple[Request, ...]]]]:
    model: Final = f"integration-batch-io-{uuid.uuid4().hex}"
    with wire_server(_batch_provider(_batch_lines(model))) as wire, gateway.scenario() as scenario:
        yield scenario, scenario.model(name=model, api_base=f"{wire.url}/v1"), wire.drain


def test_a_batch_over_the_key_model_itpm_limit_is_refused_as_an_itpm_limit_before_it_reaches_the_provider(
    gateway: Gateway,
) -> None:
    with _batch_model(gateway) as (scenario, model, drain):
        limited: Final = scenario.key(metadata={"model_itpm_limit": {model: BATCH_ROW_WORDS}})
        other: Final = scenario.key()

        refused: Final = _submit_batch(gateway, limited, model)
        assert refused.status_code == 429, refused.text
        assert "model_per_key_itpm" in refused.text and "ITPM limit" in refused.text, refused.text
        assert _batches_created(drain()) == 0

        admitted: Final = _submit_batch(gateway, other, model)
        assert admitted.status_code == 200, admitted.text
        assert _batches_created(drain()) == 1


def test_a_batch_whose_rows_may_produce_more_than_the_team_model_otpm_limit_is_refused_as_an_otpm_limit(
    gateway: Gateway,
) -> None:
    with _batch_model(gateway) as (scenario, model, drain):
        team: Final = scenario.team(metadata={"model_otpm_limit": {model: BATCH_ROWS * BATCH_ROW_MAX_TOKENS - 1}})
        member: Final = scenario.key(team_id=team)
        outsider: Final = scenario.key()

        refused: Final = _submit_batch(gateway, member, model)
        assert refused.status_code == 429, refused.text
        assert "model_per_team_otpm" in refused.text and "OTPM limit" in refused.text, refused.text
        assert _batches_created(drain()) == 0

        admitted: Final = _submit_batch(gateway, outsider, model)
        assert admitted.status_code == 200, admitted.text
        assert _batches_created(drain()) == 1


def test_key_model_tpm_limit_alone_still_counts_input_and_output_together(gateway: Gateway) -> None:
    with _model(gateway, prompt_tokens=ACTUAL_INPUT, completion_tokens=ACTUAL_OUTPUT) as (scenario, model):
        limited: Final = scenario.key(metadata={"model_tpm_limit": {model: LIMIT}})

        first: Final = _chat(gateway, limited, model, ACTUAL_OUTPUT)
        assert first.status_code == 200, first.text

        over: Final = eventually(
            lambda: _chat(gateway, limited, model, ACTUAL_OUTPUT * 2),
            lambda response: response.status_code == 429,
            return_last_on_timeout=True,
        )
        assert _limited_by(over, "model_per_key:"), over.text

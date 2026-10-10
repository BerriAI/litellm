import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.responses_stream import created, delta, error_event, failed, frame, healthy_stream
from integration._support.wire import Reply, Request, wire_server

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import _get_provider_request_id

_MODEL: Final = "bedrock/converse/anthropic.claude-sonnet-4-5-20250929-v1:0"
_TOKEN: Final = "synthetic-bedrock-bearer"


def test_bedrock_500_keeps_amzn_request_id_on_error_headers_and_failure_log(gateway: Gateway) -> None:
    identity: Final = f"bedrock-request-id-{uuid.uuid4().hex}"
    amzn_request_id: Final = str(uuid.uuid4())
    prompt: Final = f"failure probe {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse", request.target
        return Reply(
            status=500,
            headers={"x-amzn-RequestId": amzn_request_id},
            body=b'{"message":"synthetic bedrock failure"}',
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            num_retries=0,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        assert response.status_code >= 400, response.text
        assert response.headers.get("llm_provider-x-amzn-requestid") == amzn_request_id, dict(response.headers)
        call_id: Final = response.headers["x-litellm-call-id"]
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert row["status"] == "failure", row
        metadata: Final = row["metadata"]
        parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
        error_information: Final = object_value(parsed["error_information"])
        assert error_information["error_provider_request_id"] == amzn_request_id, error_information


@pytest.mark.parametrize("event_type", ("error", "response.failed"))
@pytest.mark.parametrize(
    "error_code", ("server_error", "rate_limit_exceeded", "invalid_request_error", "content_policy_violation")
)
@pytest.mark.parametrize("with_request_id", (True, False), ids=("with-id", "without-id"))
def test_responses_stream_failure_log_retains_provider_request_id(
    gateway: Gateway, event_type: str, error_code: str, with_request_id: bool
) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    provider_request_id: Final = str(uuid.uuid4()) if with_request_id else None
    message: Final = f"stream failure {identity}"
    event: Final = (
        error_event({"type": error_code, "code": error_code, "message": message})
        if event_type == "error"
        else failed(identity, error_code, message)
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        assert json.loads(request.body)["stream"] is True
        return Reply(
            headers={"x-amzn-RequestId": provider_request_id} if provider_request_id else {},
            content_type="text/event-stream",
            chunks=(frame(created(identity)), frame(delta(identity, "partial output")), frame(event)),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-6", api_base=wire.url, num_retries=0)
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": identity, "stream": True, "num_retries": 0, "cache": {"no-cache": True}},
        )
        assert response.status_code == 200, response.text
        assert "partial output" in response.text, response.text
        assert message in response.text, response.text
        assert response.headers.get("llm_provider-x-amzn-requestid") == provider_request_id, dict(response.headers)
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s',
                (response.headers["x-litellm-call-id"],),
            ),
            lambda values: len(values) >= 1,
            seconds=70,
        )
        assert len(rows) == 1, rows
        assert rows[0]["status"] == "failure", rows
        metadata: Final = rows[0]["metadata"]
        parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
        error_information: Final = object_value(parsed["error_information"])
        assert error_information["error_provider_request_id"] == provider_request_id, error_information


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("event_type", ("error", "response.failed"))
async def test_responses_sdk_stream_error_retains_provider_request_id(asynchronous: bool, event_type: str) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    provider_request_id: Final = str(uuid.uuid4())
    event: Final = (
        error_event({"type": "server_error", "code": "server_error", "message": identity})
        if event_type == "error"
        else failed(identity, "server_error", identity)
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        return Reply(
            headers={"x-amzn-RequestId": provider_request_id},
            content_type="text/event-stream",
            chunks=(frame(created(identity)), frame(delta(identity, "partial output")), frame(event)),
        )

    with wire_server(respond) as wire:
        with pytest.raises(MidStreamFallbackError) as raised:
            if asynchronous:
                async_stream: Final = await litellm.aresponses(
                    model="openai/gpt-6", input=identity, stream=True, api_base=wire.url,
                    api_key="synthetic-provider-key", num_retries=0,
                )
                async for _ in async_stream:
                    pass
            else:
                sync_stream: Final = litellm.responses(
                    model="openai/gpt-6", input=identity, stream=True, api_base=wire.url,
                    api_key="synthetic-provider-key", num_retries=0,
                )
                tuple(sync_stream)
        assert raised.value.status_code == 500
        assert raised.value.response.status_code == 500
        assert raised.value.generated_content == "partial output"
        assert _get_provider_request_id(raised.value) == provider_request_id
        assert raised.value.original_exception is not None
        assert _get_provider_request_id(raised.value.original_exception) == provider_request_id
        assert len(wire.drain()) == 1


def test_concurrent_responses_failures_keep_distinct_request_ids_and_recover(gateway: Gateway) -> None:
    identity: Final = uuid.uuid4().hex
    prompts: Final = tuple(f"{identity}-{index}" for index in range(4))
    recovery: Final = f"{identity}-recovery"

    def respond(request: Request) -> Reply:
        prompt: Final = json.loads(request.body)["input"]
        assert prompt in (*prompts, recovery)
        chunks: Final = (
            healthy_stream(f"resp_{prompt}", "recovered")
            if prompt == recovery
            else (
                frame(created(f"resp_{prompt}")),
                frame(delta(f"resp_{prompt}", "partial output")),
                frame(error_event({"type": "server_error", "code": "server_error", "message": prompt})),
            )
        )
        return Reply(content_type="text/event-stream", headers={"x-amzn-requestid": prompt}, chunks=chunks)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-6", api_base=wire.url, num_retries=0)

        def call(prompt: str) -> str:
            response: Final = gateway.request(
                "POST", "/v1/responses",
                {"model": model, "input": prompt, "stream": True, "num_retries": 0, "cache": {"no-cache": True}},
            )
            assert response.status_code == 200, response.text
            assert prompt in response.text if prompt != recovery else "recovered" in response.text
            return response.headers["x-litellm-call-id"]

        with ThreadPoolExecutor(max_workers=4) as workers:
            call_ids: Final = tuple(workers.map(call, prompts))
        recovered_call_id: Final = call(recovery)
        assert len(wire.drain()) == 5
        for prompt, call_id in zip((*prompts, recovery), (*call_ids, recovered_call_id), strict=True):
            rows: Final = eventually(
                lambda: read_rows('SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (call_id,)),
                lambda values: len(values) >= 1,
                seconds=70,
            )
            assert len(rows) == 1, rows
            assert rows[0]["status"] == ("success" if prompt == recovery else "failure"), rows
            if prompt != recovery:
                metadata: Final = rows[0]["metadata"]
                parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
                assert object_value(parsed["error_information"])["error_provider_request_id"] == prompt, parsed

import asyncio
import json
import re
import signal
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm import get_model_info

_MODEL: Final = "azure_ai/flux.2-pro"
_PROMPT: Final = "a lighthouse at dusk"
_SIZES: Final = ((1024, 1024), (1920, 1080), (2048, 2048))
_PRICE: Final = TypeAdapter(float)
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")


def _catalog_tiered(width: int, height: int) -> float:
    catalog: Final = get_model_info(_MODEL)
    megapixels: Final = width * height / 1_048_576
    return _PRICE.validate_python(catalog.get("output_cost_per_first_megapixel")) * min(
        megapixels, 1.0
    ) + _PRICE.validate_python(catalog.get("output_cost_per_additional_megapixel")) * max(megapixels - 1.0, 0.0)


def _reply(request: Request) -> Reply:
    if request.method != "POST" or not request.body:
        return Reply(status=404, body=json.dumps({"error": f"{request.method} {request.target}"}).encode())
    body: Final = json.loads(request.body)
    if request.target.startswith("/v1/chat/completions"):
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                }
            ).encode()
        )
    count: Final = int(body.get("num_images") or 1)
    return Reply(
        chunks=(json.dumps({"data": [{"b64_json": "aW1n"} for _ in range(count)]}).encode(),),
        pause_between_chunks=0.05,
    )


@dataclass(frozen=True, slots=True)
class _Call:
    path: str
    body: dict[str, JsonValue]
    expected_success_spend: float | None


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    response: httpx.Response
    client_port: int


def _mixed_calls(image_model: str, chat_model: str, count: int) -> tuple[_Call, ...]:
    def call(index: int) -> _Call:
        if index % 4 == 3:
            return _Call(
                "/v1/chat/completions",
                {"model": chat_model, "messages": [{"role": "user", "content": f"chaos {uuid.uuid4().hex}"}]},
                None,
            )
        width, height = _SIZES[index % len(_SIZES)]
        return _Call(
            "/v1/images/generations",
            {"model": image_model, "prompt": _PROMPT, "size": f"{width}x{height}"},
            _catalog_tiered(width, height),
        )

    return tuple(call(index) for index in range(count))


async def _fire(base_url: str, key: str, calls: tuple[_Call, ...]) -> tuple[_Served | BaseException, ...]:
    async def one(client: httpx.AsyncClient, call: _Call) -> _Served:
        async with client.stream("POST", call.path, json=call.body, headers={"Authorization": f"Bearer {key}"}) as r:
            port: Final = int(r.extensions["network_stream"].get_extra_info("client_addr")[1])
            await r.aread()
        return _Served(call, r, port)

    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(one(client, call) for call in calls), return_exceptions=True))


def _rows(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT spend, status FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,))


def _one_row(call_id: str) -> dict[str, JsonValue]:
    return eventually(lambda: _rows(call_id), lambda values: len(values) == 1, seconds=70)[0]


def _request_id(served: _Served) -> str:
    if served.call.expected_success_spend is None and served.response.status_code == 200:
        return TypeAdapter(str).validate_python(served.response.json()["id"])
    return served.response.headers["x-litellm-call-id"]


def _assert_billed_once(served: _Served) -> None:
    call_id: Final = _request_id(served)
    row: Final = _one_row(call_id)
    spend: Final = _PRICE.validate_python(row["spend"])
    if served.response.status_code != 200:
        assert (row["status"], spend) == ("failure", 0.0), (call_id, row, served.response.text)
        return
    assert row["status"] == "success", (call_id, row)
    if served.call.expected_success_spend is not None:
        assert spend == pytest.approx(served.call.expected_success_spend), (call_id, served.call.body, row)
        assert float(served.response.headers["x-litellm-response-cost"]) == pytest.approx(spend), call_id


def _models(gateway: Gateway, stack: ExitStack, api_base: str) -> tuple[str, str]:
    scenario: Final = stack.enter_context(gateway.scenario())
    image_model: Final = scenario.model(
        model=_MODEL, api_base=api_base, api_key="synthetic-azure-key", api_version="preview", num_retries=0
    )
    chat_model: Final = scenario.model(api_base=f"{api_base}/v1", num_retries=0)
    return image_model, chat_model


async def test_upstream_outage_mid_mixed_burst_bills_every_served_image_once_by_megapixel(
    gateway: Gateway, tmp_path: Path
) -> None:
    with ExitStack() as upstream:
        wire: Final[Wire] = upstream.enter_context(wire_server(_reply))
        port: Final = int(wire.url.rsplit(":", 1)[1])
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned, ExitStack() as models:
            candidate: Final = owned.gateway
            image_model, chat_model = _models(candidate, models, wire.url)
            burst: Final = asyncio.create_task(
                _fire(str(candidate.client.base_url), candidate.key, _mixed_calls(image_model, chat_model, 30))
            )
            await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= 10, 30)
            upstream.close()
            during_outage: Final = candidate.request(
                "POST", "/v1/images/generations", {"model": image_model, "prompt": _PROMPT, "size": "2048x2048"}
            )
            assert during_outage.status_code >= 500, during_outage.text
            assert "error" in during_outage.json(), during_outage.text
            with wire_server(_reply, port=port):
                served: Final = await burst
                recovered: Final = candidate.request(
                    "POST", "/v1/images/generations", {"model": image_model, "prompt": _PROMPT, "size": "2048x2048"}
                )
            assert recovered.status_code == 200, recovered.text
            assert len(_STARTED_WORKER.findall(owned.log.read_text())) >= 2
    assert len(served) == 30
    for item in served:
        assert isinstance(item, _Served), repr(item)
        assert "x-litellm-call-id" in item.response.headers, (item.response.status_code, item.response.text)
        _assert_billed_once(item)
    ids: Final = tuple(_request_id(item) for item in served if isinstance(item, _Served))
    assert len(set(ids)) == 30, ids
    assert any(item.response.status_code == 200 for item in served if isinstance(item, _Served))
    assert _PRICE.validate_python(_one_row(string_value(recovered.headers["x-litellm-call-id"]))["spend"]) == (
        pytest.approx(_catalog_tiered(2048, 2048))
    )
    outage_row: Final = _one_row(string_value(during_outage.headers["x-litellm-call-id"]))
    assert (outage_row["status"], _PRICE.validate_python(outage_row["spend"])) == ("failure", 0.0), outage_row


async def test_worker_sigkill_mid_burst_leaves_the_sibling_billing_by_megapixel(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_reply) as wire, owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned:
        with ExitStack() as models:
            candidate: Final = owned.gateway
            image_model, chat_model = _models(candidate, models, wire.url)
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            base_url: Final = str(candidate.client.base_url)
            burst: Final = asyncio.create_task(
                _fire(base_url, candidate.key, _mixed_calls(image_model, chat_model, 24))
            )
            await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= 5, 30)
            victim: Final = psutil.Process(workers[0])
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            interrupted: Final = await burst
            after_kill: Final = await _fire(base_url, candidate.key, _mixed_calls(image_model, chat_model, 12))
            for result in interrupted:
                assert isinstance(result, _Served) or isinstance(result, httpx.TransportError), repr(result)
            for result in after_kill:
                assert isinstance(result, _Served), repr(result)
                assert result.response.status_code == 200, result.response.text
                _assert_billed_once(result)
            for item in (result for result in interrupted if isinstance(result, _Served)):
                assert len(_rows(_request_id(item))) <= 1, item.response.headers
            assert not psutil.pid_exists(workers[0]) or psutil.Process(workers[0]).status() == psutil.STATUS_ZOMBIE

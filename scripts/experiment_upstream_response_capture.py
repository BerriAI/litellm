import asyncio
import json
import sys
from collections.abc import Callable, Iterator
from typing import Final

import httpx
from openai import AsyncOpenAI, OpenAI

from litellm.litellm_core_utils.upstream_response_capture import (
    UpstreamResponseCapture,
    async_capture_explicit_response_headers,
    async_capture_response_headers,
    capture_explicit_response_headers,
    capture_response_headers,
)


def retry_upstream(statuses: Iterator[int]) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        status: Final = next(statuses)
        return httpx.Response(
            status,
            request=request,
            headers={"x-request-id": f"exchange-{status}", "retry-after-ms": "1"},
            json={
                "id": "chatcmpl-probe",
                "object": "chat.completion",
                "created": 1,
                "model": "header-probe",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            }
            if status == 200
            else {"error": {"message": "retry probe", "type": "rate_limit_error"}},
        )

    return respond


def sdk_sync_retry() -> tuple[int, ...]:
    capture: Final = UpstreamResponseCapture()
    statuses: Final = (429, 200)
    with OpenAI(
        api_key="experiment-key",
        max_retries=1,
        http_client=httpx.Client(
            transport=httpx.MockTransport(retry_upstream(iter(statuses))),
            event_hooks={"response": [capture_response_headers]},
        ),
    ) as client:
        with capture.bind("sdk-sync"):
            result: Final = client.chat.completions.create(
                model="header-probe", messages=[{"role": "user", "content": "probe"}]
            )
        assert result.choices[0].message.content == "ok"
    captured: Final = tuple(record.status_code for record in capture.responses)
    assert captured == statuses
    return captured


async def sdk_async_retry() -> tuple[int, ...]:
    capture: Final = UpstreamResponseCapture()
    statuses: Final = (429, 200)
    async with AsyncOpenAI(
        api_key="experiment-key",
        max_retries=1,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(retry_upstream(iter(statuses))),
            event_hooks={"response": [async_capture_response_headers]},
        ),
    ) as client:
        with capture.bind("sdk-async"):
            result: Final = await client.chat.completions.create(
                model="header-probe", messages=[{"role": "user", "content": "probe"}]
            )
        assert result.choices[0].message.content == "ok"
    captured: Final = tuple(record.status_code for record in capture.responses)
    assert captured == statuses
    return captured


async def executor_binding() -> tuple[int, int, int]:
    capture: Final = UpstreamResponseCapture()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-request-id": "thread-probe"}, request=request)

    with httpx.Client(
        transport=httpx.MockTransport(upstream), event_hooks={"response": [capture_response_headers]}
    ) as client:

        def send() -> None:
            client.get("https://upstream.invalid/")

        with capture.bind("parent"):
            await asyncio.get_running_loop().run_in_executor(None, send)
        ambient_count: Final = len(capture.responses)

        def explicitly_bound_send() -> None:
            with capture.bind("explicit-worker"):
                send()

        await asyncio.get_running_loop().run_in_executor(None, explicitly_bound_send)
    explicit_count: Final = len(capture.responses) - ambient_count
    assert explicit_count == 1

    request_capture: Final = UpstreamResponseCapture()
    with httpx.Client(
        transport=httpx.MockTransport(upstream), event_hooks={"response": [capture_explicit_response_headers]}
    ) as explicit_client:

        def send_with_request_owner() -> None:
            explicit_client.get(
                "https://upstream.invalid/", extensions=request_capture.request_extensions("request-owner")
            )

        await asyncio.get_running_loop().run_in_executor(None, send_with_request_owner)
    request_count: Final = len(request_capture.responses)
    assert request_count == 1
    return ambient_count, explicit_count, request_count


async def background_task_after_scope(*, explicit: bool) -> int:
    capture: Final = UpstreamResponseCapture()
    released: Final = asyncio.Event()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-request-id": "background-probe"}, request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        event_hooks={
            "response": [async_capture_explicit_response_headers if explicit else async_capture_response_headers]
        },
    ) as client:

        async def background_request() -> None:
            await released.wait()
            await client.get("https://upstream.invalid/background")

        with capture.bind("already-finished"):
            task: Final = asyncio.create_task(background_request())
        released.set()
        await task
    return len(capture.responses)


async def main() -> None:
    sync_statuses: Final = sdk_sync_retry()
    async_statuses: Final = await sdk_async_retry()
    ambient_count, explicit_count, request_count = await executor_binding()
    ambient_background_count: Final = await background_task_after_scope(explicit=False)
    explicit_background_count: Final = await background_task_after_scope(explicit=True)
    sys.stdout.write(
        json.dumps(
            {
                "sdk_sync_retry_statuses": sync_statuses,
                "sdk_async_retry_statuses": async_statuses,
                "executor_ambient_capture_count": ambient_count,
                "executor_explicit_capture_count": explicit_count,
                "executor_request_extension_capture_count": request_count,
                "untagged_background_request_ambient_capture_count": ambient_background_count,
                "untagged_background_request_explicit_capture_count": explicit_background_count,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    asyncio.run(main())

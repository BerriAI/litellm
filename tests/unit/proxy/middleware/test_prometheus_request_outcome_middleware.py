from typing import Final

import pytest
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.middleware.prometheus_request_outcome_middleware import (
    PrometheusRequestOutcomeMiddleware,
    mark_request_stream_failed,
)
from litellm.types.integrations.prometheus import ProxyRequestOutcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario, expected",
    [
        ("success", ("success",)),
        ("stream_error", ("error",)),
        ("disconnect", ("cancelled",)),
        ("after_finish_disconnect", ("success",)),
        ("http_error", ("error",)),
        ("other_route", ()),
    ],
)
async def test_records_one_client_outcome_for_chat(scenario: str, expected: tuple[ProxyRequestOutcome, ...]) -> None:
    outcomes: Final[list[ProxyRequestOutcome]] = []  # mutable-ok: captures emitted outcomes

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 429 if scenario == "http_error" else 200, "headers": []})
        if scenario == "stream_error":
            await send({"type": "http.response.body", "body": b"data: first\n\n", "more_body": True})
            mark_request_stream_failed()
        if scenario == "disconnect":
            await receive()
            return
        await send({"type": "http.response.body", "body": b"data: last\n\n", "more_body": False})
        if scenario == "after_finish_disconnect":
            await receive()

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    scope: Final[Scope] = {
        "type": "http",
        "http_version": "1.1",
        "scheme": "http",
        "method": "POST",
        "path": "/v1/embeddings" if scenario == "other_route" else "/v1/chat/completions",
        "raw_path": b"",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": None,
        "server": None,
    }
    await PrometheusRequestOutcomeMiddleware(app, recorder=lambda: outcomes.append)(scope, receive, send)

    assert tuple(outcomes) == expected

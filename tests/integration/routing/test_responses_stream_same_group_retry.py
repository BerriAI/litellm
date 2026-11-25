import itertools
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import PROVIDER_URL, SharedProvider
from integration._support.responses_stream import (
    AZURE_TARGET,
    azure_rate_limit,
    created,
    error_event,
    frame,
    healthy_stream,
    rate_limited_stream,
    response_object,
    sse_events,
)
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_RETRY_TEXT: Final = "retry answer"


def _throttle_first_request_only(identity: str) -> Callable[[Request], Reply]:
    """Shared by both regions: whichever the router picks first is out of quota, the other one answers."""
    arrivals: Final = itertools.count()

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target.startswith(AZURE_TARGET), request.target
        stream: Final = rate_limited_stream(identity) if next(arrivals) == 0 else healthy_stream(identity, _RETRY_TEXT)
        return Reply(content_type="text/event-stream", chunks=stream)

    return respond


def test_streamed_responses_in_stream_rate_limit_is_retried_in_its_model_group(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    respond: Final = _throttle_first_request_only(identity)
    with wire_server(respond) as region_a, wire_server(respond) as region_b, gateway.scenario() as scenario:
        group: Final = f"responses-stream-retry-{uuid.uuid4().hex}"
        for region in (region_a, region_b):
            scenario.model(name=group, model="azure/gpt-6", api_base=region.url, api_key="synthetic-azure-key")
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": group, "input": identity, "stream": True, "num_retries": 2, "cache": {"no-cache": True}},
        )
        assert response.status_code == 200, response.text
        events: Final = sse_events(response.text)
        assert [name for name, _ in events][-1] == "response.completed", response.text
        assert _RETRY_TEXT in response.text, response.text
        calls_per_region: Final = (len(region_a.drain()), len(region_b.drain()))
        assert sum(calls_per_region) == 2, f"one throttled call and one retry: {calls_per_region}"
        assert response.headers["x-litellm-attempted-retries"] == "1", dict(response.headers)


@pytest.mark.parametrize("fallback", (False, True))
@pytest.mark.parametrize(
    "output_event",
    (
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {"id": "rs_first", "type": "reasoning", "summary": []},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 1,
            "output_index": 0,
            "item_id": "rs_first",
            "summary_index": 0,
            "delta": "Considering the question",
        },
        {
            "type": "response.content_part.added",
            "sequence_number": 1,
            "output_index": 0,
            "item_id": "msg_first",
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    ),
)
def test_responses_stream_never_restarts_after_non_lifecycle_output(
    gateway: Gateway, provider: SharedProvider, output_event: dict[str, JsonValue], fallback: bool
) -> None:
    provider.expect(
        Reply(
            content_type="text/event-stream",
            chunks=(
                frame(created("resp_first")),
                frame(output_event),
                frame(
                    {
                        "type": "response.in_progress",
                        "sequence_number": 2,
                        "response": response_object("resp_first", "in_progress"),
                    }
                ),
                frame({**error_event(azure_rate_limit()), "sequence_number": 3}),
            ),
        )
    )
    with gateway.scenario() as scenario:
        group: Final = scenario.model(model="azure/gpt-6", api_base=PROVIDER_URL, api_key="synthetic-azure-key")
        backup: Final = scenario.model(model="azure/gpt-6", api_base=PROVIDER_URL, api_key="synthetic-azure-key")
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": group,
                "input": "reason then fail",
                "stream": True,
                "num_retries": 0 if fallback else 2,
                "fallbacks": [{group: [backup]}] if fallback else [],
                "cache": {"no-cache": True},
            },
        )
        received: Final = provider.received()
        assert response.status_code == 200, response.text
        events: Final = sse_events(response.text)
        assert len(received) == 1, response.text
        assert received[0].method == "POST" and received[0].target.startswith(AZURE_TARGET), received
        assert sum(name == "response.created" for name, _ in events) == 1, response.text
        assert output_event["type"] in [name for name, _ in events], response.text
        assert [name for name, _ in events] == [
            "response.created",
            output_event["type"],
            "response.in_progress",
            "response.failed",
        ], response.text

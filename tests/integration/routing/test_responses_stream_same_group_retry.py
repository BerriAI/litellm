import itertools
import uuid
from collections.abc import Callable
from typing import Final

from integration._support.client import Gateway
from integration._support.responses_stream import (
    AZURE_TARGET,
    azure_rate_limit,
    created,
    error_event,
    frame,
    healthy_stream,
    rate_limited_stream,
    sse_events,
)
from integration._support.wire import Reply, Request, wire_server

_RETRY_TEXT: Final = "retry answer"


def _first_request_gets(first: tuple[bytes, ...], identity: str) -> Callable[[Request], Reply]:
    """Shared by both regions: whichever the router picks first serves ``first``, the other one answers."""
    arrivals: Final = itertools.count()

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target.startswith(AZURE_TARGET), request.target
        stream: Final = first if next(arrivals) == 0 else healthy_stream(identity, _RETRY_TEXT)
        return Reply(content_type="text/event-stream", chunks=stream)

    return respond


def test_streamed_responses_in_stream_rate_limit_is_retried_in_its_model_group(gateway: Gateway) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    respond: Final = _first_request_gets(rate_limited_stream(identity), identity)
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


def _throttled_after_an_output_item(identity: str) -> tuple[bytes, ...]:
    item: Final = {
        "type": "message",
        "id": f"msg_{identity}",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    added: Final = {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": item}
    return (frame(created(identity)), frame(added), frame(error_event(azure_rate_limit())))


def test_streamed_responses_rate_limit_after_events_reached_the_client_sends_one_response_created(
    gateway: Gateway,
) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    respond: Final = _first_request_gets(_throttled_after_an_output_item(identity), identity)
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
        names: Final = [name for name, _ in sse_events(response.text)]
        assert names == ["response.created", "response.output_item.added", "response.failed"], response.text
        assert sum((len(region_a.drain()), len(region_b.drain()))) == 1, "no retry once the client has events"


def _unavailable(request: Request) -> Reply:
    return Reply(status=500, content_type="application/json", chunks=(b'{"error":{"message":"synthetic outage"}}',))


def test_streamed_responses_retry_inside_a_fallback_group_keeps_the_attempted_fallbacks_header(
    gateway: Gateway,
) -> None:
    identity: Final = "resp_" + uuid.uuid4().hex
    respond: Final = _first_request_gets(rate_limited_stream(identity), identity)
    with (
        wire_server(_unavailable) as primary_region,
        wire_server(respond) as region_a,
        wire_server(respond) as region_b,
        gateway.scenario() as scenario,
    ):
        primary: Final = f"responses-stream-primary-{uuid.uuid4().hex}"
        group: Final = f"responses-stream-retry-{uuid.uuid4().hex}"
        scenario.model(name=primary, model="azure/gpt-6", api_base=primary_region.url, api_key="synthetic-azure-key")
        for region in (region_a, region_b):
            scenario.model(name=group, model="azure/gpt-6", api_base=region.url, api_key="synthetic-azure-key")
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": primary,
                "fallbacks": [group],
                "input": identity,
                "stream": True,
                "num_retries": 1,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert _RETRY_TEXT in response.text, response.text
        attempted: Final = {
            name: response.headers.get(name)
            for name in ("x-litellm-attempted-fallbacks", "x-litellm-attempted-retries")
        }
        assert attempted == {"x-litellm-attempted-fallbacks": "1", "x-litellm-attempted-retries": "1"}, attempted

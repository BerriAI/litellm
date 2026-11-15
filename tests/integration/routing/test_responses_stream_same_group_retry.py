import itertools
import uuid
from collections.abc import Callable
from typing import Final

from integration._support.client import Gateway
from integration._support.responses_stream import (
    AZURE_TARGET,
    healthy_stream,
    rate_limited_stream,
    sse_events,
)
from integration._support.wire import Reply, Request, wire_server

_SIBLING_TEXT: Final = "sibling answer"


def _throttle_first_request_only(identity: str) -> Callable[[Request], Reply]:
    """Shared by both regions: whichever the router picks first is out of quota, the other one answers."""
    arrivals: Final = itertools.count()

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target.startswith(AZURE_TARGET), request.target
        stream: Final = (
            rate_limited_stream(identity) if next(arrivals) == 0 else healthy_stream(identity, _SIBLING_TEXT)
        )
        return Reply(content_type="text/event-stream", chunks=stream)

    return respond


def test_streamed_responses_in_stream_rate_limit_is_retried_on_the_sibling_deployment(gateway: Gateway) -> None:
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
        assert _SIBLING_TEXT in response.text, response.text
        calls_per_region: Final = (len(region_a.drain()), len(region_b.drain()))
        assert calls_per_region == (1, 1), f"the retry must leave the throttled region: {calls_per_region}"
        assert response.headers["x-litellm-attempted-retries"] == "1", dict(response.headers)

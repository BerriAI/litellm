from __future__ import annotations

import json
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from _s3_v2_support import RecordingS3Sink, call_surface, collect_payloads, matched_ids, s3_config, surface_reply
from integration._support.client import Gateway, JsonValue, Scenario, eventually, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

Surface = Literal["chat", "messages", "responses"]
SURFACES: Final[tuple[Surface, ...]] = ("chat", "messages", "responses")
FAILURE_BODY: Final[bytes] = json.dumps(
    {
        "error": {
            "message": "synthetic upstream failure",
            "type": "server_error",
            "param": None,
            "code": "synthetic_failure",
        }
    }
).encode()


def _failure_provider(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(status=404)
    return Reply(status=503, body=FAILURE_BODY)


def _surface_target(surface: Surface) -> str:
    return {
        "chat": "/v1/chat/completions",
        "messages": "/v1/messages",
        "responses": "/v1/responses",
    }[surface]


def _surface_request(
    candidate: Gateway,
    surface: Surface,
    openai_model: str,
    anthropic_model: str,
    key: str,
    marker: str,
    stream: bool,
) -> httpx.Response:
    if surface == "chat":
        body: Final = {
            "model": openai_model,
            "messages": [{"role": "user", "content": marker}],
            "stream": stream,
        }
        return candidate.request("POST", _surface_target(surface), body, key=key)
    if surface == "messages":
        body: Final = {
            "model": anthropic_model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": marker}],
            "stream": stream,
        }
        return candidate.request("POST", _surface_target(surface), body, key=key)
    body: Final = {"model": openai_model, "input": marker, "stream": stream}
    return candidate.request("POST", _surface_target(surface), body, key=key)


def _responses_stream_id(response: httpx.Response) -> str:
    events: Final = tuple(
        object_value(json.loads(line.removeprefix("data: ")))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    completed: Final = tuple(
        object_value(event["response"])
        for event in events
        if event.get("type") == "response.completed"
    )
    assert len(completed) == 1, events
    response_id: Final = completed[0]["id"]
    assert isinstance(response_id, str), response_id
    return response_id


def _register_models(scenario: Scenario, upstream_url: str, num_retries: int = 0) -> tuple[str, str]:
    openai_model: Final = scenario.model(
        model="openai/gpt-4o-mini",
        api_base=upstream_url + "/v1",
        api_key="synthetic-provider-key",
        num_retries=num_retries,
    )
    anthropic_model: Final = scenario.model(
        model="anthropic/claude-sonnet-4-5-20250929",
        api_base=upstream_url,
        api_key="synthetic-provider-key",
        num_retries=num_retries,
    )
    return openai_model, anthropic_model


def _request_key(payload: dict[str, JsonValue], response: httpx.Response, marker: str) -> str | None:
    call_id: Final = response.headers.get("x-litellm-call-id")
    if call_id is not None and call_id in {str(payload.get("id")), str(payload.get("litellm_call_id"))}:
        return call_id
    if marker in json.dumps(payload):
        return marker
    return None


def _assert_upstream_requests(requests: tuple[Request, ...], surface: Surface, expected: int) -> None:
    target: Final = _surface_target(surface)
    posts: Final = tuple(request for request in requests if request.method == "POST")
    observed: Final = tuple(request.target for request in posts)
    assert len(posts) == expected, f"expected {expected} upstream POSTs, observed {len(posts)}: {observed}"
    assert all(request.target == target for request in posts), f"expected {target}, observed {observed}"


def _assert_one_payload(
    sink: RecordingS3Sink,
    response: httpx.Response,
    marker: str,
    status: str,
    upstream_posts: int,
) -> tuple[dict[str, JsonValue], ...]:
    first_payloads: Final = collect_payloads(sink, 1)
    assert first_payloads[0]["status"] == status
    # A full three-flush-interval window is needed to detect late duplicate uploads.
    payloads: Final = eventually(
        lambda: sink.payloads(),
        lambda observed: len(observed) >= 2,
        seconds=6,
        return_last_on_timeout=True,
    )
    assert len(payloads) == 1, (
        f"expected one {status} payload after {upstream_posts} upstream POSTs, "
        f"observed {len(payloads)} payloads"
    )
    assert (
        _request_key(payloads[0], response, marker) == response.headers.get("x-litellm-call-id")
        or marker in json.dumps(payloads[0])
    ), f"payload did not match request id or marker: {payloads[0]!r}"
    return payloads


@pytest.mark.parametrize(
    ("surface", "stream"),
    [
        pytest.param("chat", False, id="chat-nonstream"),
        pytest.param("chat", True, id="chat-stream"),
        pytest.param("messages", False, id="messages-nonstream"),
        pytest.param("messages", True, id="messages-stream"),
        pytest.param("responses", False, id="responses-nonstream"),
        pytest.param("responses", True, id="responses-stream"),
    ],
)
def test_retried_failure_uploads_one_s3_object(
    gateway: Gateway,
    tmp_path: Path,
    surface: Surface,
    stream: bool,
) -> None:
    marker: Final = f"s3-a-{surface}-{uuid.uuid4().hex}"
    sink: Final = RecordingS3Sink()
    with wire_server(_failure_provider) as upstream, wire_server(sink.respond) as bucket:
        config: Final = s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "2"},
                config=config,
                workers=2,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            openai_model, anthropic_model = _register_models(scenario, upstream.url, num_retries=2)
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            response: Final = _surface_request(
                candidate,
                surface,
                openai_model,
                anthropic_model,
                key,
                marker,
                stream,
            )
            requests: Final = upstream.drain()
            _assert_upstream_requests(requests, surface, 3)
            assert not 200 <= response.status_code < 300, f"{response.status_code}: {response.text}"
            assert "synthetic upstream failure" in response.text, response.text
            _assert_one_payload(sink, response, marker, "failure", len(requests))


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_fallback_chain_failure_uploads_one_s3_object(
    gateway: Gateway,
    tmp_path: Path,
    stream: bool,
) -> None:
    marker: Final = f"s3-b-{uuid.uuid4().hex}"
    sink: Final = RecordingS3Sink()
    with (
        wire_server(_failure_provider) as primary_upstream,
        wire_server(_failure_provider) as secondary_upstream,
        wire_server(sink.respond) as bucket,
    ):
        config: Final = s3_config(tmp_path, bucket.url, {}, settings={"num_retries": 0})
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "2"},
                config=config,
                workers=2,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            primary_model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=primary_upstream.url + "/v1",
                api_key="synthetic-primary-key",
            )
            secondary_model: Final = scenario.model(
                model="openai/gpt-4o-mini",
                api_base=secondary_upstream.url + "/v1",
                api_key="synthetic-secondary-key",
            )
            key: Final = scenario.key(models=[primary_model, secondary_model])
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": primary_model,
                    "messages": [{"role": "user", "content": marker}],
                    "stream": stream,
                    "fallbacks": [secondary_model],
                    "num_retries": 0,
                },
                key=key,
            )
            primary_requests: Final = primary_upstream.drain()
            secondary_requests: Final = secondary_upstream.drain()
            _assert_upstream_requests(primary_requests, "chat", 1)
            _assert_upstream_requests(secondary_requests, "chat", 1)
            assert not 200 <= response.status_code < 300, f"{response.status_code}: {response.text}"
            assert "synthetic upstream failure" in response.text, response.text
            _assert_one_payload(sink, response, marker, "failure", len(primary_requests) + len(secondary_requests))


@pytest.mark.parametrize("surface", SURFACES, ids=SURFACES)
def test_streaming_success_uploads_one_s3_object(
    gateway: Gateway,
    tmp_path: Path,
    surface: Surface,
) -> None:
    marker: Final = f"s3-c-{surface}-{uuid.uuid4().hex}"
    sink: Final = RecordingS3Sink()
    with wire_server(surface_reply) as upstream, wire_server(sink.respond) as bucket:
        config: Final = s3_config(tmp_path, bucket.url, {}, settings={"num_retries": 0})
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "2"},
                config=config,
                workers=2,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            openai_model, anthropic_model = _register_models(scenario, upstream.url)
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            captured_responses: Final[list[httpx.Response]] = []
            candidate.client.event_hooks["response"].append(captured_responses.append)
            surface_response_id, call_id = call_surface(
                candidate,
                f"{surface}_stream",
                openai_model,
                anthropic_model,
                key,
                marker,
            )
            if surface == "responses":
                assert len(captured_responses) == 1, captured_responses
            response_id: Final = (
                _responses_stream_id(captured_responses[0]) if surface == "responses" else surface_response_id
            )
            payloads: Final = collect_payloads(sink, 1)
            payloads_after_window: Final = eventually(
                lambda: sink.payloads(),
                lambda observed: len(observed) >= 2,
                seconds=6,
                return_last_on_timeout=True,
            )
            assert len(payloads_after_window) == 1, payloads_after_window
            assert payloads[0]["status"] == "success"
            if surface == "responses":
                assert payloads[0]["litellm_call_id"] == call_id
            else:
                assert payloads[0]["id"] == response_id
            assert matched_ids(payloads, ((response_id, call_id),)) == frozenset({str(payloads[0]["id"])})


def test_failure_burst_through_sink_outage_lands_each_request_once(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    requests_per_variant: Final = 4
    jobs: Final = tuple(
        (surface, stream, f"s3-d-{surface}-{'stream' if stream else 'nonstream'}-{index}")
        for surface in SURFACES
        for stream in (False, True)
        for index in range(requests_per_variant)
    )
    sink: Final = RecordingS3Sink()
    with wire_server(_failure_provider) as upstream, wire_server(sink.respond) as bucket:
        config: Final = s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "2"},
                config=config,
                workers=2,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            openai_model, anthropic_model = _register_models(scenario, upstream.url, num_retries=2)
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            sink.fail_until = float("inf")

            def call(job: tuple[Surface, bool, str]) -> httpx.Response:
                surface, stream, marker = job
                return _surface_request(candidate, surface, openai_model, anthropic_model, key, marker, stream)

            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                responses: Final = tuple(pool.map(call, jobs))

            upstream_requests: Final = tuple(
                request for request in upstream.drain() if request.method == "POST"
            )
            route_counts: Final = Counter(request.target for request in upstream_requests)
            expected_route_counts: Final = {
                _surface_target(surface): 2 * requests_per_variant * 3 for surface in SURFACES
            }
            assert len(upstream_requests) == len(jobs) * 3, (
                f"expected {len(jobs) * 3} upstream POSTs, observed {len(upstream_requests)}: {route_counts}"
            )
            assert route_counts == expected_route_counts, f"unexpected upstream routes: {route_counts}"
            assert all(not 200 <= response.status_code < 300 for response in responses)
            assert all("synthetic upstream failure" in response.text for response in responses)
            eventually(
                lambda: sink.attempts,
                lambda attempts: attempts > len(sink.store),
                seconds=30,
            )
            rejected: Final = sink.attempts - len(sink.store)
            assert rejected >= 1, f"expected at least one rejected sink upload, rejected={rejected}"
            assert len(sink.store) == 0, (
                f"expected all sink uploads to fail before recovery, rejected={rejected}, "
                f"attempts={sink.attempts}"
            )
            sink.fail_until = 0.0
            payloads: Final = eventually(
                lambda: sink.payloads(),
                lambda observed: len(observed) >= len(jobs),
                seconds=90,
                return_last_on_timeout=True,
            )
            payloads_after_window: Final = eventually(
                lambda: sink.payloads(),
                lambda observed: len(observed) > len(payloads),
                seconds=6,
                return_last_on_timeout=True,
            )
            payload_matches: Final = tuple(
                tuple(
                    index
                    for index, (response, job) in enumerate(zip(responses, jobs))
                    if _request_key(payload, response, job[2]) is not None
                )
                for payload in payloads_after_window
            )
            assert all(len(matches) == 1 for matches in payload_matches), (
                f"payloads could not be matched to one request: rejected={rejected}, "
                f"matches={payload_matches}, payloads={payloads_after_window}"
            )
            landed_keys: Final = tuple(
                _request_key(payload, responses[matches[0]], jobs[matches[0]][2])
                for payload, matches in zip(payloads_after_window, payload_matches)
            )
            duplicate_keys: Final = tuple(key for key in set(landed_keys) if landed_keys.count(key) > 1)
            expected_keys: Final = tuple(
                response.headers.get("x-litellm-call-id") or job[2] for response, job in zip(responses, jobs)
            )
            missing_keys: Final = tuple(key for key in expected_keys if key not in landed_keys)
            assert not duplicate_keys, (
                f"duplicate request ids landed: {duplicate_keys}; "
                f"missing={missing_keys}; rejected={rejected}; upstream_posts={len(upstream_requests)} "
                f"sink_payloads={len(payloads_after_window)}"
            )
            assert not missing_keys, (
                f"requests missing after sink recovery: {missing_keys}; "
                f"rejected={rejected}; upstream_posts={len(upstream_requests)} "
                f"sink_payloads={len(payloads_after_window)}"
            )

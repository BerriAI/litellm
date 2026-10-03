import threading
import uuid
from pathlib import Path
from typing import Final

import httpx
from _langfuse_otel import (
    _drained_spans,
    _generation_marker_span_attributes,
    _generation_span_attributes,
    _langfuse_proxy,
    _marker_from_body,
    _sink,
    _span_containing_marker,
    _upstream_reply_for,
)
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, Wire, wire_server


def _identity(attributes: dict[str, object]) -> dict[str, object]:
    return {key: attributes.get(key) for key in ("user.id", "session.id")}


def _await_generation(collector: Wire, batches: list[bytes], marker: str) -> dict[str, object]:
    return eventually(
        lambda: _generation_span_attributes(collector, batches, marker),
        lambda spans: len(spans) == 1,
        seconds=30,
    )[0]


def _end_user_request(
    candidate: Gateway, model: str, marker: str, key: str | None = None, **extra: object
) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": marker}],
            "cache": {"no-cache": True},
            **extra,
        },
        key=key,
        headers={"x-litellm-end-user-id": f"end-user-{marker}"},
    )


def _subsequent_request_still_lands(candidate: Gateway, collector: Wire, batches: list[bytes], model: str) -> None:
    next_marker: Final = uuid.uuid4().hex
    response: Final = _end_user_request(candidate, model, next_marker)
    assert response.status_code == 200, response.text
    assert _await_generation(collector, batches, next_marker) is not None


def test_langfuse_otel_five_kb_end_user_header_lands_in_user_id(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    end_user: Final = "e" * 5120
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            headers={"x-litellm-end-user-id": end_user},
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": end_user, "session.id": None}, attributes


def test_langfuse_otel_empty_end_user_header_writes_no_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            headers={"x-litellm-end-user-id": ""},
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": None, "session.id": None}, attributes
        _subsequent_request_still_lands(candidate, collector, batches, model)


def test_langfuse_otel_integer_body_user_does_not_crash(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "user": 123,
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert attributes.get("session.id") is None, attributes
        _subsequent_request_still_lands(candidate, collector, batches, model)


def test_langfuse_otel_list_body_user_does_not_crash(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "user": ["a", "b"],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert attributes.get("session.id") is None, attributes
        _subsequent_request_still_lands(candidate, collector, batches, model)


def test_langfuse_otel_integer_trace_user_id_does_not_crash(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = _end_user_request(candidate, model, marker, metadata={"trace_user_id": 123})
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert attributes.get("user.id") in ("123", f"end-user-{marker}"), attributes
        assert attributes.get("session.id") is None, attributes
        _subsequent_request_still_lands(candidate, collector, batches, model)


def test_langfuse_otel_null_metadata_still_maps_the_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = _end_user_request(candidate, model, marker, metadata=None)
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


def test_langfuse_otel_duplicate_end_user_header_uses_the_first(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.client.request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"first-call-{marker}"}],
                "cache": {"no-cache": True},
            },
            headers={
                "Authorization": f"Bearer {candidate.key}",
                "x-litellm-end-user-id": f"first-{marker}",
            },
        )
        assert response.status_code == 200, response.text
        duplicate: Final = candidate.client.request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            headers=[
                ("Authorization", f"Bearer {candidate.key}"),
                ("x-litellm-end-user-id", f"first-{marker}"),
                ("x-litellm-end-user-id", f"second-{marker}"),
            ],
        )
        assert duplicate.status_code == 200, duplicate.text
        batches: Final[list[bytes]] = []
        spans: Final = eventually(
            lambda: _generation_marker_span_attributes(collector, batches, marker),
            lambda found: len(found) == 2,
            seconds=30,
        )
        assert all(attributes.get("user.id") == f"first-{marker}" for attributes in spans), spans


def test_langfuse_otel_bad_key_emits_no_generation_span(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            key="sk-not-a-real-key",
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 401, response.text
        batches: Final[list[bytes]] = []
        _subsequent_request_still_lands(candidate, collector, batches, model)
        offending: Final = tuple(
            attributes
            for attributes in _generation_marker_span_attributes(collector, batches, marker)
            if attributes.get("user.id") not in (None, f"end-user-{marker}")
        )
        assert offending == (), offending


def test_langfuse_otel_upstream_failure_never_invents_an_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(
            lambda request: (
                Reply(status=401, body=b'{"error": "upstream denied ' + marker.encode() + b'"}')
                if marker.encode() in (request.body or b"")
                else _upstream_reply_for(request, _marker_from_body(request))
            )
        ) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = _end_user_request(candidate, model, marker)
        assert response.status_code == 401, response.text
        batches: Final[list[bytes]] = []
        _subsequent_request_still_lands(candidate, collector, batches, model)
        offending: Final = tuple(
            attributes
            for _t, attributes in _span_containing_marker(_drained_spans(collector, batches), marker)
            if attributes.get("user.id") not in (None, f"end-user-{marker}")
        )
        assert offending == (), offending


def test_langfuse_otel_sink_rejections_do_not_drop_the_proxy(gateway: Gateway, tmp_path: Path) -> None:
    calls: Final[dict[str, int]] = {"count": 0}
    lock: Final = threading.Lock()

    def rejecting_sink(request: Request) -> Reply:
        with lock:
            calls["count"] += 1
            seen: Final = calls["count"]
        if seen == 1:
            return Reply(status=403)
        if seen == 2:
            return Reply(status=404)
        return _sink(request)

    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(rejecting_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        markers: Final = tuple(uuid.uuid4().hex for _ in range(3))
        batches: Final[list[bytes]] = []
        for index, item in enumerate(markers):
            response: Final = _end_user_request(candidate, model, item)
            assert response.status_code == 200, response.text
            if index < 2:
                eventually(
                    lambda collector=collector, batches=batches: (
                        batches.extend(request.body for request in collector.drain()) or len(batches)
                    ),
                    lambda seen, index=index: seen >= index + 1,
                    seconds=30,
                )
        attributes: Final = _await_generation(collector, batches, markers[2])
        assert attributes.get("user.id") == f"end-user-{markers[2]}", attributes
        _subsequent_request_still_lands(candidate, collector, batches, model)


def test_langfuse_otel_unknown_model_error_reaches_the_caller(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = _end_user_request(candidate, "no-such-model-" + marker[:12], marker)
        assert response.status_code in (400, 404), response.text
        batches: Final[list[bytes]] = []
        _subsequent_request_still_lands(candidate, collector, batches, model)


def test_langfuse_otel_empty_and_null_session_id_stay_absent(gateway: Gateway, tmp_path: Path) -> None:
    markers: Final = tuple(uuid.uuid4().hex for _ in range(3))
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        batches: Final[list[bytes]] = []
        sessions: Final = ({"session_id": ""}, {"session_id": None}, {})
        for marker, metadata in zip(markers, sessions):
            response: Final = _end_user_request(candidate, model, marker, metadata=metadata)
            assert response.status_code == 200, response.text
        for marker, metadata in zip(markers, sessions):
            attributes: Final = _await_generation(collector, batches, marker)
            assert _identity(attributes) == {
                "user.id": f"end-user-{marker}",
                "session.id": metadata.get("session_id"),
            }, attributes


def test_langfuse_otel_empty_trace_user_id_falls_back_to_the_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = _end_user_request(candidate, model, marker, metadata={"trace_user_id": ""})
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


def test_langfuse_otel_three_identical_requests_each_land_the_end_user(gateway: Gateway, tmp_path: Path) -> None:
    markers: Final = tuple(uuid.uuid4().hex for _ in range(3))
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        batches: Final[list[bytes]] = []
        for marker in markers:
            response: Final = _end_user_request(candidate, model, marker)
            assert response.status_code == 200, response.text
        for marker in markers:
            attributes: Final = _await_generation(collector, batches, marker)
            assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


def test_langfuse_otel_litellm_metadata_session_id_on_responses(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": marker,
                "litellm_metadata": {"session_id": f"sess-{marker}"},
                "cache": {"no-cache": True},
            },
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_marker_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": f"sess-{marker}"}, attributes

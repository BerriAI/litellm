import uuid
from pathlib import Path
from typing import Final

from _langfuse_otel import (
    _generation_span_attributes,
    _langfuse_proxy,
    _sink,
    _span_attributes_containing_marker,
    _trace_user_span_attributes,
    _upstream_reply,
)
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, wire_server


def test_langfuse_otel_header_end_user_lands_in_user_id_not_session_id_for_a_key_owned_by_an_internal_user(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            key=key,
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": None,
        }, attributes


def test_langfuse_otel_header_end_user_lands_in_user_id_for_a_service_account_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key()
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            key=key,
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": None,
        }, attributes


def test_langfuse_otel_body_user_is_never_a_session(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
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
                "user": f"end-user-{marker}",
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": None,
        }, attributes


def test_langfuse_otel_caller_trace_user_id_wins_over_the_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
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
                "user": f"end-user-{marker}",
                "metadata": {"trace_user_id": f"caller-{marker}"},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"caller-{marker}",
            "session.id": None,
        }, attributes


def test_langfuse_otel_caller_session_id_stays_the_session_beside_the_end_user(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
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
                "user": f"end-user-{marker}",
                "metadata": {"session_id": f"sess-{marker}"},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": f"sess-{marker}",
        }, attributes


def test_langfuse_otel_v2_header_end_user_lands_in_user_id(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url, {"LITELLM_OTEL_V2": "1"}) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            key=key,
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        user_spans: Final = eventually(
            lambda: _trace_user_span_attributes(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )
        assert {key: user_spans[0].get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"end-user-{marker}",
            "session.id": None,
        }, user_spans[0]


def test_langfuse_otel_messages_caller_trace_user_id_under_litellm_metadata_wins_over_the_end_user(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex
    upstream_bodies: Final[list[bytes]] = []

    def upstream(request: Request) -> Reply:
        upstream_bodies.append(request.body)
        return _upstream_reply(marker)

    with (
        wire_server(upstream) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "max_tokens": 5,
                "litellm_metadata": {"trace_user_id": f"caller-{marker}"},
                "cache": {"no-cache": True},
            },
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        assert any(marker.encode() in body for body in upstream_bodies), upstream_bodies
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _span_attributes_containing_marker(collector, batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert {key: attributes.get(key) for key in ("user.id", "session.id")} == {
            "user.id": f"caller-{marker}",
            "session.id": None,
        }, attributes

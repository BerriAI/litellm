import uuid
from pathlib import Path
from typing import Final

from _langfuse_otel import (
    _arize_generation_span_attributes,
    _generation_marker_span_attributes,
    _generation_span_attributes,
    _langfuse_proxy,
    _observability_proxy,
    _sink,
    _trace_user_span_attributes,
    _upstream_reply_for,
)
from anthropic import Anthropic, AsyncAnthropic
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Wire, wire_server
from openai import AsyncOpenAI, OpenAI


def _openai_client(gateway: Gateway, key: str, marker: str) -> OpenAI:
    return OpenAI(
        api_key=key,
        base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
        timeout=30,
        max_retries=0,
        default_headers={"x-litellm-end-user-id": f"end-user-{marker}"},
    )


def _async_openai_client(gateway: Gateway, key: str, marker: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=key,
        base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
        timeout=30,
        max_retries=0,
        default_headers={"x-litellm-end-user-id": f"end-user-{marker}"},
    )


def _anthropic_client(gateway: Gateway, key: str, marker: str) -> Anthropic:
    return Anthropic(
        api_key=key,
        base_url=str(gateway.client.base_url).rstrip("/"),
        timeout=30,
        max_retries=0,
        default_headers={"x-litellm-end-user-id": f"end-user-{marker}"},
    )


def _async_anthropic_client(gateway: Gateway, key: str, marker: str) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=key,
        base_url=str(gateway.client.base_url).rstrip("/"),
        timeout=30,
        max_retries=0,
        default_headers={"x-litellm-end-user-id": f"end-user-{marker}"},
    )


def _await_generation(collector: Wire, batches: list[bytes], marker: str) -> dict[str, object]:
    return eventually(
        lambda: _generation_span_attributes(collector, batches, marker),
        lambda spans: len(spans) == 1,
        seconds=30,
    )[0]


def _await_marker_span(collector: Wire, batches: list[bytes], marker: str) -> dict[str, object]:
    return eventually(
        lambda: _generation_marker_span_attributes(collector, batches, marker),
        lambda spans: len(spans) == 1,
        seconds=30,
    )[0]


def _identity(attributes: dict[str, object]) -> dict[str, object]:
    return {key: attributes.get(key) for key in ("user.id", "session.id")}


def test_langfuse_otel_customer_id_header_lands_in_user_id(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            headers={"x-litellm-customer-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


def test_langfuse_otel_langfuse_trace_user_id_header_wins_over_the_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            headers={
                "x-litellm-end-user-id": f"end-user-{marker}",
                "langfuse_trace_user_id": f"caller-{marker}",
            },
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"caller-{marker}", "session.id": None}, attributes


def test_langfuse_otel_no_end_user_never_exposes_the_internal_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
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
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": None, "session.id": None}, attributes


def test_langfuse_otel_team_key_end_user_keeps_the_litellm_attributes(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(team_alias=f"team-alias-{marker[:12]}")
        key: Final = scenario.key(team_id=team, key_alias=f"key-alias-{marker[:12]}")
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            key=key,
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes
        assert attributes.get("litellm.team_id") == team, attributes
        assert attributes.get("litellm.team_alias") == f"team-alias-{marker[:12]}", attributes
        assert attributes.get("litellm.key_alias") == f"key-alias-{marker[:12]}", attributes


def test_langfuse_otel_header_end_user_beats_the_body_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
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
                "user": f"body-user-{marker}",
                "cache": {"no-cache": True},
            },
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT end_user FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (str(response.json()["id"]),)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes
        assert attributes.get("user.id") == rows[0]["end_user"], (attributes, rows)


def test_langfuse_otel_openai_sdk_streaming_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        with _openai_client(candidate, key, marker) as client:
            stream: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                stream=True,
                extra_body={"cache": {"no-cache": True}},
            )
            with stream:
                chunks: Final = tuple(stream)
            assert {chunk.id for chunk in chunks} == {marker}
            assert f"reply {marker}" in "".join(
                choice.delta.content or "" for chunk in chunks for choice in chunk.choices
            )
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


async def test_langfuse_otel_openai_async_sdk_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        async with _async_openai_client(candidate, key, marker) as client:
            completion: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                extra_body={"cache": {"no-cache": True}},
            )
            assert completion.id == marker
        batches: Final[list[bytes]] = []
        attributes: Final = _await_generation(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


def test_langfuse_otel_responses_api_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        with _openai_client(candidate, key, marker) as client:
            completion: Final = client.responses.create(
                model=model, input=marker, extra_body={"cache": {"no-cache": True}}
            )
            assert marker in completion.output_text, completion.output_text
        batches: Final[list[bytes]] = []
        attributes: Final = _await_marker_span(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


async def test_langfuse_otel_responses_api_async_streaming_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        async with _async_openai_client(candidate, key, marker) as client:
            stream: Final = await client.responses.create(
                model=model, input=marker, stream=True, extra_body={"cache": {"no-cache": True}}
            )
            events: Final = tuple([event async for event in stream])
            assert events, events
        batches: Final[list[bytes]] = []
        attributes: Final = _await_marker_span(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


def test_langfuse_otel_anthropic_sdk_messages_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        with _anthropic_client(candidate, key, marker) as client:
            message: Final = client.messages.create(
                model=model, max_tokens=5, messages=[{"role": "user", "content": marker}]
            )
            assert marker in message.content[0].text, message.model_dump_json()
        batches: Final[list[bytes]] = []
        attributes: Final = _await_marker_span(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


async def test_langfuse_otel_anthropic_async_streaming_messages_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        owner: Final = scenario.user()
        key: Final = scenario.key(user_id=owner)
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        async with _async_anthropic_client(candidate, key, marker) as client:
            stream: Final = await client.messages.create(
                model=model, max_tokens=5, messages=[{"role": "user", "content": marker}], stream=True
            )
            events: Final = tuple([event async for event in stream])
            assert events, events
        batches: Final[list[bytes]] = []
        attributes: Final = _await_marker_span(collector, batches, marker)
        assert _identity(attributes) == {"user.id": f"end-user-{marker}", "session.id": None}, attributes


def test_langfuse_otel_v2_caller_trace_user_id_wins_over_the_end_user(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
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
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "metadata": {"trace_user_id": f"caller-{marker}"},
                "cache": {"no-cache": True},
            },
            key=key,
            headers={"x-litellm-end-user-id": f"end-user-{marker}"},
        )
        assert response.status_code == 200, response.text
        batches: Final[list[bytes]] = []
        user_spans: Final = eventually(
            lambda: _trace_user_span_attributes(collector, batches, marker),
            lambda spans: len(spans) >= 1,
            seconds=30,
        )
        assert all(
            _identity(attributes) == {"user.id": f"caller-{marker}", "session.id": None} for attributes in user_spans
        ), user_spans


def test_arize_phoenix_header_end_user_keeps_session_mapping(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _observability_proxy(
            gateway,
            tmp_path,
            {
                "PHOENIX_COLLECTOR_ENDPOINT": collector.url + "/v1/traces",
                "PHOENIX_API_KEY": "phoenix-integration",
            },
            callbacks=("arize_phoenix",),
            config_name="arize_phoenix.yaml",
        ) as candidate,
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
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _arize_generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) >= 1,
            seconds=30,
        )[0]
        assert _identity(attributes) == {"user.id": owner, "session.id": f"end-user-{marker}"}, attributes
        assert attributes.get("litellm.trace_id") is not None, attributes


def test_arize_phoenix_caller_session_id_stays_session(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as collector,
        _observability_proxy(
            gateway,
            tmp_path,
            {
                "PHOENIX_COLLECTOR_ENDPOINT": collector.url + "/v1/traces",
                "PHOENIX_API_KEY": "phoenix-integration",
            },
            callbacks=("arize_phoenix",),
            config_name="arize_phoenix.yaml",
        ) as candidate,
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
        batches: Final[list[bytes]] = []
        attributes: Final = eventually(
            lambda: _arize_generation_span_attributes(collector, batches, marker),
            lambda spans: len(spans) >= 1,
            seconds=30,
        )[0]
        assert _identity(attributes) == {
            "user.id": f"end-user-{marker}",
            "session.id": f"end-user-{marker}",
        }, attributes
        assert attributes.get("litellm.trace_id") == f"sess-{marker}", attributes


def test_langfuse_otel_and_arize_phoenix_together_keep_each_mapping(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as langfuse_collector,
        wire_server(_sink) as phoenix_collector,
        _observability_proxy(
            gateway,
            tmp_path,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-integration",
                "LANGFUSE_SECRET_KEY": "sk-integration",
                "LANGFUSE_HOST": langfuse_collector.url,
                "PHOENIX_COLLECTOR_ENDPOINT": phoenix_collector.url + "/v1/traces",
                "PHOENIX_API_KEY": "phoenix-integration",
            },
            callbacks=("langfuse_otel", "arize_phoenix"),
            config_name="dual_callbacks.yaml",
        ) as candidate,
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
        langfuse_batches: Final[list[bytes]] = []
        phoenix_batches: Final[list[bytes]] = []
        langfuse_attributes: Final = _await_generation(langfuse_collector, langfuse_batches, marker)
        phoenix_attributes: Final = eventually(
            lambda: _arize_generation_span_attributes(phoenix_collector, phoenix_batches, marker),
            lambda spans: len(spans) >= 1,
            seconds=30,
        )[0]
        assert _identity(langfuse_attributes) == {
            "user.id": f"end-user-{marker}",
            "session.id": None,
        }, langfuse_attributes
        assert _identity(phoenix_attributes) == {
            "user.id": owner,
            "session.id": f"end-user-{marker}",
        }, phoenix_attributes

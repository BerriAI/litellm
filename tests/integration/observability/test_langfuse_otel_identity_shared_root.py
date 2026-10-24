import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import pytest
from _langfuse_otel import (
    _generation_span_attributes,
    _marker_trace_root_attributes,
    _observability_proxy,
    _sink,
    _upstream_reply_for,
)
from integration._support.client import Gateway, eventually
from integration._support.wire import wire_server


def _generic_otel_env(collector_url: str) -> Mapping[str, str]:
    return {"OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": collector_url + "/v1/traces"}


def _phoenix_env(collector_url: str) -> Mapping[str, str]:
    return {"PHOENIX_COLLECTOR_ENDPOINT": collector_url + "/v1/traces", "PHOENIX_API_KEY": "phoenix-integration"}


@pytest.mark.parametrize(
    ("first_callback", "first_env"),
    [("otel", _generic_otel_env), ("arize_phoenix", _phoenix_env)],
    ids=["generic-otel-first", "arize-phoenix-first"],
)
def test_langfuse_otel_v2_keeps_the_end_user_off_another_callbacks_root(
    gateway: Gateway, tmp_path: Path, first_callback: str, first_env: Callable[[str], Mapping[str, str]]
) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _upstream_reply_for(request, marker)) as provider,
        wire_server(_sink) as first_collector,
        wire_server(_sink) as langfuse_collector,
        _observability_proxy(
            gateway,
            tmp_path,
            {
                "LITELLM_OTEL_V2": "1",
                "LANGFUSE_PUBLIC_KEY": "pk-integration",
                "LANGFUSE_SECRET_KEY": "sk-integration",
                "LANGFUSE_HOST": langfuse_collector.url,
                **first_env(first_collector.url),
            },
            callbacks=(first_callback, "langfuse_otel"),
            config_name="shared_root.yaml",
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
        first_batches: Final[list[bytes]] = []
        roots: Final = eventually(
            lambda: _marker_trace_root_attributes(first_collector, first_batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )
        assert roots[0].get("http.route") == "/v1/chat/completions", roots
        assert "user.id" not in roots[0], roots
        langfuse_batches: Final[list[bytes]] = []
        generation: Final = eventually(
            lambda: _generation_span_attributes(langfuse_collector, langfuse_batches, marker),
            lambda spans: len(spans) == 1,
            seconds=30,
        )[0]
        assert generation.get("user.id") == f"end-user-{marker}", generation

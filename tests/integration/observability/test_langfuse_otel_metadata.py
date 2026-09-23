import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


def _span_attributes(body: bytes) -> tuple[dict[str, object], ...]:
    request: Final = ExportTraceServiceRequest.FromString(body)
    return tuple(
        {attribute.key: getattr(attribute.value, attribute.value.WhichOneof("value")) for attribute in span.attributes}
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )


@contextmanager
def _langfuse_rig(
    gateway: Gateway,
    tmp_path: Path,
    settings: Mapping[str, object],
    marker: str,
) -> Iterator[tuple[Gateway, Wire, Wire]]:
    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"data":[]}')
        assert request.target.endswith("/chat/completions"), request.target
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{marker}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "langfuse-echo"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                }
            ).encode()
        )

    def sink(_request: Request) -> Reply:
        return Reply()

    with wire_server(upstream) as provider, wire_server(sink) as collector:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"].update({"callbacks": ["langfuse_otel"], **settings})
        path: Final = tmp_path / "langfuse_otel.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(
            gateway,
            tmp_path,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-lf-integration",
                "LANGFUSE_SECRET_KEY": "sk-lf-integration",
                "LANGFUSE_HOST": collector.url,
            },
            config=path,
            workers=2,
        ) as candidate:
            yield candidate, provider, collector


def _observation_span(collector: Wire, response_id: str) -> dict[str, object]:
    batches: Final = []

    def spans() -> tuple[dict[str, object], ...]:
        batches.extend(collector.drain())
        return tuple(
            attributes
            for batch in batches
            for attributes in _span_attributes(batch.body)
            if attributes.get("llm.response.id") == response_id
        )

    observed: Final = eventually(spans, lambda values: len(values) == 1, seconds=25)
    return observed[0]


@pytest.mark.covers("other.observability.langfuse_otel.request_metadata_on_langfuse_keys")
def test_langfuse_otel_emits_request_metadata_under_langfuse_observation_and_trace_keys(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-meta-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {}, marker) as (candidate, provider, collector),
        candidate.scenario() as scenario,
    ):
        team_alias: Final = "lf-team-" + uuid.uuid4().hex
        team_id: Final = scenario.team(team_alias=team_alias)
        key_alias: Final = "lf-key-" + uuid.uuid4().hex
        key: Final = scenario.key(
            team_id=team_id,
            key_alias=key_alias,
            metadata={"spend_logs_metadata": {"ticket": "LIT-8283"}},
        )
        model: Final = scenario.model(api_base=provider.url + "/v1")
        end_user: Final = "lf-end-user-" + uuid.uuid4().hex
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "user": end_user,
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "langfuse-echo"

        attrs: Final = _observation_span(collector, response.json()["id"])
        assert "metadata" in attrs, sorted(attrs)
        assert json.loads(str(attrs["metadata"]))["user_api_key_alias"] == key_alias
        assert "langfuse.observation.metadata" in attrs, sorted(attrs)
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        expected: Final = {
            "user_api_key_alias": key_alias,
            "user_api_key_team_id": team_id,
            "user_api_key_team_alias": team_alias,
            "user_api_key_end_user_id": end_user,
        }
        assert {key_: observation.get(key_) for key_ in expected} == expected, observation
        assert observation["spend_logs_metadata"] == {"ticket": "LIT-8283"}, observation
        assert {
            attribute: attrs.get(attribute)
            for attribute in (
                "langfuse.trace.metadata.user_api_key_alias",
                "langfuse.trace.metadata.user_api_key_team_id",
                "langfuse.trace.metadata.user_api_key_team_alias",
                "langfuse.trace.metadata.user_api_key_end_user_id",
            )
        } == {
            "langfuse.trace.metadata.user_api_key_alias": key_alias,
            "langfuse.trace.metadata.user_api_key_team_id": team_id,
            "langfuse.trace.metadata.user_api_key_team_alias": team_alias,
            "langfuse.trace.metadata.user_api_key_end_user_id": end_user,
        }, attrs


@pytest.mark.covers("other.observability.langfuse_otel.request_metadata_redaction_matches_vanilla")
def test_langfuse_otel_metadata_redacts_user_api_key_fields_like_vanilla_langfuse(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-redact-" + uuid.uuid4().hex
    with (
        _langfuse_rig(gateway, tmp_path, {"redact_user_api_key_info": True}, marker) as (
            candidate,
            provider,
            collector,
        ),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(
            key_alias="lf-redact-" + uuid.uuid4().hex,
            metadata={"spend_logs_metadata": {"ticket": "LIT-8283"}},
        )
        model: Final = scenario.model(api_base=provider.url + "/v1")
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text

        attrs: Final = _observation_span(collector, response.json()["id"])
        assert "langfuse.observation.metadata" in attrs, sorted(attrs)
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        assert not [name for name in observation if name.startswith("user_api_key")], observation
        assert observation["spend_logs_metadata"] == {"ticket": "LIT-8283"}, observation
        assert not [name for name in attrs if name.startswith("langfuse.trace.metadata.")], sorted(attrs)

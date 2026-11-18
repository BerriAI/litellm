from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.openai_wire import chat_reply
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span


def _completion_spans(request: Request, identity: str) -> Iterator[tuple[Request, Span]]:
    if not request.target.endswith("/v1/traces"):
        return
    batch: Final = ExportTraceServiceRequest.FromString(request.body)
    for resource in batch.resource_spans:
        for scope in resource.scope_spans:
            for span in scope.spans:
                if any(
                    attr.key == "gen_ai.response.id" and attr.value.string_value == identity for attr in span.attributes
                ):
                    yield request, span


def test_wildedge_receives_authenticated_completion_trace(tmp_path: Path, gateway: Gateway) -> None:
    model: Final = "wildedge-test-model"
    identity: Final = "chatcmpl-" + uuid4().hex
    secret: Final = "project,token%2F=" + uuid4().hex
    with (
        wire_server(lambda _: Reply(body=b"", content_type="application/x-protobuf")) as collector,
        wire_server(lambda _: chat_reply(identity, model, "ok", stream=False)) as upstream,
    ):
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": model,
                            "litellm_params": {
                                "model": f"openai/{model}",
                                "api_base": upstream.url + "/v1",
                                "api_key": "synthetic",
                                "input_cost_per_token": 0.001,
                                "output_cost_per_token": 0.002,
                            },
                        }
                    ],
                    "litellm_settings": {"callbacks": ["wildedge"]},
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "coordination_redis": {"host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
                    },
                    "router_settings": {"redis_host": "os.environ/REDIS_HOST", "redis_port": "os.environ/REDIS_PORT"},
                }
            )
        )
        environment: Final = {
            "LITELLM_OTEL_V2": "true",
            "WILDEDGE_PROJECT_SECRET": secret,
            "WILDEDGE_INGEST_URL": collector.url + "/api/otlp/",
        }
        with owned_proxy(gateway, tmp_path, environment, config=config, workers=2, database_setup=()) as proxy:
            completion: Final = proxy.post(
                "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Hello"}]}
            )
            assert completion["id"] == identity

            def delivered() -> tuple[tuple[Request, Span], ...]:
                return tuple(chain.from_iterable(_completion_spans(request, identity) for request in collector.drain()))

            ((batch, span),) = eventually(delivered, bool, seconds=30)
            assert batch.target == "/api/otlp/v1/traces"
            assert batch.headers["content-type"] == "application/x-protobuf"
            assert batch.headers["x-project-secret"] == secret
            assert 0 < span.start_time_unix_nano < span.end_time_unix_nano
            usage: Final = object_value(completion["usage"])
            attributes: Final = {attr.key: attr.value for attr in span.attributes}
            assert attributes["gen_ai.response.model"].string_value == model
            assert attributes["gen_ai.usage.input_tokens"].int_value == usage["prompt_tokens"]
            assert attributes["gen_ai.usage.output_tokens"].int_value == usage["completion_tokens"]
            assert attributes["litellm.cost.total"].double_value == pytest.approx(0.011)

import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml

from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


def _span_attributes(body: bytes) -> tuple[dict[str, object], ...]:
    return tuple(
        {attribute["key"]: attribute["value"] for attribute in span.get("attributes", ())}
        for resource in json.loads(body)["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    )


@pytest.mark.covers("other.observability.otel.text_completion_choices_keep_provider_fields")
def test_otel_weave_output_keeps_text_completion_provider_fields_beside_the_synthesized_message(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "otel-text-" + uuid.uuid4().hex
    logprobs: Final = {
        "tokens": ["Hello", " there"],
        "token_logprobs": [-0.1, -0.2],
        "top_logprobs": None,
        "text_offset": [0, 5],
    }
    content_filter: Final = {"hate": {"filtered": False, "severity": "safe"}}

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/completions"), request.target
        return Reply(
            body=json.dumps(
                {
                    "id": marker,
                    "object": "text_completion",
                    "created": 1,
                    "model": "gpt-3.5-turbo-instruct",
                    "choices": [
                        {
                            "index": 0,
                            "text": "Hello there",
                            "finish_reason": "stop",
                            "logprobs": logprobs,
                            "content_filter_results": content_filter,
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
        config["litellm_settings"].update({"callbacks": ["otel"]})
        config["callback_settings"] = {
            "otel": {
                "exporter": "http/json",
                "endpoint": collector.url,
                "mapper_names": ["genai", "openinference", "weave"],
                "capture_message_content": "span_only",
                "use_simple_processor": True,
            }
        }
        path: Final = tmp_path / "otel.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy(gateway, tmp_path, {"LITELLM_OTEL_V2": "1"}, config=path) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model="openai/gpt-3.5-turbo-instruct", api_base=provider.url + "/v1")
            response: Final = candidate.request(
                "POST", "/v1/completions", {"model": model, "prompt": marker, "cache": {"no-cache": True}}
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["text"] == "Hello there"
            batches = []

            def outputs() -> tuple[list[dict[str, object]], ...]:
                batches.extend(collector.drain())
                return tuple(
                    json.loads(attributes["weave.output"]["stringValue"])
                    for batch in batches
                    for attributes in _span_attributes(batch.body)
                    if "weave.output" in attributes
                    and attributes.get("gen_ai.response.id", {}).get("stringValue") == marker
                )

            choices: Final = eventually(outputs, lambda values: len(values) == 1, seconds=20)[0]
            assert len(choices) == 1, choices
            choice: Final = choices[0]
            assert choice["message"]["content"] == "Hello there", choice
            assert "text" not in choice, choice
            assert {
                key: choice.get(key) for key in ("index", "finish_reason", "logprobs", "content_filter_results")
            } == {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": logprobs,
                "content_filter_results": content_filter,
            }, choice

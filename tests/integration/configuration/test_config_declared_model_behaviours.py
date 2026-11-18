from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.provider import PROVIDER_URL, SharedProvider
from tests.integration._support.wire import Reply

_TAGGED_GROUP: Final = "tag-filtered-group"
_TAGGED_IDS: Final = frozenset({"tag-filtered-team-a", "tag-filtered-team-b"})
_VISION_MODEL: Final = "llava-hf"
_ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _config(directory: Path) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        *(
            {
                "model_name": _TAGGED_GROUP,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-fixture",
                    "api_base": f"{PROVIDER_URL}/v1",
                    "tags": [tag],
                },
                "model_info": {"id": identity},
            }
            for tag, identity in (("teamA", "tag-filtered-team-a"), ("teamB", "tag-filtered-team-b"))
        ),
        {
            "model_name": _VISION_MODEL,
            "litellm_params": {
                "model": "openai/llava-hf",
                "api_key": "sk-fixture",
                "api_base": "http://127.0.0.1:9/v1",
            },
            "model_info": {"supports_vision": True},
        },
    ]
    configuration.setdefault("router_settings", {})["enable_tag_filtering"] = True
    path: Final = directory / "config-declared-models.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def declared(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("config-declared-models")
    with (
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"OPENAI_API_KEY": "sk-fixture", "OPENAI_BASE_URL": f"{PROVIDER_URL}/v1", "GCS_FLUSH_INTERVAL": "1"},
            config=_config(directory),
            remove_environment=("GCS_BUCKET_NAME", "OPENAI_API_BASE"),
        ) as owned,
    ):
        yield owned


def test_model_info_reports_the_vision_capability_declared_in_the_config(declared: Gateway) -> None:
    listing: Final = _ITEMS.validate_python(declared.get("/model/info")["data"])
    vision: Final = [item for item in listing if item["model_name"] == _VISION_MODEL]
    assert len(vision) == 1, [item["model_name"] for item in listing]
    assert object_value(vision[0]["model_info"])["supports_vision"] is True, vision[0]


def test_an_untagged_request_is_served_by_a_group_whose_deployments_are_all_tagged(
    declared: Gateway, provider: SharedProvider
) -> None:
    provider.expect(
        Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "tagged"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            ).encode()
        )
    )
    response: Final = declared.request(
        "POST",
        "/v1/chat/completions",
        {"model": _TAGGED_GROUP, "messages": [{"role": "user", "content": f"untagged {uuid.uuid4().hex}"}]},
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] in _TAGGED_IDS, dict(response.headers)
    message: Final = object_value(object_value(JSON_OBJECT.validate_json(response.content)["choices"][0])["message"])
    assert message["content"] == "tagged"
    assert [request.target for request in provider.received()] == ["/v1/chat/completions"]


def test_a_moderation_request_without_a_model_reaches_the_provider_default(
    declared: Gateway, provider: SharedProvider
) -> None:
    provider.expect(
        Reply(
            body=json.dumps(
                {
                    "id": f"modr-{uuid.uuid4().hex}",
                    "model": "omni-moderation-latest",
                    "results": [
                        {"flagged": True, "categories": {"violence": True}, "category_scores": {"violence": 0.9}}
                    ],
                }
            ).encode()
        )
    )
    phrase: Final = f"I want to harm someone {uuid.uuid4().hex}"
    response: Final = declared.request("POST", "/moderations", {"input": phrase})
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert body["model"] == "omni-moderation-latest", body
    assert object_value(_ITEMS.validate_python(body["results"])[0])["flagged"] is True, body
    sent: Final = provider.received()
    assert [request.target for request in sent] == ["/v1/moderations"]
    payload: Final = JSON_OBJECT.validate_json(sent[0].body)
    assert payload == {"input": phrase}, payload


def test_key_health_reports_an_unconfigured_key_logging_callback_as_unhealthy(declared: Gateway) -> None:
    with declared.scenario() as scenario:
        key: Final = scenario.key(
            metadata={
                "logging": [
                    {
                        "callback_name": "gcs_bucket",
                        "callback_type": "success_and_failure",
                        "callback_vars": {
                            "gcs_bucket_name": "key-logging-project1",
                            "gcs_path_service_account": "bad-service-account",
                        },
                    }
                ]
            }
        )
        health: Final = declared.request("POST", "/key/health", {}, key=key)
        assert health.status_code == 200, health.text
        body: Final = JSON_OBJECT.validate_json(health.content)
        assert "key" in body, body
        status: Final = object_value(body["logging_callbacks"])
        assert status["callbacks"] == ["gcs_bucket"], status
        assert status["status"] == "unhealthy", status
        assert "GCS_BUCKET_NAME is not set in the environment" in string_value(status["details"]), status

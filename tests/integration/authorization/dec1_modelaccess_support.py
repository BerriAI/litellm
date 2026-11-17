"""Shared helpers for the `test_dec1_modelaccess_*` model access contract files.

Every helper talks to the proxy over HTTP like a customer would, and reads Postgres or the scripted upstream
only to observe an effect. Nothing here patches or imports proxy internals. The denial message and the
error types are copied from `litellm/proxy/auth/model_access_denied.py` and `litellm/proxy/_types.py`
(`ProxyErrorTypes`) so a rewrite that changes their wording fails loudly.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import httpx
import yaml
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value

CHAT_PATH: Final = "/v1/chat/completions"
MODEL_DENIED_TEMPLATE: Final = (
    "The requested model '{model}' is not available for this API key, or the model name is invalid. "
    "Check the models available to you and try again."
)
CUSTOMER_MODEL_DENIED_TEMPLATE: Final = (
    "The requested model '{model}' is not in the allowed models for this customer. "
    "Check the models this customer can use and try again."
)
KEY_DENIED: Final = "key_model_access_denied"
TEAM_DENIED: Final = "team_model_access_denied"
USER_DENIED: Final = "user_model_access_denied"
CUSTOMER_DENIED: Final = "customer_model_access_denied"


def marker() -> str:
    return f"dec1-modelaccess-{uuid.uuid4().hex}"


def unique_name(prefix: str = "dec1-ma") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def chat_body(model: str, text: str | None = None, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": text or marker()}], **extra}


def chat(gateway: Gateway, key: str, model: str, text: str | None = None, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", CHAT_PATH, chat_body(model, text, **extra), key=key)


def post_as(
    gateway: Gateway,
    key: str,
    path: str,
    body: Mapping[str, JsonValue],
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
) -> httpx.Response:
    return gateway.request("POST", path, dict(body), key=key, headers=headers, params=params)


def get_as(gateway: Gateway, key: str, path: str, params: Mapping[str, str] | None = None) -> httpx.Response:
    return gateway.request("GET", path, key=key, params=params)


def body_of(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(body_of(response)["error"])


def assert_served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return body_of(response)


def openai_model_denial(model: str, error_type: str) -> dict[str, JsonValue]:
    """The exact body every OpenAI shaped route answers for a model denial."""
    return {
        "error": {
            "message": MODEL_DENIED_TEMPLATE.format(model=model),
            "type": error_type,
            "param": "model",
            "code": "403",
        }
    }


def assert_model_denied(response: httpx.Response, model: str, error_type: str) -> None:
    assert response.status_code == 403, f"{response.status_code} {response.text}"
    assert body_of(response) == openai_model_denial(model, error_type), response.text


def assert_not_model_denied(response: httpx.Response) -> None:
    """The request passed every model access check, whatever the router did with it afterwards."""
    if response.status_code != 403:
        return
    error: Final = body_of(response).get("error")
    error_type: Final = object_value(error).get("type") if isinstance(error, dict) else None
    assert not (isinstance(error_type, str) and error_type.endswith("_model_access_denied")), response.text


def listed_ids(response: httpx.Response) -> list[str]:
    """`data[].id` of a `/v1/models` answer, in the order the proxy listed them."""
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return [string_value(object_value(entry)["id"]) for entry in list_value(body_of(response)["data"])]


def model_ids(gateway: Gateway, key: str, params: Mapping[str, str] | None = None) -> list[str]:
    return listed_ids(get_as(gateway, key, "/v1/models", params))


def model_info_names(gateway: Gateway, key: str, params: Mapping[str, str] | None = None) -> list[str]:
    response: Final = get_as(gateway, key, "/model/info", params)
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return [string_value(object_value(row)["model_name"]) for row in list_value(body_of(response)["data"])]


def model_group_rows(gateway: Gateway, key: str, model_group: str | None = None) -> list[dict[str, JsonValue]]:
    response: Final = get_as(
        gateway, key, "/model_group/info", {"model_group": model_group} if model_group is not None else None
    )
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return [object_value(row) for row in list_value(body_of(response)["data"])]


def model_group_names(gateway: Gateway, key: str) -> list[str]:
    return [string_value(row["model_group"]) for row in model_group_rows(gateway, key)]


def upstream_requests(gateway: Gateway, text: str) -> list[dict[str, JsonValue]]:
    """Requests the shared scripted upstream received whose body mentions `text`."""
    observed: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False)
    observed.raise_for_status()
    requests: Final = list_value(body_of(observed)["requests"])
    return [object_value(request) for request in requests if text in str(object_value(request)["body"])]


def upstream_models(gateway: Gateway, text: str) -> list[str]:
    return [string_value(object_value(request["body"])["model"]) for request in upstream_requests(gateway, text)]


def chat_reply(model: str, content: str = "dec1 model access reply") -> bytes:
    return JSON_OBJECT.dump_json(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )


def wire_deployment(name: str, provider_model: str, wire_url: str, **model_info: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"openai/{provider_model}",
            "api_key": "integration-provider-key",
            "api_base": f"{wire_url}/v1",
        },
        "model_info": dict(model_info),
    }


def owned_config(
    directory: Path,
    deployments: Sequence[Mapping[str, JsonValue]],
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> Path:
    """A minimal proxy config for an owned proxy whose only deployments are `deployments`."""
    config: Final = {
        "model_list": [dict(deployment) for deployment in deployments],
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "store_model_in_db": False,
            **(general_settings or {}),
        },
        "litellm_settings": {"cache": False, **(litellm_settings or {})},
        "router_settings": {"num_retries": 0, "disable_cooldowns": True, **(router_settings or {})},
    }
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / f"dec1-modelaccess-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def statuses(responses: Sequence[httpx.Response]) -> list[int]:
    return [response.status_code for response in responses]

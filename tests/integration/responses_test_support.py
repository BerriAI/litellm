from __future__ import annotations

import json
import os
import time
from typing import Final

import httpx
from integration._support.client import Gateway, eventually, object_value
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CONFIG_RELOAD_INTERVAL_SECONDS: Final = int(os.environ.get("PROXY_CONFIG_RELOAD_INTERVAL_SECONDS", "30"))


def model_discovery_reply(model: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"id": model, "object": "model", "created": 1, "owned_by": "openai"}],
            }
        ).encode()
    )


def drain_contract_requests(wire: Wire) -> tuple[Request, ...]:
    requests: Final = wire.drain()
    discovery_requests: Final = tuple(request for request in requests if request.target == "/v1/models")
    assert all(request.method == "GET" and request.body == b"" for request in discovery_requests), requests
    return tuple(request for request in requests if request.target != "/v1/models")


def _fresh_model_info(gateway: Gateway, model_id: str | None = None) -> httpx.Response:
    params: Final = () if model_id is None else (("litellm_model_id", model_id),)
    with httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False) as client:
        return client.get(
            "/model/info",
            params=params,
            headers={"Authorization": f"Bearer {gateway.key}"},
        )


def _deployment_id(entry: JsonValue) -> str | None:
    model_info: Final = object_value(object_value(entry).get("model_info"))
    identity: Final = model_info.get("id")
    return identity if isinstance(identity, str) else None


def _deployment_id_in_group(entry: JsonValue, model_group: str) -> str | None:
    model: Final = object_value(entry)
    if model.get("model_name") != model_group:
        return None
    return _deployment_id(entry)


def _deployment_ids_for_group(gateway: Gateway, model_group: str) -> tuple[str, ...]:
    response: Final = _fresh_model_info(gateway)
    if response.status_code != 200:
        return ()
    entries: Final = _JSON_OBJECT.validate_json(response.content).get("data")
    if not isinstance(entries, list):
        return ()
    return tuple(identity for entry in entries if (identity := _deployment_id_in_group(entry, model_group)) is not None)


def _deployment_is_loaded(gateway: Gateway, model_id: str) -> bool:
    response: Final = _fresh_model_info(gateway, model_id)
    if response.status_code != 200:
        return False
    entries: Final = _JSON_OBJECT.validate_json(response.content).get("data")
    return isinstance(entries, list) and any(_deployment_id(entry) == model_id for entry in entries)


def wait_for_model_group_workers(gateway: Gateway, model_group: str) -> None:
    model_ids: Final = eventually(
        lambda: _deployment_ids_for_group(gateway, model_group),
        bool,
        seconds=30,
    )
    worker_count: Final = max(1, int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1")))
    ready_after: Final = time.monotonic() + (_CONFIG_RELOAD_INTERVAL_SECONDS + 1 if worker_count > 1 else 0)
    eventually(
        lambda: (
            tuple(
                tuple(_deployment_is_loaded(gateway, model_id) for _ in range(worker_count)) for model_id in model_ids
            ),
            time.monotonic(),
        ),
        lambda observation: all(all(results) for results in observation[0]) and observation[1] >= ready_after,
        seconds=(_CONFIG_RELOAD_INTERVAL_SECONDS * 2) if worker_count > 1 else 30,
    )

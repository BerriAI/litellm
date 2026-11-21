"""Shared rig for the `test_dec1_config_*` contracts.

Every contract here boots its own proxy from a config file the test writes, so the file is the only
source of the behaviour under test. `booted` wraps `owned_proxy_process` (private scratch database and
Redis, `LITELLM_MASTER_KEY` set to the harness key, `STORE_MODEL_IN_DB=True`), `refused` wraps
`refused_boot_log` for configs that must stop the boot, and `scripted_upstream` is a local
OpenAI-shaped peer that records every request the proxy sends so a test can assert the outbound wire.

Assumption shared by all files: the harness rewrites a config whose `router_settings` has no
`num_retries` to `num_retries: 0` (see `private_configuration` in `_support/process.py`), so a failed
upstream call is attempted exactly once unless a test sets `num_retries` itself.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value, string_value
from tests.integration._support.process import OwnedProxy, owned_proxy_process, refused_boot_log
from tests.integration._support.wire import Reply, Request, Wire, wire_server

UPSTREAM_KEY: Final = "sk-dec1-config-upstream"
ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])
_NO_ENVIRONMENT: Final[Mapping[str, str]] = MappingProxyType({})
_MODEL_DISCOVERY: Final = ("GET", "/v1/models")


def chat_body(content: str = "dec1 config reply", *, model: str = "gpt-4o-mini") -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    ).encode()


def chat_reply(content: str = "dec1 config reply", *, status: int = 200) -> Reply:
    if status == 200:
        return Reply(body=chat_body(content))
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": content, "type": "upstream_error", "code": str(status)}}).encode(),
    )


def _always_ok(request: Request) -> Reply:
    if (request.method, request.target) == _MODEL_DISCOVERY:
        return Reply(body=b'{"object":"list","data":[]}')
    return chat_reply()


@contextmanager
def scripted_upstream(respond: Callable[[Request], Reply] = _always_ok) -> Iterator[Wire]:
    """An OpenAI-shaped peer on a free loopback port. It answers every chat call with a 200 unless
    `respond` says otherwise, and keeps each request for `sent`."""
    with wire_server(respond) as wire:
        yield wire


def sent(wire: Wire) -> tuple[Request, ...]:
    """What the proxy sent upstream since the last call, without model-discovery polls."""
    return tuple(request for request in wire.drain() if (request.method, request.target) != _MODEL_DISCOVERY)


def sent_json(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def deployment(
    name: str,
    api_base: str,
    *,
    model: str = "openai/gpt-4o-mini",
    api_key: str | None = UPSTREAM_KEY,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> dict[str, JsonValue]:
    """One `model_list` entry aimed at `api_base` (the upstream URL plus `/v1`)."""
    litellm_params: Final[dict[str, JsonValue]] = {
        "model": model,
        "api_base": api_base,
        **({} if api_key is None else {"api_key": api_key}),
        **params,
    }
    return {
        "model_name": name,
        "litellm_params": litellm_params,
        **({} if model_info is None else {"model_info": dict(model_info)}),
    }


def write_config(directory: Path, config: Mapping[str, object], name: str = "dec1-config.yaml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / name
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False))
    return path


@contextmanager
def booted(
    gateway: Gateway,
    directory: Path,
    config: Mapping[str, object] | Path,
    *,
    environment: Mapping[str, str] = _NO_ENVIRONMENT,
    remove_environment: tuple[str, ...] = (),
    extra_arguments: tuple[str, ...] = (),
) -> Iterator[OwnedProxy]:
    """Boot one proxy on `config` (a mapping is written to `directory` first) and yield it once ready."""
    path: Final = config if isinstance(config, Path) else write_config(directory, config)
    with owned_proxy_process(
        gateway,
        directory,
        environment,
        config=path,
        remove_environment=remove_environment,
        extra_arguments=extra_arguments,
    ) as owned:
        yield owned


def refused(
    gateway: Gateway,
    directory: Path,
    config: Mapping[str, object] | Path,
    *,
    environment: Mapping[str, str] = _NO_ENVIRONMENT,
) -> str:
    """Boot a proxy that must exit non-zero before becoming ready, and return everything it logged.

    Assumes `refused_boot_log` keeps the harness environment (`LITELLM_MASTER_KEY`, `DATABASE_URL`), so
    the only reason to refuse is the config under test. It has no `remove_environment`, so a test that
    needs a variable gone passes it as an empty override and says so.
    """
    path: Final = config if isinstance(config, Path) else write_config(directory, config)
    return refused_boot_log(gateway, directory, environment, config=path)


def with_key(gateway: Gateway, key: str) -> Gateway:
    """The same proxy, called with another bearer key (for a config that sets its own master key)."""
    return replace(gateway, key=key)


def chat(
    gateway: Gateway,
    model: str,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    **fields: JsonValue,
) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"dec1 config {uuid.uuid4().hex}"}], **fields},
        key=key,
        headers=headers,
    )


def error_message(response: httpx.Response) -> str:
    """The `error.message` string of an OpenAI-shaped proxy error body."""
    return string_value(object_value(JSON_OBJECT.validate_json(response.content)["error"])["message"])


def model_entries(gateway: Gateway, name: str) -> Sequence[dict[str, JsonValue]]:
    """The `/model/info` rows whose `model_name` is `name`."""
    return [item for item in ITEMS.validate_python(gateway.get("/model/info")["data"]) if item["model_name"] == name]


def readiness_details(gateway: Gateway) -> dict[str, JsonValue]:
    return gateway.get("/health/readiness/details")


def config_rows(gateway: Gateway, section: str) -> dict[str, JsonValue]:
    """The `LiteLLM_Config` row for one section, or an empty mapping when the row is absent."""
    rows: Final = gateway.rows('SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s', (section,))
    if not rows:
        return {}
    value: Final = rows[0]["param_value"]
    return JSON_OBJECT.validate_json(value) if isinstance(value, str) else object_value(value)


def served_model_names(gateway: Gateway) -> frozenset[str]:
    """The ids `/v1/models` lists for the master key."""
    return frozenset(string_value(item["id"]) for item in ITEMS.validate_python(gateway.get("/v1/models")["data"]))


def active_settings(gateway: Gateway) -> dict[str, JsonValue]:
    """`GET /active/callbacks`: the module-level litellm settings the worker is running with."""
    return gateway.get("/active/callbacks")

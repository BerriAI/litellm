"""Shared rig for the `test_dec1_healthchk_*` files: model health checks and what they do to routing.

One in-process scripted upstream stands in for every provider endpoint a health probe can reach
(`/chat/completions`, `/completions`, `/embeddings`, `/images/generations`, `/audio/speech`,
`/audio/transcriptions`, `/rerank`, `/batches`, `/responses`, `/v1/messages`). It records every request it
saw with its path, headers and JSON body (or the raw body for multipart), and picks its answer from the
upstream model name the proxy sent:

`ok-<label>` answers 200. `status-<code>-<label>` always answers that HTTP status. `probefail-<code>-<label>`
answers that status only to health probes and 200 to client calls, so a test can tell "the probe marked it
unhealthy" apart from "a client call failed". `slow-<label>` answers 200 after `SLOW_SECONDS`, and
`probeslow-<label>` does so only for health probes. `lag-<label>`
answers 200 after `LAG_SECONDS` and is used to observe probe concurrency.

A request counts as a health probe when its body carries one of the fixed probe inputs the proxy uses:
the chat messages `Hey how's it going?` / `What's 1 + 1?` (`_get_random_llm_message`,
litellm/proxy/health_check.py:208) or the `DEFAULT_HEALTH_CHECK_PROMPT` text `test from litellm`
(litellm/constants.py:45) as `prompt`, `input` or `query`. Client calls made through `chat` here always
send a unique marker instead, so the two never collide.

Successful answers carry `x-ratelimit-remaining-requests: 4999` and `x-ratelimit-remaining-tokens: 99999`
so the `/health` report can be checked for the provider headers it surfaces
(`create_health_check_response`, litellm/litellm_core_utils/health_check_utils.py:23).
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx
import yaml
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

PROVIDER_KEY: Final = "dec1-healthchk-provider-key"
UNREACHABLE_API_BASE: Final = "http://127.0.0.1:9/v1"
PROBE_CHAT_MESSAGES: Final = frozenset({"Hey how's it going?", "What's 1 + 1?"})
PROBE_PROMPT: Final = "test from litellm"
HEALTH_SERVICE_ACCOUNT: Final = "litellm-internal-health-check"
SLOW_SECONDS: Final = 8
LAG_SECONDS: Final = 0.6
RATE_LIMIT_HEADERS: Final[Mapping[str, str]] = {
    "x-ratelimit-remaining-requests": "4999",
    "x-ratelimit-remaining-tokens": "99999",
}
HEALTH_TABLE: Final = '"LiteLLM_HealthCheckTable"'
_MULTIPART_MODEL: Final = re.compile(rb'name="model"\r\n\r\n([^\r\n]+)')


@dataclass(frozen=True, slots=True)
class Seen:
    model: str
    path: str
    method: str
    body: Mapping[str, JsonValue]
    raw: bytes
    headers: Mapping[str, str]
    at: float
    probe: bool


@dataclass(slots=True)
class ProbeUpstream:
    wire: Wire | None = None
    log: list[Seen] = field(default_factory=list)  # mutable-ok: appended by the upstream's handler threads
    in_flight: int = 0
    peak_in_flight: int = 0
    overrides: dict[str, int] = field(default_factory=dict)  # mutable-ok: set by tests, read by the handler
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        assert self.wire is not None
        return self.wire.url

    @property
    def api_base(self) -> str:
        return f"{self.url}/v1"

    def calls(self, model: str) -> tuple[Seen, ...]:
        return tuple(seen for seen in tuple(self.log) if seen.model == model)

    def probes(self, model: str) -> tuple[Seen, ...]:
        return tuple(seen for seen in self.calls(model) if seen.probe)

    def client_calls(self, model: str) -> tuple[Seen, ...]:
        return tuple(seen for seen in self.calls(model) if not seen.probe)

    def count(self, model: str) -> int:
        return len(self.calls(model))

    def force(self, model: str, status: int | None) -> None:
        """Make every request for `model` answer `status` from now on; `None` restores its scripted behaviour."""
        if status is None:
            self.overrides.pop(model, None)
        else:
            self.overrides[model] = status

    def on_path(self, suffix: str) -> tuple[Seen, ...]:
        return tuple(seen for seen in tuple(self.log) if seen.path.endswith(suffix))


def _body_of(request: Request) -> Mapping[str, JsonValue]:
    if not request.body:
        return {}
    try:
        return JSON_OBJECT.validate_json(request.body)
    except ValueError:
        return {}


def _model_of(request: Request, body: Mapping[str, JsonValue]) -> str:
    if isinstance(body.get("model"), str):
        return string_value(body["model"])
    found: Final = _MULTIPART_MODEL.search(request.body)
    if found is not None:
        return found.group(1).decode()
    return request.target.split("?")[0].rsplit("/", 1)[-1]


def is_probe(body: Mapping[str, JsonValue], raw: bytes) -> bool:
    messages: Final = body.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict) and first.get("content") in PROBE_CHAT_MESSAGES | {PROBE_PROMPT, "test"}:
            return True
    for name in ("prompt", "input", "query"):
        value = body.get(name)
        if value == PROBE_PROMPT or value == [PROBE_PROMPT]:
            return True
    return b"health_check" in raw or (not body and b"filename=" in raw)


def _error(status: int, model: str) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": f"scripted {status} from {model}", "type": "scripted_error"}}).encode(),
    )


def _success(request: Request, model: str) -> Reply:
    path: Final = request.target.split("?")[0]
    text: Final = f"served {model}"
    if path.endswith("/slack"):
        return Reply(body=b"ok", content_type="text/plain")
    if path.endswith("/embeddings"):
        payload: dict[str, JsonValue] = {
            "object": "list",
            "model": model,
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5, 0.75]}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    elif path.endswith("/images/generations"):
        payload = {"created": int(time.time()), "data": [{"url": "https://example.invalid/dec1-healthchk.png"}]}
    elif path.endswith("/audio/speech"):
        return Reply(body=b"ID3dec1-healthchk-audio", content_type="audio/mpeg", headers=RATE_LIMIT_HEADERS)
    elif path.endswith("/audio/transcriptions"):
        payload = {"text": text}
    elif path.endswith("/rerank"):
        payload = {
            "id": f"rerank-{uuid.uuid4().hex}",
            "results": [{"index": 0, "relevance_score": 0.5}],
            "meta": {"billed_units": {"search_units": 1}},
        }
    elif path.endswith("/batches"):
        payload = {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}
    elif path.endswith("/completions") and not path.endswith("/chat/completions"):
        payload = {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
    elif path.endswith("/responses"):
        payload = {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        }
    elif path.endswith("/messages"):
        payload = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }
    else:
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
    return Reply(body=json.dumps(payload).encode(), headers=RATE_LIMIT_HEADERS)


def _answer(upstream: ProbeUpstream, request: Request) -> Reply:
    body: Final = _body_of(request)
    model: Final = _model_of(request, body)
    probe: Final = is_probe(body, request.body)
    upstream.log.append(
        Seen(
            model,
            request.target.split("?")[0],
            request.method,
            body,
            request.body,
            request.headers,
            time.monotonic(),
            probe,
        )
    )
    kind: Final = model.split("-", 1)[0]
    with upstream.lock:
        upstream.in_flight += 1
        upstream.peak_in_flight = max(upstream.peak_in_flight, upstream.in_flight)
    try:
        forced = upstream.overrides.get(model)
        if forced is not None:
            return _error(forced, model)
        if kind == "status":
            return _error(int(model.split("-")[1]), model)
        if kind == "probefail" and probe:
            return _error(int(model.split("-")[1]), model)
        if kind == "slow" or (kind == "probeslow" and probe):
            time.sleep(SLOW_SECONDS)
        if kind == "lag":
            time.sleep(LAG_SECONDS)
        return _success(request, model)
    finally:
        with upstream.lock:
            upstream.in_flight -= 1


@contextmanager
def probe_upstream() -> Iterator[ProbeUpstream]:
    upstream: Final = ProbeUpstream()

    def respond(request: Request) -> Reply:
        return _answer(upstream, request)

    with wire_server(respond) as wire:
        upstream.wire = wire
        yield upstream


def unique(label: str) -> str:
    """A model or group name no other test or earlier run shares, so rows in the shared database stay apart."""
    return f"{label}-{uuid.uuid4().hex[:8]}"


def deployment(
    group: str,
    upstream_model: str,
    api_base: str,
    *,
    provider: str = "openai",
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> dict[str, JsonValue]:
    """One `model_list` entry whose deployment id is `hc-<upstream_model>` unless `model_info` names one."""
    return {
        "model_name": group,
        "litellm_params": {
            "model": f"{provider}/{upstream_model}",
            "api_key": PROVIDER_KEY,
            "api_base": api_base,
            **params,
        },
        "model_info": {"id": f"hc-{upstream_model}", **(model_info or {})},
    }


def health_config(
    directory: Path,
    *,
    model_list: Sequence[Mapping[str, JsonValue]],
    router_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    assert isinstance(base, dict)
    config: Final = {
        **base,
        "model_list": [dict(entry) for entry in model_list],
        "router_settings": {"num_retries": 0, "disable_cooldowns": True, **(router_settings or {})},
        "general_settings": {**base.get("general_settings", {}), **(general_settings or {})},
        "litellm_settings": {**base.get("litellm_settings", {}), **(litellm_settings or {})},
    }
    path: Final = directory / f"dec1-healthchk-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    upstream: ProbeUpstream


@contextmanager
def health_rig(
    directory: Path,
    models: Callable[[str], Sequence[Mapping[str, JsonValue]]],
    *,
    router_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | Callable[[str], Mapping[str, str]] | None = None,
) -> Iterator[Rig]:
    """The probe upstream plus one owned proxy whose `model_list` is `models(upstream.api_base)`.

    `environment`, when callable, is called with the upstream's root URL (for a webhook pointed at it)."""
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        upstream: Final = stack.enter_context(probe_upstream())
        config: Final = health_config(
            directory,
            model_list=models(upstream.api_base),
            router_settings=router_settings,
            general_settings=general_settings,
            litellm_settings=litellm_settings,
        )
        overrides: Final = environment(upstream.url) if callable(environment) else (environment or {})
        gateway: Final = stack.enter_context(owned_proxy(shared, directory, dict(overrides), config=config))
        yield Rig(gateway, upstream)


def health(gateway: Gateway, *, key: str | None = None, **params: str) -> httpx.Response:
    """`GET /health` with the given query parameters, as the master key unless `key` is given."""
    return gateway.request("GET", "/health", params=params or None, key=key)


def report(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def endpoints(response: httpx.Response, kind: str) -> tuple[dict[str, JsonValue], ...]:
    """The `healthy_endpoints` or `unhealthy_endpoints` entries of a `/health` answer."""
    return tuple(object_value(entry) for entry in list_value(report(response)[f"{kind}_endpoints"]))


def endpoint_for(response: httpx.Response, deployment_id: str) -> tuple[str, dict[str, JsonValue]]:
    """Which list (`healthy` or `unhealthy`) holds the entry whose `model_id` is `deployment_id`, and the entry."""
    for kind in ("healthy", "unhealthy"):
        for entry in endpoints(response, kind):
            if entry.get("model_id") == deployment_id:
                return kind, entry
    raise AssertionError(f"{deployment_id} is in no endpoint list: {response.text}")


def marker() -> str:
    return f"dec1-healthchk-client-{uuid.uuid4().hex}"


def chat(gateway: Gateway, model: str, *, key: str | None = None) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker()}]},
        key=key,
    )


def served_text(response: httpx.Response) -> str:
    choices: Final = list_value(report(response)["choices"])
    return string_value(object_value(object_value(choices[0])["message"])["content"])


def health_rows(gateway: Gateway, model_name: str) -> list[dict[str, JsonValue]]:
    """Every `LiteLLM_HealthCheckTable` row for `model_name`, newest first."""
    return gateway.rows(
        f"SELECT model_name, model_id, status, healthy_count, unhealthy_count, error_message, response_time_ms, "
        f"checked_by, details FROM {HEALTH_TABLE} WHERE model_name = %s ORDER BY checked_at DESC",
        (model_name,),
    )


def listed_ids(gateway: Gateway, *, params: Mapping[str, str] | None = None) -> frozenset[str]:
    """The `id` of every entry `GET /v1/models` lists for the master key."""
    response: Final = gateway.request("GET", "/v1/models", params=params)
    assert response.status_code == 200, response.text
    return frozenset(string_value(object_value(entry)["id"]) for entry in list_value(report(response)["data"]))

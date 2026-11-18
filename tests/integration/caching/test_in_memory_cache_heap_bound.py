import itertools
import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.openai_wire import answering_model_discovery, chat_reply, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import anthropic_peer, marker_of
from litellm.caching.in_memory_cache import DEFAULT_MAX_SIZE_IN_MEMORY

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 300)

_FIRST: Final = "heap-bound-first"
_SECOND: Final = "heap-bound-second"
_SLOTS: Final = 8
_CYCLES: Final = 6 * _SLOTS
_FILLERS: Final = 2 * _SLOTS
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _rig_config() -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))


def _config(directory: Path, api_base: str) -> Path:
    config: Final = _rig_config()
    deployments: Final = [
        {
            "model_name": name,
            "litellm_params": {"model": "anthropic/claude-opus-5-5", "api_base": api_base, "api_key": "synthetic"},
        }
        for name in (_FIRST, _SECOND)
    ]
    merged: Final = {
        **config,
        "model_list": deployments,
        "general_settings": {
            **object_value(config["general_settings"]),
            "user_api_key_cache_ttl": 3600,
            "user_api_key_cache_max_size": _SLOTS,
        },
        "litellm_settings": {
            **object_value(config["litellm_settings"]),
            "cache": False,
            "enable_redis_auth_cache": False,
        },
    }
    path: Final = directory / f"heap-bound-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(merged))
    return path


def _chat(gateway: Gateway, key: str, model: str) -> tuple[int, str]:
    marker: Final = uuid.uuid4().hex
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": f"Reply. marker-{marker}"}]},
        key=key,
    )
    return response.status_code, marker


def _cycle(gateway: Gateway, key: str, turn: int) -> tuple[tuple[int, int], str]:
    allowed, denied = (_SECOND, _FIRST) if turn % 2 == 0 else (_FIRST, _SECOND)
    gateway.post("/key/update", {"key": key, "models": [allowed]})
    allowed_status, marker = _chat(gateway, key, allowed)
    denied_status, _ = _chat(gateway, key, denied)
    return (allowed_status, denied_status), marker


def _fillers(scenario: Scenario, count: int) -> Sequence[str]:
    keys: Final = tuple(string_value(scenario.gateway.post("/key/generate", {})["key"]) for _ in range(count))
    scenario.cleanups.callback(scenario.gateway.post, "/key/delete", {"keys": list(keys)})
    return keys


def _reached(peer: Wire) -> list[str]:
    return sorted(marker_of(request) for request in peer.drain())


def test_key_updates_past_twice_the_key_cache_slots_take_effect_on_the_next_request(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(anthropic_peer) as peer:
        config: Final = _config(tmp_path, peer.url)
        with (
            owned_proxy_process(gateway, tmp_path, {}, config=config, workers=1) as owned,
            owned.gateway.scenario() as scenario,
        ):
            anchor: Final = scenario.key(models=[_FIRST])
            anchor_status, anchor_marker = _chat(owned.gateway, anchor, _FIRST)
            churned: Final = scenario.key(models=[_FIRST])
            cycles: Final = tuple(_cycle(owned.gateway, churned, turn) for turn in range(_CYCLES))
            assert [statuses for statuses, _ in cycles] == [(200, 403)] * _CYCLES
            filled: Final = tuple(
                owned.gateway.request("GET", "/v1/models", key=key).status_code for key in _fillers(scenario, _FILLERS)
            )
            assert filled == (200,) * _FILLERS
            after_anchor, after_anchor_marker = _chat(owned.gateway, anchor, _FIRST)
            last_allowed: Final = _SECOND if (_CYCLES - 1) % 2 == 0 else _FIRST
            after_churned, after_churned_marker = _chat(owned.gateway, churned, last_allowed)
            assert (anchor_status, after_anchor, after_churned) == (200, 200, 200)
            assert _reached(peer) == sorted(
                (anchor_marker, *(marker for _, marker in cycles), after_anchor_marker, after_churned_marker)
            )


_RESPONSE_SLOTS: Final = DEFAULT_MAX_SIZE_IN_MEMORY
_OVERFLOW: Final = 20
_BURST_PER_ENDPOINT: Final = 6
_NO_STORE: Final[Mapping[str, JsonValue]] = MappingProxyType({"no-store": True})
_FRESH_TTL: Final[Mapping[str, JsonValue]] = MappingProxyType({"ttl": 600})
_EMBEDDING: Final = json.dumps(
    {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "text-embedding-3-large",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
).encode()


class _PeerBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    model: str = ""
    stream: bool = False


def _response_cache_peer(request: Request) -> Reply:
    body: Final = _PeerBody.model_validate_json(request.body or b"{}")
    identity: Final = f"{marker_of(request)}-{uuid.uuid4().hex[:8]}"
    match request.target.split("?", 1)[0]:
        case "/v1/messages":
            return anthropic_peer(request)
        case "/v1/chat/completions":
            return chat_reply(f"chatcmpl-{identity}", body.model, "scripted", stream=body.stream)
        case "/v1/responses":
            return responses_reply(f"resp_{identity}", body.model, "scripted", stream=body.stream)
        case "/v1/embeddings":
            return Reply(body=_EMBEDDING)
        case _:
            return Reply(status=404, body=b'{"error": {"message": "unscripted route"}}')


_NAN_TTL: Final[Mapping[str, JsonValue]] = MappingProxyType({"ttl": "NaN"})
_UNUSABLE_TTLS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"lowercase-nan": "nan", "word": "abc", "empty-string": "", "list": [1], "object": {"x": 1}}
)


def _prompt_fields(path: str, text: str) -> dict[str, JsonValue]:
    match path:
        case "/v1/chat/completions":
            return {"messages": [{"role": "user", "content": text}]}
        case "/v1/messages":
            return {"max_tokens": 16, "messages": [{"role": "user", "content": text}]}
        case _:
            return {"input": text}


@dataclass(frozen=True, slots=True)
class _Endpoint:
    path: str
    model: str
    streamed: bool

    def body(self, marker: str, cache: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return {
            "model": self.model,
            **_prompt_fields(self.path, f"Reply. marker-{marker}"),
            **({"stream": True} if self.streamed else {}),
            "cache": dict(cache),
        }


_CHAT: Final = _Endpoint("/v1/chat/completions", "rc-openai", streamed=False)
_ENDPOINTS: Final = MappingProxyType(
    {
        "chat": _CHAT,
        "chat-stream": _Endpoint("/v1/chat/completions", "rc-openai", streamed=True),
        "messages": _Endpoint("/v1/messages", "rc-anthropic", streamed=False),
        "messages-stream": _Endpoint("/v1/messages", "rc-anthropic", streamed=True),
        "responses": _Endpoint("/v1/responses", "rc-openai", streamed=False),
        "responses-stream": _Endpoint("/v1/responses", "rc-openai", streamed=True),
        "embeddings": _Endpoint("/v1/embeddings", "rc-embedding", streamed=False),
    }
)


def _posted_markers(peer: Wire) -> frozenset[str]:
    return frozenset(marker_of(request) for request in peer.drain() if request.method == "POST")


@dataclass(frozen=True, slots=True)
class _CacheRig:
    gateway: Gateway
    peer: Wire

    def send(self, endpoint: _Endpoint, marker: str, cache: Mapping[str, JsonValue]) -> int:
        return self.gateway.request("POST", endpoint.path, endpoint.body(marker, cache)).status_code

    def served(self, endpoint: _Endpoint, marker: str) -> bool:
        self.peer.drain()
        response: Final = self.gateway.request("POST", endpoint.path, endpoint.body(marker, _NO_STORE))
        assert response.status_code == 200, response.text
        return marker not in _posted_markers(self.peer)


def _response_cache_config(directory: Path, peer_url: str) -> Path:
    config: Final = _rig_config()
    deployments: Final = [
        {
            "model_name": "rc-openai",
            "litellm_params": {"model": "openai/gpt-5.6", "api_base": f"{peer_url}/v1", "api_key": "synthetic"},
        },
        {
            "model_name": "rc-anthropic",
            "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_base": peer_url, "api_key": "synthetic"},
        },
        {
            "model_name": "rc-embedding",
            "litellm_params": {
                "model": "openai/text-embedding-3-large",
                "api_base": f"{peer_url}/v1",
                "api_key": "synthetic",
            },
        },
    ]
    merged: Final = {
        **config,
        "model_list": deployments,
        "litellm_settings": {
            **object_value(config["litellm_settings"]),
            "cache": True,
            "cache_params": {"type": "local"},
            "enable_redis_auth_cache": False,
        },
    }
    path: Final = directory / f"response-cache-bound-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(merged))
    return path


@pytest.fixture(scope="module")
def response_cache_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_CacheRig]:
    directory: Final = tmp_path_factory.mktemp("response-cache-bound")
    with (
        wire_server(answering_model_discovery(_response_cache_peer)) as peer,
        owned_proxy_process(
            session_gateway, directory, {}, config=_response_cache_config(directory, peer.url), workers=1
        ) as owned,
    ):
        yield _CacheRig(owned.gateway, peer)


def _flood(rig: _CacheRig, endpoint: _Endpoint, cache: Mapping[str, JsonValue], count: int) -> tuple[str, ...]:
    markers: Final = tuple(uuid.uuid4().hex for _ in range(count))
    statuses: Final = tuple(rig.send(endpoint, marker, cache) for marker in markers)
    assert statuses == (200,) * count
    return markers


def _served_indexes(rig: _CacheRig, endpoint: _Endpoint, markers: Sequence[str]) -> tuple[int, ...]:
    return tuple(index for index, marker in enumerate(markers) if rig.served(endpoint, marker))


def _assert_newest_suffix(served: Sequence[int], flooded: int) -> None:
    summary: Final = f"{len(served)} of {flooded} served, first {served[:3]}, last {served[-3:]}"
    assert served, summary
    assert tuple(served) == tuple(range(served[0], flooded)), summary


def _assert_next_response_cached(rig: _CacheRig, endpoint: _Endpoint) -> None:
    fresh: Final = uuid.uuid4().hex
    assert rig.send(endpoint, fresh, _FRESH_TTL) == 200
    assert eventually(lambda: rig.served(endpoint, fresh), lambda hit: hit, return_last_on_timeout=True), (
        "a response with a usable ttl was not cached after the flood"
    )


def _assert_flood_keeps_the_newest_responses(
    rig: _CacheRig, endpoint: _Endpoint, cache: Mapping[str, JsonValue]
) -> None:
    markers: Final = _flood(rig, endpoint, cache, _RESPONSE_SLOTS + _OVERFLOW)
    eventually(lambda: rig.served(endpoint, markers[-1]), lambda hit: hit, return_last_on_timeout=True)
    served: Final = _served_indexes(rig, endpoint, markers)
    _assert_newest_suffix(served, len(markers))
    assert _RESPONSE_SLOTS - 1 <= len(served) <= _RESPONSE_SLOTS, f"{len(served)} served"
    _assert_next_response_cached(rig, endpoint)


@pytest.mark.parametrize("endpoint", tuple(_ENDPOINTS.values()), ids=tuple(_ENDPOINTS))
def test_nan_ttl_flood_past_the_response_cache_size_keeps_the_newest_responses_and_caches_the_next_one(
    response_cache_rig: _CacheRig, endpoint: _Endpoint
) -> None:
    _assert_flood_keeps_the_newest_responses(response_cache_rig, endpoint, _NAN_TTL)


@pytest.mark.parametrize("ttl", tuple(_UNUSABLE_TTLS.values()), ids=tuple(_UNUSABLE_TTLS))
def test_unusable_ttl_shapes_flooded_past_the_response_cache_size_keep_the_newest_responses(
    response_cache_rig: _CacheRig, ttl: JsonValue
) -> None:
    _assert_flood_keeps_the_newest_responses(response_cache_rig, _CHAT, {"ttl": ttl})


def _post(gateway: Gateway, endpoint: _Endpoint, marker: str) -> int:
    with httpx.Client(base_url=gateway.client.base_url, timeout=60, trust_env=False) as client:
        return client.post(
            endpoint.path,
            json=endpoint.body(marker, _NAN_TTL),
            headers={"Authorization": f"Bearer {gateway.key}"},
        ).status_code


def _liveliness(gateway: Gateway) -> int:
    with httpx.Client(base_url=gateway.client.base_url, timeout=60, trust_env=False) as client:
        return client.get("/health/liveliness").status_code


def test_concurrent_burst_into_a_response_cache_full_of_nan_ttls_caches_every_burst_response(
    response_cache_rig: _CacheRig,
) -> None:
    rig: Final = response_cache_rig
    fill: Final = _flood(rig, _CHAT, _NAN_TTL, _RESPONSE_SLOTS)
    eventually(lambda: rig.served(_CHAT, fill[-1]), lambda hit: hit)
    burst: Final = tuple(
        (endpoint, uuid.uuid4().hex)
        for endpoint, _ in itertools.product(_ENDPOINTS.values(), range(_BURST_PER_ENDPOINT))
    )
    with ThreadPoolExecutor(2 * len(burst)) as pool:
        answers: Final = tuple(pool.submit(_post, rig.gateway, endpoint, marker) for endpoint, marker in burst)
        probes: Final = tuple(pool.submit(_liveliness, rig.gateway) for _ in burst)
        statuses: Final = tuple(answer.result() for answer in answers)
        liveliness: Final = tuple(probe.result() for probe in probes)
    assert statuses == (200,) * len(burst)
    assert liveliness == (200,) * len(burst)
    missing: Final = eventually(
        lambda: tuple(marker for endpoint, marker in burst if not rig.served(endpoint, marker)),
        lambda left: not left,
        return_last_on_timeout=True,
    )
    assert not missing, f"{len(missing)} of {len(burst)} burst responses were never served from the cache"
    served_fill: Final = _served_indexes(rig, _CHAT, fill)
    _assert_newest_suffix(served_fill, len(fill))
    assert _RESPONSE_SLOTS - len(burst) <= len(served_fill) + len(burst) <= _RESPONSE_SLOTS, (
        f"{len(served_fill)} fill and {len(burst)} burst responses served"
    )
    _assert_next_response_cached(rig, _CHAT)

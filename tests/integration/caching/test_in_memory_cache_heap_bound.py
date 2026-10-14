import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Wire, wire_server
from integration.providers._cache_control_marks_support import anthropic_peer, marker_of

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 300)

_FIRST: Final = "heap-bound-first"
_SECOND: Final = "heap-bound-second"
_SLOTS: Final = 8
_CYCLES: Final = 6 * _SLOTS
_FILLERS: Final = 2 * _SLOTS
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _config(directory: Path, api_base: str) -> Path:
    config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
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

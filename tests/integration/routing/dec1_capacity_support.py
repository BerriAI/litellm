"""Shared pieces for the ``test_dec1_capacity_*`` contracts (model capacity, the dynamic rate limiter v3).

Model capacity is the ``dynamic_rate_limiter_v3`` callback: a model group's ``rpm`` and ``tpm`` (summed over its
deployments) become one window shared by every caller, ``priority_reservation`` splits that window into priority
pools, and ``priority_reservation_settings.saturation_threshold`` (default 0.5, ``litellm/types/utils.py``
``PriorityReservationSettings``) decides when the pools stop lending to each other. The Python handler is
``litellm/proxy/hooks/dynamic_rate_limiter_v3.py`` and draft PR #51540 replaces it with a Rust callback, so every
literal here is read from that file on main.

The provider, request helpers and Redis helpers come from the limits family
(``tests/integration/authorization/dec1_limits_support.py``): every provider answer reports 11 prompt and 7
completion tokens (18 total) unless a test passes its own ``Usage``. This module adds the capacity proxy, its
config with an explicit ``model_list``, the counter keys the limiter writes and the exact refusal texts.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import yaml
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value
from tests.integration._support.process import OwnedState, owned_proxy
from tests.integration.authorization.dec1_limits_support import (
    ANTHROPIC_MODEL,
    DEFAULT_USAGE,
    PROVIDER_MODEL,
    REDIS_ENVIRONMENT,
    WINDOW_SECONDS,
    Usage,
    anthropic_messages,
    chat,
    counter_key,
    embed,
    error_message,
    error_of,
    hashed,
    posts,
    probe,
    provider,
    redis_integer,
    respond,
    scope_headers,
    shared_redis,
    statuses,
)

__all__ = (
    "ANTHROPIC_MODEL",
    "DEFAULT_USAGE",
    "MODEL_SCOPE",
    "PRIORITY_SCOPE",
    "PROVIDER_KEY",
    "WINDOW_SECONDS",
    "Usage",
    "anthropic_entry",
    "anthropic_messages",
    "capacity_message",
    "capacity_proxy",
    "chat",
    "counter_key",
    "embed",
    "entry",
    "error_message",
    "error_of",
    "group_name",
    "hashed",
    "model_counter",
    "posts",
    "priority_counter",
    "priority_message",
    "probe",
    "provider",
    "redis_integer",
    "respond",
    "scope_headers",
    "settled",
    "shared_redis",
    "statuses",
    "wait_for_model",
)

MODEL_SCOPE: Final = "model_saturation_check"
PRIORITY_SCOPE: Final = "priority_model"
PROVIDER_KEY: Final = "dec1-capacity-provider-key"


def group_name(label: str) -> str:
    """A model group name no other test uses, so its Redis counters start empty."""
    return f"dec1-capacity-{label}-{uuid.uuid4().hex[:10]}"


def entry(
    group: str,
    api_base: str,
    *,
    model_id: str | None = None,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> dict[str, JsonValue]:
    """One OpenAI-shaped ``model_list`` deployment of ``group`` that calls ``api_base`` (a wire URL)."""
    return {
        "model_name": group,
        "litellm_params": {
            "model": f"openai/{PROVIDER_MODEL}",
            "api_key": PROVIDER_KEY,
            "api_base": f"{api_base}/v1",
            **params,
        },
        "model_info": {"id": model_id or f"{group}-{uuid.uuid4().hex[:6]}", **(model_info or {})},
    }


def anthropic_entry(group: str, api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    """One Anthropic deployment of ``group``, so ``/v1/messages`` reaches the wire as ``POST /v1/messages``."""
    return {
        "model_name": group,
        "litellm_params": {
            "model": f"anthropic/{ANTHROPIC_MODEL}",
            "api_key": PROVIDER_KEY,
            "api_base": api_base,
            **params,
        },
        "model_info": {"id": f"{group}-{uuid.uuid4().hex[:6]}"},
    }


def capacity_config(
    directory: Path,
    *,
    model_list: Sequence[Mapping[str, JsonValue]] = (),
    priority_reservation: Mapping[str, JsonValue] | None = None,
    priority_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    callbacks: Sequence[str] = ("dynamic_rate_limiter_v3",),
    without_redis: bool = False,
) -> Path:
    """The suite's proxy config with the dynamic limiter on, ``model_list`` appended and ``num_retries: 0``."""
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    inherited: Final = {
        name: value
        for name, value in object_value(base["litellm_settings"]).items()
        if not (without_redis and name in ("cache", "cache_params", "enable_redis_auth_cache"))
    }
    capacity_settings: Final[dict[str, JsonValue]] = {
        **({"callbacks": list(callbacks)} if callbacks else {}),
        **({"priority_reservation": dict(priority_reservation)} if priority_reservation is not None else {}),
        **({"priority_reservation_settings": dict(priority_settings)} if priority_settings is not None else {}),
    }
    base_models: Final = base.get("model_list")
    config: Final = {
        **base,
        "model_list": [*(base_models if isinstance(base_models, list) else []), *(dict(item) for item in model_list)],
        "general_settings": {**object_value(base["general_settings"]), **(general_settings or {})},
        "litellm_settings": {**inherited, **capacity_settings, **(litellm_settings or {})},
        "router_settings": {**object_value(base["router_settings"]), "num_retries": 0, **(router_settings or {})},
    }
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / f"dec1-capacity-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def capacity_proxy(
    gateway: Gateway,
    directory: Path,
    *,
    model_list: Sequence[Mapping[str, JsonValue]] = (),
    priority_reservation: Mapping[str, JsonValue] | None = None,
    priority_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    callbacks: Sequence[str] = ("dynamic_rate_limiter_v3",),
    environment: Mapping[str, str] | None = None,
    without_redis: bool = False,
    shared: bool = False,
) -> Iterator[Gateway]:
    """An owned proxy with the dynamic limiter on.

    ``shared=True`` boots it on the shared gateway's database and Redis (``OwnedState.SHARED_WITH_WORKER``), so its
    counters can be read through ``shared_redis()`` and a second ``shared=True`` proxy with the same config is a
    second instance on the same Redis. ``without_redis`` removes Redis entirely.
    """
    config: Final = capacity_config(
        directory,
        model_list=model_list,
        priority_reservation=priority_reservation,
        priority_settings=priority_settings,
        litellm_settings=litellm_settings,
        general_settings=general_settings,
        router_settings=router_settings,
        callbacks=callbacks,
        without_redis=without_redis,
    )
    with owned_proxy(
        gateway,
        directory,
        dict(environment or {}),
        config=config,
        remove_environment=REDIS_ENVIRONMENT if without_redis else (),
        state=OwnedState.SHARED_WITH_WORKER if shared else None,
    ) as candidate:
        yield candidate


def model_counter(model: str, kind: str) -> str:
    """The Redis key of the model-wide window: ``{model_saturation_check:<group>}:requests|tokens``."""
    return counter_key(MODEL_SCOPE, model, kind)


def priority_counter(model: str, pool: str, kind: str) -> str:
    """The Redis key of a priority pool: ``{priority_model:<group>:<priority or default_pool>}:requests|tokens``."""
    return counter_key(PRIORITY_SCOPE, f"{model}:{pool}", kind)


def settled(redis: Redis, key: str, expected: int, seconds: float = 30) -> int | None:
    """Poll a counter until settlement wrote ``expected`` into it, and return what it holds."""
    return eventually(lambda: redis_integer(redis, key), lambda value: value == expected, seconds=seconds)


def capacity_message(
    model: str,
    priority: str | None,
    kind: str,
    *,
    tpm: int | None = None,
    rpm: int | None = None,
    remaining: int = 0,
) -> str:
    """``_check_rate_limits``' text when the model-wide window refuses (dynamic_rate_limiter_v3.py:1309)."""
    return (
        f"Model capacity reached for {model}. Priority: {priority}, Rate limit type: {kind}, "
        f"Model TPM: {tpm if tpm is not None else 'not configured'}, "
        f"Model RPM: {rpm if rpm is not None else 'not configured'}, Remaining: {remaining}"
    )


def priority_message(
    model: str,
    priority: str | None,
    kind: str,
    saturation: str,
    *,
    tpm: int | None = None,
    rpm: int | None = None,
    remaining: int = 0,
) -> str:
    """``_check_rate_limits``' text when a saturated model refuses a priority pool (dynamic_rate_limiter_v3.py:1331).
    ``saturation`` is the ``{:.1%}`` rendering, for example ``50.0%``."""
    return (
        f"Priority-based rate limit exceeded. Model: {model}, Priority: {priority}, Rate limit type: {kind}, "
        f"Model TPM: {tpm if tpm is not None else 'not configured'}, "
        f"Model RPM: {rpm if rpm is not None else 'not configured'}, Remaining: {remaining}, "
        f"Model saturation: {saturation}"
    )


def wait_for_model(instance: Gateway, model: str, seconds: float = 90) -> None:
    """Wait until ``instance`` lists ``model`` in ``/model/info`` without sending it traffic, so no capacity is used."""

    def names() -> frozenset[str]:
        response: Final = instance.request("GET", "/model/info")
        if response.status_code != 200:
            return frozenset()
        data: Final = JSON_OBJECT.validate_json(response.content).get("data")
        return frozenset(str(object_value(item).get("model_name")) for item in (data if isinstance(data, list) else []))

    eventually(names, lambda listed: model in listed, seconds=seconds)

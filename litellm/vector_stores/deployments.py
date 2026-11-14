"""Fallback routing for a vector store served by several equivalent search deployments.

A ``vector_store_deployments`` group lists deployments that hold the same data, such as one
Azure AI Search service per region. A read tries them in ``order``. A deployment that fails
with a retryable error is cooled down and the same read moves to the next one.
"""

import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Generic, TypeAlias, TypeVar
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import DEFAULT_COOLDOWN_TIME_SECONDS
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.utils import should_retry

VECTOR_STORE_DEPLOYMENT_PARAM: Final = "vector_store_deployment"
SERVED_BY_HEADER: Final = "x-litellm-vector-store-deployment"
_ROUTING_PARAMS: Final = frozenset({"order", "cooldown_time", "allowed_fails"})
_FAILURE_WINDOW_SECONDS: Final = 60
_CONNECTION_PARAMS: Final = frozenset(
    {"api_base", "api_key", "api_version", "litellm_credential_name", "azure_search_service_name"}
)
_Value = TypeVar("_Value")  # rebind-ok: a TypeVar cannot be Final


class _DeploymentParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    custom_llm_provider: str
    order: int = 1
    cooldown_time: float = Field(default=DEFAULT_COOLDOWN_TIME_SECONDS, ge=0)
    allowed_fails: int = Field(default=1, ge=1)


class _DeploymentEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    deployment_name: str
    deployment_id: str | None = None
    litellm_params: _DeploymentParams


_ENTRIES: Final = TypeAdapter(tuple[_DeploymentEntry, ...])
_PARAMS: Final = TypeAdapter(dict[str, object])
_CONFIG: Final = TypeAdapter(dict[str, object])
_CACHED: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class VectorStoreDeployment:
    group: str
    deployment_id: str
    order: int
    cooldown_time: float
    allowed_fails: int
    connection: Mapping[str, object]

    @property
    def custom_llm_provider(self) -> str:
        return str(self.connection["custom_llm_provider"])


def parse_vector_store_deployments(config: object) -> tuple[VectorStoreDeployment, ...]:
    entries: Final = _ENTRIES.validate_python(_CONFIG.validate_python(config).get("vector_store_deployments") or ())
    return tuple(
        VectorStoreDeployment(
            group=entry.deployment_name,
            deployment_id=entry.deployment_id or f"{entry.deployment_name}-{position}",
            order=entry.litellm_params.order,
            cooldown_time=entry.litellm_params.cooldown_time,
            allowed_fails=entry.litellm_params.allowed_fails,
            connection=MappingProxyType(
                {
                    key: value
                    for key, value in _PARAMS.validate_python(entry.litellm_params.model_dump()).items()
                    if key not in _ROUTING_PARAMS
                }
            ),
        )
        for position, entry in enumerate(entries, start=1)
    )


class VectorStoreDeploymentRegistry:
    """The config's deployment groups, read by every path that serves a grouped vector store."""

    def __init__(self) -> None:
        self.deployments: tuple[VectorStoreDeployment, ...] = ()

    def load(self, deployments: tuple[VectorStoreDeployment, ...]) -> None:
        self.deployments = deployments

    def group(self, name: str) -> tuple[VectorStoreDeployment, ...]:
        members: Final = tuple(deployment for deployment in self.deployments if deployment.group == name)
        if not members:
            raise ValueError(f"vector_store_deployment '{name}' is not defined in vector_store_deployments")
        return tuple(sorted(members, key=lambda deployment: deployment.order))


vector_store_deployment_registry: Final = VectorStoreDeploymentRegistry()
vector_store_deployment_cooldowns: Final = DualCache()


def _cooldown_key(deployment: VectorStoreDeployment) -> str:
    return f"vector_store_deployment_cooldown:{deployment.group}:{deployment.deployment_id}"


def _failures_key(deployment: VectorStoreDeployment) -> str:
    return f"vector_store_deployment_failures:{deployment.group}:{deployment.deployment_id}"


def _store_problem(store: object, deployments: tuple[VectorStoreDeployment, ...]) -> str | None:
    fields: Final = as_str_mapping(store) or {}
    params: Final = as_str_mapping(fields.get("litellm_params")) or {}
    group: Final = params.get(VECTOR_STORE_DEPLOYMENT_PARAM)
    if group is None:
        return None
    name: Final = fields.get("vector_store_name")
    members: Final = tuple(deployment for deployment in deployments if deployment.group == group)
    if not members:
        return f"vector store '{name}' uses vector_store_deployment '{group}', which is not defined"
    provider: Final = members[0].custom_llm_provider
    if params.get("custom_llm_provider") != provider:
        return f"vector store '{name}' has custom_llm_provider '{params.get('custom_llm_provider')}' but '{group}' is {provider}"
    owned: Final = frozenset[str]().union(*(member.connection.keys() for member in members))
    clash: Final = sorted((_CONNECTION_PARAMS | (owned - {"custom_llm_provider"})) & params.keys())
    if clash:
        return f"vector store '{name}' sets {clash}, which its vector_store_deployment '{group}' owns"
    return None


def _group_problems(deployments: tuple[VectorStoreDeployment, ...]) -> tuple[str, ...]:
    groups: Final = frozenset(deployment.group for deployment in deployments)
    providers: Final = {
        group: frozenset(d.custom_llm_provider for d in deployments if d.group == group) for group in groups
    }
    mixed: Final = tuple(
        f"vector_store_deployments '{group}' mixes providers {sorted(found)}"
        for group, found in providers.items()
        if len(found) > 1
    )
    ids: Final = tuple((deployment.group, deployment.deployment_id) for deployment in deployments)
    repeated: Final = tuple(
        f"vector_store_deployments '{group}' repeats deployment_id '{deployment_id}'"
        for group, deployment_id in sorted(frozenset(ids))
        if ids.count((group, deployment_id)) > 1
    )
    return mixed + repeated


def vector_store_deployment_problems(config: object) -> tuple[str, ...]:
    """Why the config's vector_store_deployments, or the stores that use them, cannot serve."""
    deployments: Final = parse_vector_store_deployments(config)
    stores: Final = as_object_list(_CONFIG.validate_python(config).get("vector_store_registry")) or ()
    return _group_problems(deployments) + tuple(
        problem for store in stores if (problem := _store_problem(store, deployments)) is not None
    )


def connection_override_keys(group: str) -> frozenset[str]:
    """The request keys that would replace a setting the group's deployments own."""
    owned: Final = frozenset[str]().union(
        *(deployment.connection.keys() for deployment in vector_store_deployment_registry.group(group))
    )
    return _CONNECTION_PARAMS | (owned - {"custom_llm_provider"})


def on_deployment(store_params: Mapping[str, object], deployment: VectorStoreDeployment) -> Mapping[str, object]:
    """A grouped store's params with the deployment's connection, which no store or request value replaces."""
    return MappingProxyType(
        {
            **{key: value for key, value in store_params.items() if key != VECTOR_STORE_DEPLOYMENT_PARAM},
            **deployment.connection,
        }
    )


def primary_connection(store_params: Mapping[str, object]) -> Mapping[str, object]:
    """A grouped store's params on its lowest-order deployment, for paths that do not fail over."""
    group: Final = group_of(store_params)
    if group is None:
        return store_params
    return on_deployment(store_params, vector_store_deployment_registry.group(group)[0])


def served_api_base(deployment: VectorStoreDeployment) -> str:
    """The deployment's endpoint without the userinfo or query string a URL can carry credentials in."""
    api_base: Final = deployment.connection.get("api_base")
    if not isinstance(api_base, str):
        return ""
    parts: Final = urlsplit(api_base)
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


# --- the failover loop -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Served(Generic[_Value]):
    deployment: VectorStoreDeployment
    value: _Value
    failed_over: int


@dataclass(frozen=True, slots=True)
class Failed(Generic[_Value]):
    """A deployment's failure. ``cools_down`` is false for an error tied to the request or the index."""

    deployment: VectorStoreDeployment
    retryable: bool
    cools_down: bool
    error: Exception | None
    value: _Value | None = None


Attempt: TypeAlias = Served[_Value] | Failed[_Value]


def classify_status(status: int) -> tuple[bool, bool]:
    """(retryable, cools_down) for an HTTP status from a search deployment."""
    if status == 404:
        return True, False
    retryable: Final = should_retry(status)
    return retryable, retryable


_TRANSPORT_ERRORS: Final = (
    litellm.Timeout,
    litellm.APIConnectionError,
    httpx.TransportError,
    aiohttp.ClientConnectionError,
    TimeoutError,
)


def _raised_by_transport(error: BaseException | None) -> bool:
    """Whether the error, or one it was raised from, means the deployment could not be reached.

    Passthrough reports an unreachable host as a ProxyException raised while handling the aiohttp error."""
    if error is None:
        return False
    return isinstance(error, _TRANSPORT_ERRORS) or _raised_by_transport(error.__cause__ or error.__context__)


def classify_error(deployment: VectorStoreDeployment, error: Exception) -> Failed[_Value]:
    if _raised_by_transport(error):
        return Failed(deployment=deployment, retryable=True, cools_down=True, error=error)
    status: Final = getattr(error, "status_code", None)
    retryable, cools_down = classify_status(status) if isinstance(status, int) else (False, False)
    return Failed(deployment=deployment, retryable=retryable, cools_down=cools_down, error=error)


async def _cooldown_ends(keys: Sequence[str]) -> Mapping[str, object]:
    """Each key's cooldown end, read from Redis when the proxy shares one so every worker sees the same state."""
    redis: Final = vector_store_deployment_cooldowns.redis_cache
    if redis is not None:
        try:
            return _CACHED.validate_python(await redis.async_batch_get_cache(key_list=list(keys)))
        except Exception as error:  # noqa: BLE001  # Redis down must not stop healthy searches; this worker's state stands in
            verbose_logger.warning("vector store deployment cooldowns: Redis read failed, using local state: %s", error)
    values: Final[object] = await vector_store_deployment_cooldowns.in_memory_cache.async_batch_get_cache(
        keys=list(keys)
    )  # pyright: ignore[reportUnknownMemberType]  # the in-memory cache's reads are untyped
    return MappingProxyType(dict(zip(keys, as_object_list(values) or (), strict=False)))


def _still_cooling(cooldown_end: object, now: float) -> bool:
    try:
        return float(str(cooldown_end)) > now
    except ValueError:
        return False


async def _cooled_down_ids(deployments: tuple[VectorStoreDeployment, ...]) -> frozenset[str]:
    ends: Final = await _cooldown_ends([_cooldown_key(deployment) for deployment in deployments])
    now: Final = time.time()
    return frozenset(
        deployment.deployment_id
        for deployment in deployments
        if _still_cooling(ends.get(_cooldown_key(deployment)), now)
    )


async def _record_failure(failure: Failed[_Value]) -> None:
    """Counts the failure in a one-minute window and cools the deployment down once it reaches ``allowed_fails``."""
    deployment: Final = failure.deployment
    failures: Final = await vector_store_deployment_cooldowns.async_increment_cache(  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
        _failures_key(deployment), 1, ttl=_FAILURE_WINDOW_SECONDS
    )
    if deployment.cooldown_time <= 0 or (failures is not None and failures < deployment.allowed_fails):
        return
    verbose_logger.warning(
        "vector store deployment %s of group %s cooled down for %ss: %s",
        deployment.deployment_id,
        deployment.group,
        deployment.cooldown_time,
        str(failure.error)[:200],
    )
    await vector_store_deployment_cooldowns.async_set_cache(  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
        _cooldown_key(deployment), time.time() + deployment.cooldown_time, ttl=math.ceil(deployment.cooldown_time)
    )


def _attempt_order(
    deployments: tuple[VectorStoreDeployment, ...], cooled: frozenset[str]
) -> tuple[VectorStoreDeployment, ...]:
    """Deployments by order, those in cooldown last, so a read is still tried when every one is cooling."""
    return (
        *(d for d in deployments if d.deployment_id not in cooled),
        *(d for d in deployments if d.deployment_id in cooled),
    )


async def with_failover(
    group: str, attempt: Callable[[VectorStoreDeployment], Awaitable[Attempt[_Value]]]
) -> Attempt[_Value]:
    """Runs ``attempt`` on each deployment of ``group`` in order until one serves or one fails for good."""
    deployments: Final = vector_store_deployment_registry.group(group)
    return await _next(_attempt_order(deployments, await _cooled_down_ids(deployments)), attempt, 0)


async def _next(
    remaining: tuple[VectorStoreDeployment, ...],
    attempt: Callable[[VectorStoreDeployment], Awaitable[Attempt[_Value]]],
    failed_over: int,
) -> Attempt[_Value]:
    outcome: Final = await attempt(remaining[0])
    if isinstance(outcome, Served):
        return Served(deployment=outcome.deployment, value=outcome.value, failed_over=failed_over)
    if outcome.cools_down:
        await _record_failure(outcome)
    if not outcome.retryable or len(remaining) == 1:
        return outcome
    return await _next(remaining[1:], attempt, failed_over + 1)


def served_headers(served: Served[_Value]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            SERVED_BY_HEADER: served.deployment.deployment_id,
            "x-litellm-attempted-fallbacks": str(served.failed_over),
        }
    )


def group_of(store_params: Mapping[str, object]) -> str | None:
    group: Final = store_params.get(VECTOR_STORE_DEPLOYMENT_PARAM)
    return group if isinstance(group, str) else None

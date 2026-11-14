#### What this does ####
#   identifies lowest tpm deployment
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_logger, verbose_router_logger
from litellm.constants import USAGE_BASED_ROUTING_COUNTED_TOKENS_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type
from litellm.litellm_core_utils.deployment_pricing import deployment_backend_models
from litellm.litellm_core_utils.token_counter import embedding_aware_token_count
from litellm.redis.stores.router_usage import (
    ROUTER_USAGE_TARGET,
    CounterIncrement,
    RouterUsage,
    UsageCounter,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import get_counted_usage_tokens
from litellm.types.router import RouterErrors
from litellm.types.utils import LiteLLMPydanticObjectBase, StandardLoggingPayload
from litellm.utils import get_utc_datetime, is_health_check_probe_log, print_verbose

from .base_routing_strategy import BaseRoutingStrategy

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


class RoutingArgs(LiteLLMPydanticObjectBase):
    ttl: int = 1 * 60  # 1min (RPM/TPM expire key)


class _SuccessKwargs(TypedDict, total=False):
    litellm_params: ReadOnly[Mapping[str, object] | None]
    standard_logging_object: ReadOnly[object]


_SUCCESS_KWARGS_ADAPTER: Final = TypeAdapter(_SuccessKwargs)


_active_prefetched_usage: Final[ContextVar["PrefetchedUsage | None"]] = ContextVar("prefetched_usage", default=None)


@dataclass(frozen=True)
class PrefetchedUsage:
    """
    tpm/rpm counter values another read of this request already fetched from the usage store.

    `values` is None when that read failed, which is what `async_batch_get_cache` returns on failure.
    """

    counters: frozenset[UsageCounter]
    values: Mapping[UsageCounter, float | None] | None

    def covers(self, counters: Sequence[UsageCounter]) -> bool:
        return self.counters.issuperset(counters)

    def values_for(self, counters: Sequence[UsageCounter]) -> list[float | None] | None:
        if self.values is None:
            return None
        return [self.values.get(counter) for counter in counters]

    @staticmethod
    @contextmanager
    def scoped(usage: "PrefetchedUsage | None") -> Iterator[None]:
        token: Final = _active_prefetched_usage.set(usage)
        try:
            yield
        finally:
            _active_prefetched_usage.reset(token)

    @staticmethod
    def active() -> "PrefetchedUsage | None":
        return _active_prefetched_usage.get()


class LowestTPMLoggingHandler_v2(BaseRoutingStrategy, CustomLogger):
    """
    Updated version of TPM/RPM Logging.

    Meant to work across instances.

    Caches individual models, not model_groups

    Uses batch get (redis.mget)

    Increments tpm/rpm limit using redis.incr
    """

    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour

    def __init__(self, usage: RouterUsage, routing_args: dict = {}):
        self.routing_args = RoutingArgs(**routing_args)
        BaseRoutingStrategy.__init__(
            self,
            usage=usage,
            should_batch_redis_writes=True,
            default_sync_interval=0.1,
        )

    @with_service_target(ROUTER_USAGE_TARGET)
    def pre_call_check(self, deployment: dict) -> dict | None:
        """
        Pre-call check + update model rpm

        Returns - deployment

        Raises - RateLimitError if deployment over defined RPM limit
        """
        try:
            # ------------
            # Setup values
            # ------------

            dt: Final = get_utc_datetime()
            current_minute: Final = dt.strftime("%H-%M")
            model_id: Final[object] = deployment.get("model_info", {}).get("id")
            deployment_name: Final[object] = deployment.get("litellm_params", {}).get("model")
            rpm_counter: Final = UsageCounter.for_deployment(f"{model_id}", f"{deployment_name}", "rpm", current_minute)

            local_result: Final = self.usage.get_local(rpm_counter)  # check local result first

            deployment_rpm = None
            if deployment_rpm is None:
                deployment_rpm = deployment.get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("litellm_params", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("model_info", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = float("inf")

            if local_result is not None and local_result >= deployment_rpm:
                raise litellm.RateLimitError(
                    message=f"Deployment over defined rpm limit={deployment_rpm}. current usage={local_result}",
                    llm_provider="",
                    model=deployment.get("litellm_params", {}).get("model"),
                    response=httpx.Response(
                        status_code=429,
                        content="{} rpm limit={}. current usage={}. id={}, model_group={}. Get the model info by calling 'router.get_model_info(id)".format(
                            RouterErrors.user_defined_ratelimit_error.value,
                            deployment_rpm,
                            local_result,
                            model_id,
                            deployment.get("model_name", ""),
                        ),
                        request=httpx.Request(
                            method="tpm_rpm_limits",
                            url="https://github.com/BerriAI/litellm",
                        ),
                    ),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks

                result: Final = self.usage.increment_sync(rpm_counter, 1, self.routing_args.ttl)
                if result is not None and result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message=f"Deployment over defined rpm limit={deployment_rpm}. current usage={result}",
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content=f"{RouterErrors.user_defined_ratelimit_error.value} rpm limit={deployment_rpm}. current usage={result}",
                            request=httpx.Request(
                                method="tpm_rpm_limits",
                                url="https://github.com/BerriAI/litellm",
                            ),
                        ),
                    )
            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_pre_call_check(self, deployment: dict, parent_otel_span: Span | None) -> dict | None:
        """
        Pre-call check + update model rpm
        - Used inside semaphore
        - raise rate limit error if deployment over limit

        Why? solves concurrency issue - https://github.com/BerriAI/litellm/issues/2994

        Returns - deployment

        Raises - RateLimitError if deployment over defined RPM limit
        """
        try:
            # ------------
            # Setup values
            # ------------
            dt: Final = get_utc_datetime()
            current_minute: Final = dt.strftime("%H-%M")
            model_id: Final[object] = deployment.get("model_info", {}).get("id")
            deployment_name: Final[object] = deployment.get("litellm_params", {}).get("model")

            rpm_counter: Final = UsageCounter.for_deployment(f"{model_id}", f"{deployment_name}", "rpm", current_minute)
            local_result: Final = self.usage.get_local(rpm_counter)  # check local result first

            deployment_rpm = None
            if deployment_rpm is None:
                deployment_rpm = deployment.get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("litellm_params", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("model_info", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = float("inf")
            if local_result is not None and local_result >= deployment_rpm:
                raise litellm.RateLimitError(
                    message=f"Deployment over defined rpm limit={deployment_rpm}. current usage={local_result}",
                    llm_provider="",
                    model=deployment.get("litellm_params", {}).get("model"),
                    response=httpx.Response(
                        status_code=429,
                        content=f"{RouterErrors.user_defined_ratelimit_error.value} rpm limit={deployment_rpm}. current usage={local_result}",
                        headers={"retry-after": str(60)},
                        request=httpx.Request(
                            method="tpm_rpm_limits",
                            url="https://github.com/BerriAI/litellm",
                        ),
                    ),
                    num_retries=deployment.get("num_retries"),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks
                result = await self._increment_value_in_current_window(rpm_counter, 1, self.routing_args.ttl)
                if result is not None and result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message=f"Deployment over defined rpm limit={deployment_rpm}. current usage={result}",
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content=f"{RouterErrors.user_defined_ratelimit_error.value} rpm limit={deployment_rpm}. current usage={result}",
                            headers={"retry-after": str(60)},
                            request=httpx.Request(
                                method="tpm_rpm_limits",
                                url="https://github.com/BerriAI/litellm",
                            ),
                        ),
                        num_retries=deployment.get("num_retries"),
                    )
            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

    @with_service_target(ROUTER_USAGE_TARGET)
    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        if is_batch_retrieve_call_type(kwargs.get("call_type")):
            return
        try:
            """
            Update TPM/RPM usage on success
            """
            standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group: Final = standard_logging_object.get("model_group")
            model: Final[object] = standard_logging_object["hidden_params"].get("litellm_model_name")
            id: object = standard_logging_object.get("model_id")
            if (
                model_group is None
                or id is None
                or model is None
                or is_health_check_probe_log(
                    _SUCCESS_KWARGS_ADAPTER.validate_python(kwargs).get("standard_logging_object")
                )
            ):
                return
            elif isinstance(id, int):
                id = str(id)

            total_tokens: Final = standard_logging_object.get("total_tokens")

            # ------------
            # Setup values
            # ------------
            dt: Final = get_utc_datetime()
            current_minute: Final = dt.strftime("%H-%M")  # use the same timezone regardless of system clock

            ## TPM
            self.usage.increment_sync(
                UsageCounter.for_deployment(f"{id}", f"{model}", "tpm", current_minute),
                total_tokens,
                self.routing_args.ttl,
            )
            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.lowest_tpm_rpm_v2.py::log_success_event(): Exception occured - %s", e
            )

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        if is_batch_retrieve_call_type(kwargs.get("call_type")):
            return
        try:
            """
            Update TPM usage on success
            """
            standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group: Final = standard_logging_object.get("model_group")
            model: Final = standard_logging_object["hidden_params"]["litellm_model_name"]
            model_id: Final = standard_logging_object.get("model_id")
            total_tokens: Final = standard_logging_object.get("total_tokens")
            if (
                model_group is None
                or not isinstance(model_id, (str, int))
                or not isinstance(total_tokens, int)
                or is_health_check_probe_log(
                    _SUCCESS_KWARGS_ADAPTER.validate_python(kwargs).get("standard_logging_object")
                )
            ):
                return
            counted_tokens: Final = get_counted_usage_tokens(
                _SUCCESS_KWARGS_ADAPTER.validate_python(kwargs).get("litellm_params") or {},
                USAGE_BASED_ROUTING_COUNTED_TOKENS_METADATA_KEY,
            )
            await self.async_increment_tpm(
                deployment_id=str(model_id),
                deployment_name=model if isinstance(model, str) else None,
                tokens=total_tokens if counted_tokens is None else max(0, total_tokens - counted_tokens),
            )

            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.lowest_tpm_rpm_v2.py::async_log_success_event(): Exception occured - %s", e
            )

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_increment_tpm(self, *, deployment_id: str, deployment_name: str | None, tokens: int) -> None:
        current_minute: Final = get_utc_datetime().strftime("%H-%M")
        counter: Final = UsageCounter.for_deployment(deployment_id, f"{deployment_name}", "tpm", current_minute)
        await self.usage.increment_post_call((CounterIncrement(counter, tokens, self.routing_args.ttl),))

    def _return_potential_deployments(
        self,
        healthy_deployments: list[dict],
        all_deployments: dict,
        input_tokens: int,
        rpm_dict: dict,
    ):
        lowest_tpm = float("inf")
        potential_deployments = []  # if multiple deployments have the same low value
        deployment_lookup: Final = {
            deployment.get("model_info", {}).get("id"): deployment for deployment in healthy_deployments
        }
        for item, item_tpm in all_deployments.items():
            ## get the item from model list
            item = item.split(":")[0]
            _deployment = deployment_lookup.get(item)
            if _deployment is None:
                continue  # skip to next one
            elif item_tpm is None:
                continue  # skip if unhealthy deployment

            _deployment_tpm = None
            if _deployment_tpm is None:
                _deployment_tpm = _deployment.get("tpm")
            if _deployment_tpm is None:
                _deployment_tpm = _deployment.get("litellm_params", {}).get("tpm")
            if _deployment_tpm is None:
                _deployment_tpm = _deployment.get("model_info", {}).get("tpm")
            if _deployment_tpm is None:
                _deployment_tpm = float("inf")

            _deployment_rpm = None
            if _deployment_rpm is None:
                _deployment_rpm = _deployment.get("rpm")
            if _deployment_rpm is None:
                _deployment_rpm = _deployment.get("litellm_params", {}).get("rpm")
            if _deployment_rpm is None:
                _deployment_rpm = _deployment.get("model_info", {}).get("rpm")
            if _deployment_rpm is None:
                _deployment_rpm = float("inf")
            if item_tpm + input_tokens > _deployment_tpm or (
                (rpm_dict is not None and item in rpm_dict)
                and rpm_dict[item] is not None
                and (rpm_dict[item] + 1 >= _deployment_rpm)
            ):
                continue
            elif item_tpm == lowest_tpm:
                potential_deployments.append(_deployment)
            elif item_tpm < lowest_tpm:
                lowest_tpm = item_tpm
                potential_deployments = [_deployment]
        return potential_deployments

    def _common_checks_available_deployment(
        self,
        model_group: str,
        healthy_deployments: list,
        tpm_counters: Sequence[UsageCounter],
        tpm_values: Sequence[float | None] | None,
        rpm_counters: Sequence[UsageCounter],
        rpm_values: Sequence[float | None] | None,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
    ) -> dict | None:
        """
        Common checks for get available deployment, across sync + async implementations
        """

        if tpm_values is None or rpm_values is None:
            return None

        tpm_dict = {
            counter.deployment_id.split(":")[0]: value for counter, value in zip(tpm_counters, tpm_values)
        }  # {model_id: 1, ..}
        rpm_dict: Final = {
            counter.deployment_id.split(":")[0]: value for counter, value in zip(rpm_counters, rpm_values)
        }  # {model_id: 1, ..}

        try:
            input_tokens = embedding_aware_token_count(
                messages, input, partial(deployment_backend_models, healthy_deployments)
            )
        except Exception:
            input_tokens = 0
        verbose_router_logger.debug("input_tokens=%s", input_tokens)
        # -----------------------
        # Find lowest used model
        # ----------------------

        if tpm_dict is None:  # base case - none of the deployments have been used
            # initialize a tpm dict with {model_id: 0}
            tpm_dict = {}
            for deployment in healthy_deployments:
                tpm_dict[deployment["model_info"]["id"]] = 0
        else:
            for d in healthy_deployments:
                ## if healthy deployment not yet used
                tpm_key = d["model_info"]["id"]
                if tpm_key not in tpm_dict or tpm_dict[tpm_key] is None:
                    tpm_dict[tpm_key] = 0

        all_deployments: Final = tpm_dict
        potential_deployments: Final = self._return_potential_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
            input_tokens=input_tokens,
            rpm_dict=rpm_dict,
        )
        print_verbose("returning picked lowest tpm/rpm deployment.")

        if len(potential_deployments) > 0:
            return random.choice(potential_deployments)
        else:
            return None

    def usage_counters(
        self, healthy_deployments: Sequence[object]
    ) -> tuple[tuple[UsageCounter, ...], tuple[UsageCounter, ...]]:
        """The tpm and rpm counters selection reads, this minute's for each deployment."""
        current_minute: Final = get_utc_datetime().strftime("%H-%M")
        named: Final[tuple[tuple[object, object], ...]] = tuple(
            (m.get("model_info", {}).get("id"), m.get("litellm_params", {}).get("model"))
            for m in healthy_deployments
            if isinstance(m, dict)
        )
        return (
            tuple(
                UsageCounter.for_deployment(f"{model_id}", f"{model}", "tpm", current_minute)
                for model_id, model in named
            ),
            tuple(
                UsageCounter.for_deployment(f"{model_id}", f"{model}", "rpm", current_minute)
                for model_id, model in named
            ),
        )

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
    ):
        """
        Async implementation of get deployments.

        Reduces time to retrieve the tpm/rpm values from cache. A `PrefetchedUsage` scoped
        to this request skips the cache read when it already holds its counters (see
        `RoutingReadBatch`).
        """
        # get list of potential deployments
        verbose_router_logger.debug(
            "get_available_deployments - Usage Based. model_group: %s, healthy_deployments: %s",
            model_group,
            healthy_deployments,
        )

        tpm_counters, rpm_counters = self.usage_counters(healthy_deployments)
        combined_tpm_rpm_counters: Final = tpm_counters + rpm_counters

        prefetched_usage: Final = PrefetchedUsage.active()
        if prefetched_usage is not None and prefetched_usage.covers(combined_tpm_rpm_counters):
            combined_tpm_rpm_values = prefetched_usage.values_for(combined_tpm_rpm_counters)
        else:
            combined_tpm_rpm_values = await self.usage.get_many(combined_tpm_rpm_counters)  # [1, 2, None, ..]

        if combined_tpm_rpm_values is not None:
            tpm_values = combined_tpm_rpm_values[: len(tpm_counters)]
            rpm_values = combined_tpm_rpm_values[len(tpm_counters) :]
        else:
            tpm_values = None
            rpm_values = None

        deployment: Final = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_counters=tpm_counters,
            tpm_values=tpm_values,
            rpm_counters=rpm_counters,
            rpm_values=rpm_values,
            messages=messages,
            input=input,
        )

        try:
            assert deployment is not None
            return deployment
        except Exception:
            ### GET THE DICT OF TPM / RPM + LIMITS PER DEPLOYMENT ###
            deployment_dict: Final = {}
            for index, _deployment in enumerate(healthy_deployments):
                if isinstance(_deployment, dict):
                    id = _deployment.get("model_info", {}).get("id")
                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_tpm = None
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("litellm_params", {}).get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("model_info", {}).get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = float("inf")

                    ### GET CURRENT TPM ###
                    current_tpm = tpm_values[index] if tpm_values else 0

                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_rpm = None
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("litellm_params", {}).get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("model_info", {}).get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = float("inf")

                    ### GET CURRENT RPM ###
                    current_rpm = rpm_values[index] if rpm_values else 0

                    deployment_dict[id] = {
                        "current_tpm": current_tpm,
                        "tpm_limit": _deployment_tpm,
                        "current_rpm": current_rpm,
                        "rpm_limit": _deployment_rpm,
                    }
            raise litellm.RateLimitError(
                message=f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}",
                llm_provider="",
                model=model_group,
                response=httpx.Response(
                    status_code=429,
                    content="",
                    headers={"retry-after": str(60)},
                    request=httpx.Request(
                        method="tpm_rpm_limits",
                        url="https://github.com/BerriAI/litellm",
                    ),
                ),
            )

    @with_service_target(ROUTER_USAGE_TARGET)
    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        parent_otel_span: Span | None = None,
    ):
        """
        Returns a deployment with the lowest TPM/RPM usage.
        """
        # get list of potential deployments
        verbose_router_logger.debug(
            "get_available_deployments - Usage Based. model_group: %s, healthy_deployments: %s",
            model_group,
            healthy_deployments,
        )

        tpm_counters, rpm_counters = self.usage_counters(healthy_deployments)
        tpm_values: Final = self.usage.get_many_sync(tpm_counters)  # [1, 2, None, ..]
        rpm_values: Final = self.usage.get_many_sync(rpm_counters)  # [1, 2, None, ..]

        deployment: Final = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_counters=tpm_counters,
            tpm_values=tpm_values,
            rpm_counters=rpm_counters,
            rpm_values=rpm_values,
            messages=messages,
            input=input,
        )

        try:
            assert deployment is not None
            return deployment
        except Exception:
            ### GET THE DICT OF TPM / RPM + LIMITS PER DEPLOYMENT ###
            deployment_dict: Final = {}
            for index, _deployment in enumerate(healthy_deployments):
                if isinstance(_deployment, dict):
                    id = _deployment.get("model_info", {}).get("id")
                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_tpm = None
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("litellm_params", {}).get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("model_info", {}).get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = float("inf")

                    ### GET CURRENT TPM ###
                    current_tpm = tpm_values[index] if tpm_values else 0

                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_rpm = None
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("litellm_params", {}).get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("model_info", {}).get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = float("inf")

                    ### GET CURRENT RPM ###
                    current_rpm = rpm_values[index] if rpm_values else 0

                    deployment_dict[id] = {
                        "current_tpm": current_tpm,
                        "tpm_limit": _deployment_tpm,
                        "current_rpm": current_rpm,
                        "rpm_limit": _deployment_rpm,
                    }
            raise ValueError(
                f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}"
            )

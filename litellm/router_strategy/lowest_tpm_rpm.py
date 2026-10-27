#### What this does ####
#   identifies lowest tpm deployment
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final

from litellm import token_counter
from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.batch_utils import is_batch_retrieve_call_type
from litellm.types.utils import LiteLLMPydanticObjectBase
from litellm.utils import print_verbose


class RoutingArgs(LiteLLMPydanticObjectBase):
    ttl: int = 1 * 60  # 1min (RPM/TPM expire key)


class LowestTPMLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour

    def __init__(self, router_cache: DualCache, routing_args: dict = {}):
        self.router_cache = router_cache
        self.routing_args = RoutingArgs(**routing_args)

    def _cached_usage(self, usage_key: str) -> Mapping[str, float]:
        """The {deployment_id: count} dict this process has recorded for a model group this minute."""
        cached: Final = self.router_cache.get_cache(key=usage_key, local_only=True)
        if not isinstance(cached, Mapping):
            return {}
        return {k: float(v) for k, v in cached.items() if isinstance(k, str) and isinstance(v, (int, float))}

    def _record_usage(self, usage_key: str, deployment_id: str, total: float | None) -> None:
        """Copy a deployment's running total into the in-memory {deployment_id: count} dict for its
        model group, which routing and the router's pre-call RPM check read. Nothing awaits between
        the read and the write, so concurrent callbacks in this process can't overwrite each other."""
        if total is None:
            return
        recorded: Final = self._cached_usage(usage_key)
        usage: Final = {**recorded, deployment_id: max(recorded.get(deployment_id, 0.0), total)}
        self.router_cache.set_cache(key=usage_key, value=usage, local_only=True, ttl=self.routing_args.ttl)

    def _get_usage(self, usage_key: str, deployment_ids: Sequence[str]) -> Mapping[str, float]:
        """This process's recorded usage, plus the shared counter for any deployment it hasn't served
        this minute (read from Redis, if there is one, so other instances' traffic counts).

        Counters are kept per deployment, f"{usage_key}:{deployment_id}", so each success is one
        atomic increment (INCRBYFLOAT in Redis) instead of a read-modify-write of a shared dict,
        which loses updates when callbacks run concurrently."""
        recorded: Final = self._cached_usage(usage_key)
        unseen: Final = tuple(i for i in deployment_ids if i not in recorded)
        if not unseen:
            return recorded
        counts: Final = self.router_cache.batch_get_cache(keys=[f"{usage_key}:{i}" for i in unseen])
        if not isinstance(counts, Sequence):
            return recorded
        shared: Final = {i: float(c) for i, c in zip(unseen, counts) if c is not None}
        return {**shared, **recorded}

    @with_service_target("router_usage")
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        if is_batch_retrieve_call_type(kwargs.get("call_type")):
            return
        try:
            """
            Update TPM/RPM usage on success
            """
            if "litellm_params" not in kwargs or kwargs["litellm_params"] is None:
                return
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                total_tokens: Final = response_obj["usage"]["total_tokens"]

                # ------------
                # Setup values
                # ------------
                current_minute: Final = datetime.now().strftime("%H-%M")
                tpm_key: Final = f"{model_group}:tpm:{current_minute}"
                rpm_key: Final = f"{model_group}:rpm:{current_minute}"

                # ------------
                # Update usage
                # ------------

                if not isinstance(id, str) or not isinstance(total_tokens, int):
                    return

                ## TPM
                tpm_total: Final = self.router_cache.increment_cache(
                    key=f"{tpm_key}:{id}", value=total_tokens, ttl=self.routing_args.ttl
                )
                self._record_usage(tpm_key, id, tpm_total)

                ## RPM
                rpm_total: Final = self.router_cache.increment_cache(
                    key=f"{rpm_key}:{id}", value=1, ttl=self.routing_args.ttl
                )
                self._record_usage(rpm_key, id, rpm_total)

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            verbose_router_logger.error(
                "litellm.router_strategy.lowest_tpm_rpm.py::async_log_success_event(): Exception occured - %s", e
            )
            verbose_router_logger.debug(traceback.format_exc())

    @with_service_target("router_usage")
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if is_batch_retrieve_call_type(kwargs.get("call_type")):
            return
        try:
            """
            Update TPM/RPM usage on success
            """
            if "litellm_params" not in kwargs or kwargs["litellm_params"] is None:
                return
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)

                model_info: Final = kwargs["litellm_params"].get("model_info")
                id = None
                if model_info is not None and isinstance(model_info, dict):
                    id = model_info.get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                if "usage" not in response_obj:
                    return
                total_tokens: Final = response_obj["usage"]["total_tokens"]

                # ------------
                # Setup values
                # ------------
                current_minute: Final = datetime.now().strftime("%H-%M")
                tpm_key: Final = f"{model_group}:tpm:{current_minute}"
                rpm_key: Final = f"{model_group}:rpm:{current_minute}"

                # ------------
                # Update usage
                # ------------
                # update cache

                if not isinstance(id, str) or not isinstance(total_tokens, (int, float)):
                    return

                ## TPM
                tpm_total: Final = await self.router_cache.async_increment_cache(
                    key=f"{tpm_key}:{id}", value=total_tokens, ttl=self.routing_args.ttl
                )
                self._record_usage(tpm_key, id, tpm_total)

                ## RPM
                rpm_total: Final = await self.router_cache.async_increment_cache(
                    key=f"{rpm_key}:{id}", value=1, ttl=self.routing_args.ttl
                )
                self._record_usage(rpm_key, id, rpm_total)

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            verbose_router_logger.exception(
                "litellm.router_strategy.lowest_tpm_rpm.py::async_log_success_event(): Exception occured - %s", e
            )
            verbose_router_logger.debug(traceback.format_exc())

    @with_service_target("router_usage")
    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
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
        current_minute: Final = datetime.now().strftime("%H-%M")
        tpm_key: Final = f"{model_group}:tpm:{current_minute}"
        rpm_key: Final = f"{model_group}:rpm:{current_minute}"

        deployment_ids: Final = tuple(str(d["model_info"]["id"]) for d in healthy_deployments)
        tpm_dict: Final = dict(self._get_usage(tpm_key, deployment_ids))
        rpm_dict: Final = self._get_usage(rpm_key, deployment_ids)

        verbose_router_logger.debug("tpm_key=%s, tpm_dict: %s, rpm_dict: %s", tpm_key, tpm_dict, rpm_dict)
        try:
            input_tokens = token_counter(messages=messages, text=input)
        except Exception:
            input_tokens = 0
        verbose_router_logger.debug("input_tokens=%s", input_tokens)
        # -----------------------
        # Find lowest used model
        # ----------------------
        lowest_tpm = float("inf")

        for deployment_id in deployment_ids:  ## if healthy deployment not yet used
            _ = tpm_dict.setdefault(deployment_id, 0)

        all_deployments: Final = tpm_dict

        deployment = None
        for item, item_tpm in all_deployments.items():
            ## get the item from model list
            _deployment = None
            for m in healthy_deployments:
                if item == m["model_info"]["id"]:
                    _deployment = m

            if _deployment is None:
                continue  # skip to next one

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

            if (
                item_tpm + input_tokens > _deployment_tpm
                or (item in rpm_dict)
                and (rpm_dict[item] + 1 >= _deployment_rpm)
            ):
                continue
            elif item_tpm < lowest_tpm:
                lowest_tpm = item_tpm
                deployment = _deployment
        print_verbose("returning picked lowest tpm/rpm deployment.")
        return deployment

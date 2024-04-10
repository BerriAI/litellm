#### What this does ####
#   identifies lowest tpm deployment

import dotenv, os, requests, random
from typing import Optional, Union, List, Dict
import datetime as datetime_og
from datetime import datetime

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback, asyncio
from litellm import token_counter
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_router_logger
from litellm.utils import print_verbose


class LowestTPMLoggingHandler_v2(CustomLogger):
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

    def __init__(self, router_cache: DualCache, model_list: list):
        self.router_cache = router_cache
        self.model_list = model_list

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM/RPM usage on success
            """
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                total_tokens = response_obj["usage"]["total_tokens"]

                # ------------
                # Setup values
                # ------------
                current_minute = datetime.now(datetime_og.UTC).strftime("%H-%M")
                tpm_key = f"{model_group}:tpm:{current_minute}"
                rpm_key = f"{model_group}:rpm:{current_minute}"

                # ------------
                # Update usage
                # ------------

                ## TPM
                request_count_dict = self.router_cache.get_cache(key=tpm_key) or {}
                request_count_dict[id] = request_count_dict.get(id, 0) + total_tokens

                self.router_cache.set_cache(key=tpm_key, value=request_count_dict)

                ## RPM
                request_count_dict = self.router_cache.get_cache(key=rpm_key) or {}
                request_count_dict[id] = request_count_dict.get(id, 0) + 1

                self.router_cache.set_cache(key=rpm_key, value=request_count_dict)

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            traceback.print_exc()
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM/RPM usage on success
            """
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                total_tokens = response_obj["usage"]["total_tokens"]

                # ------------
                # Setup values
                # ------------
                current_minute = datetime.now(datetime_og.UTC).strftime(
                    "%H-%M"
                )  # use the same timezone regardless of system clock

                tpm_key = f"{id}:tpm:{current_minute}"
                rpm_key = f"{id}:rpm:{current_minute}"

                # ------------
                # Update usage
                # ------------
                # update cache

                ## TPM
                await self.router_cache.async_increment_cache(
                    key=tpm_key, value=total_tokens
                )
                ## RPM
                await self.router_cache.async_increment_cache(key=rpm_key, value=1)

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            traceback.print_exc()
            pass

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ):
        """
        Async implementation of get deployments.

        Reduces time to retrieve the tpm/rpm values from cache
        """
        pass

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ):
        """
        Returns a deployment with the lowest TPM/RPM usage.
        """
        # get list of potential deployments
        verbose_router_logger.debug(
            f"get_available_deployments - Usage Based. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        current_minute = datetime.now(datetime_og.UTC).strftime("%H-%M")
        tpm_keys = []
        rpm_keys = []
        for m in healthy_deployments:
            if isinstance(m, dict):
                id = m.get("model_info", {}).get(
                    "id"
                )  # a deployment should always have an 'id'. this is set in router.py
                tpm_key = "{}:tpm:{}".format(id, current_minute)
                rpm_key = "{}:rpm:{}".format(id, current_minute)

                tpm_keys.append(tpm_key)
                rpm_keys.append(rpm_key)

        tpm_values = self.router_cache.batch_get_cache(
            keys=tpm_keys
        )  # [1, 2, None, ..]
        rpm_values = self.router_cache.batch_get_cache(
            keys=rpm_keys
        )  # [1, 2, None, ..]

        tpm_dict = {}  # {model_id: 1, ..}
        for idx, key in enumerate(tpm_keys):
            tpm_dict[tpm_keys[idx]] = tpm_values[idx]

        rpm_dict = {}  # {model_id: 1, ..}
        for idx, key in enumerate(rpm_keys):
            rpm_dict[rpm_keys[idx]] = rpm_values[idx]

        try:
            input_tokens = token_counter(messages=messages, text=input)
        except:
            input_tokens = 0
        verbose_router_logger.debug(f"input_tokens={input_tokens}")
        # -----------------------
        # Find lowest used model
        # ----------------------
        lowest_tpm = float("inf")

        if tpm_dict is None:  # base case - none of the deployments have been used
            # initialize a tpm dict with {model_id: 0}
            tpm_dict = {}
            for deployment in healthy_deployments:
                tpm_dict[deployment["model_info"]["id"]] = 0
        else:
            for d in healthy_deployments:
                ## if healthy deployment not yet used
                if d["model_info"]["id"] not in tpm_dict:
                    tpm_dict[d["model_info"]["id"]] = 0

        all_deployments = tpm_dict

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

            if item_tpm + input_tokens > _deployment_tpm:
                continue
            elif (rpm_dict is not None and item in rpm_dict) and (
                rpm_dict[item] + 1 > _deployment_rpm
            ):
                continue
            elif item_tpm < lowest_tpm:
                lowest_tpm = item_tpm
                deployment = _deployment
        print_verbose("returning picked lowest tpm/rpm deployment.")
        return deployment

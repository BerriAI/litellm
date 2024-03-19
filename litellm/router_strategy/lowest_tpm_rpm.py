#### What this does ####
#   identifies lowest tpm deployment

import dotenv, os, requests, random
from typing import Optional, Union, List, Dict
from datetime import datetime

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
from litellm import token_counter
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_router_logger


class LowestTPMLoggingHandler(CustomLogger):
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

                total_tokens = response_obj["usage"]["total_tokens"]

                # ------------
                # Setup values
                # ------------
                current_minute = datetime.now().strftime("%H-%M")
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

                total_tokens = response_obj["usage"]["total_tokens"]

                # ------------
                # Setup values
                # ------------
                current_minute = datetime.now().strftime("%H-%M")
                tpm_key = f"{model_group}:tpm:{current_minute}"
                rpm_key = f"{model_group}:rpm:{current_minute}"

                # ------------
                # Update usage
                # ------------
                # update cache

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
        current_minute = datetime.now().strftime("%H-%M")
        tpm_key = f"{model_group}:tpm:{current_minute}"
        rpm_key = f"{model_group}:rpm:{current_minute}"

        tpm_dict = self.router_cache.get_cache(key=tpm_key)
        rpm_dict = self.router_cache.get_cache(key=rpm_key)

        verbose_router_logger.debug(
            f"tpm_key={tpm_key}, tpm_dict: {tpm_dict}, rpm_dict: {rpm_dict}"
        )
        try:
            input_tokens = token_counter(messages=messages, text=input)
        except:
            input_tokens = 0
        verbose_router_logger.debug(f"input_tokens={input_tokens}")
        # -----------------------
        # Find lowest used model
        # ----------------------
        lowest_tpm = float("inf")
        deployment = None
        if tpm_dict is None:  # base case - none of the deployments have been used
            # Return the 1st deployment where deployment["tpm"] >= input_tokens
            for deployment in healthy_deployments:
                _deployment_tpm = (
                    deployment.get("tpm", None)
                    or deployment.get("litellm_params", {}).get("tpm", None)
                    or deployment.get("model_info", {}).get("tpm", None)
                    or float("inf")
                )

                if _deployment_tpm >= input_tokens:
                    return deployment
            return None

        all_deployments = tpm_dict
        for d in healthy_deployments:
            ## if healthy deployment not yet used
            if d["model_info"]["id"] not in all_deployments:
                all_deployments[d["model_info"]["id"]] = 0

        for item, item_tpm in all_deployments.items():
            ## get the item from model list
            _deployment = None
            for m in healthy_deployments:
                if item == m["model_info"]["id"]:
                    _deployment = m

            if _deployment is None:
                continue  # skip to next one

            _deployment_tpm = (
                _deployment.get("tpm", None)
                or _deployment.get("litellm_params", {}).get("tpm", None)
                or _deployment.get("model_info", {}).get("tpm", None)
                or float("inf")
            )

            _deployment_rpm = (
                _deployment.get("rpm", None)
                or _deployment.get("litellm_params", {}).get("rpm", None)
                or _deployment.get("model_info", {}).get("rpm", None)
                or float("inf")
            )

            if item_tpm == 0:
                deployment = _deployment
                break
            elif item_tpm + input_tokens > _deployment_tpm:
                continue
            elif (rpm_dict is not None and item in rpm_dict) and (
                rpm_dict[item] + 1 > _deployment_rpm
            ):
                continue
            elif item_tpm < lowest_tpm:
                lowest_tpm = item_tpm
                deployment = _deployment
        verbose_router_logger.info(f"returning picked lowest tpm/rpm deployment.")
        return deployment

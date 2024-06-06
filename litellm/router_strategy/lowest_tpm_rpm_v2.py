#### What this does ####
#   identifies lowest tpm deployment
from pydantic import BaseModel
import random
from typing import Optional, Union, List, Dict
import traceback
import httpx
import litellm
from litellm import token_counter
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_router_logger, verbose_logger
from litellm.utils import print_verbose, get_utc_datetime
from litellm.types.router import RouterErrors


class LiteLLMBase(BaseModel):
    """
    Implements default functions, all pydantic objects should have.
    """

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except Exception as e:
            # if using pydantic v1
            return self.dict()


class RoutingArgs(LiteLLMBase):
    ttl: int = 1 * 60  # 1min (RPM/TPM expire key)


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

    def __init__(
        self, router_cache: DualCache, model_list: list, routing_args: dict = {}
    ):
        self.router_cache = router_cache
        self.model_list = model_list
        self.routing_args = RoutingArgs(**routing_args)

    def pre_call_check(self, deployment: Dict) -> Optional[Dict]:
        """
        Pre-call check + update model rpm

        Returns - deployment

        Raises - RateLimitError if deployment over defined RPM limit
        """
        try:

            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            model_id = deployment.get("model_info", {}).get("id")
            rpm_key = f"{model_id}:rpm:{current_minute}"
            local_result = self.router_cache.get_cache(
                key=rpm_key, local_only=True
            )  # check local result first

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
                    message="Deployment over defined rpm limit={}. current usage={}".format(
                        deployment_rpm, local_result
                    ),
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
                        request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                    ),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks
                result = self.router_cache.increment_cache(
                    key=rpm_key, value=1, ttl=self.routing_args.ttl
                )
                if result is not None and result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message="Deployment over defined rpm limit={}. current usage={}".format(
                            deployment_rpm, result
                        ),
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content="{} rpm limit={}. current usage={}".format(
                                RouterErrors.user_defined_ratelimit_error.value,
                                deployment_rpm,
                                result,
                            ),
                            request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                        ),
                    )
            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

    async def async_pre_call_check(self, deployment: Dict) -> Optional[Dict]:
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
            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            model_id = deployment.get("model_info", {}).get("id")
            rpm_key = f"{model_id}:rpm:{current_minute}"
            local_result = await self.router_cache.async_get_cache(
                key=rpm_key, local_only=True
            )  # check local result first

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
                    message="Deployment over defined rpm limit={}. current usage={}".format(
                        deployment_rpm, local_result
                    ),
                    llm_provider="",
                    model=deployment.get("litellm_params", {}).get("model"),
                    response=httpx.Response(
                        status_code=429,
                        content="{} rpm limit={}. current usage={}".format(
                            RouterErrors.user_defined_ratelimit_error.value,
                            deployment_rpm,
                            local_result,
                        ),
                        request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                    ),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks
                result = await self.router_cache.async_increment_cache(
                    key=rpm_key, value=1, ttl=self.routing_args.ttl
                )
                if result is not None and result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message="Deployment over defined rpm limit={}. current usage={}".format(
                            deployment_rpm, result
                        ),
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content="{} rpm limit={}. current usage={}".format(
                                RouterErrors.user_defined_ratelimit_error.value,
                                deployment_rpm,
                                result,
                            ),
                            request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                        ),
                    )

            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

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
                dt = get_utc_datetime()
                current_minute = dt.strftime(
                    "%H-%M"
                )  # use the same timezone regardless of system clock

                tpm_key = f"{id}:tpm:{current_minute}"
                # ------------
                # Update usage
                # ------------
                # update cache

                ## TPM
                self.router_cache.increment_cache(
                    key=tpm_key, value=total_tokens, ttl=self.routing_args.ttl
                )
                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            verbose_logger.error(
                "litellm.proxy.hooks.prompt_injection_detection.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_logger.debug(traceback.format_exc())
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM usage on success
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
                dt = get_utc_datetime()
                current_minute = dt.strftime(
                    "%H-%M"
                )  # use the same timezone regardless of system clock

                tpm_key = f"{id}:tpm:{current_minute}"
                # ------------
                # Update usage
                # ------------
                # update cache

                ## TPM
                await self.router_cache.async_increment_cache(
                    key=tpm_key, value=total_tokens, ttl=self.routing_args.ttl
                )

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            verbose_logger.error(
                "litellm.proxy.hooks.prompt_injection_detection.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_logger.debug(traceback.format_exc())
            pass

    def _common_checks_available_deployment(
        self,
        model_group: str,
        healthy_deployments: list,
        tpm_keys: list,
        tpm_values: list,
        rpm_keys: list,
        rpm_values: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ):
        """
        Common checks for get available deployment, across sync + async implementations
        """
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
            dt = get_utc_datetime()
            current_minute = dt.strftime(
                "%H-%M"
            )  # use the same timezone regardless of system clock

            for d in healthy_deployments:
                ## if healthy deployment not yet used
                tpm_key = f"{d['model_info']['id']}:tpm:{current_minute}"
                if tpm_key not in tpm_dict or tpm_dict[tpm_key] is None:
                    tpm_dict[tpm_key] = 0

        all_deployments = tpm_dict
        potential_deployments = []  # if multiple deployments have the same low value
        for item, item_tpm in all_deployments.items():
            ## get the item from model list
            _deployment = None
            item = item.split(":")[0]
            for m in healthy_deployments:
                if item == m["model_info"]["id"]:
                    _deployment = m
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
            if item_tpm + input_tokens > _deployment_tpm:
                continue
            elif (rpm_dict is not None and item in rpm_dict) and (
                rpm_dict[item] + 1 >= _deployment_rpm
            ):
                continue
            elif item_tpm == lowest_tpm:
                potential_deployments.append(_deployment)
            elif item_tpm < lowest_tpm:
                lowest_tpm = item_tpm
                potential_deployments = [_deployment]
        print_verbose("returning picked lowest tpm/rpm deployment.")

        if len(potential_deployments) > 0:
            return random.choice(potential_deployments)
        else:
            return None

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
        # get list of potential deployments
        verbose_router_logger.debug(
            f"get_available_deployments - Usage Based. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")

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

        combined_tpm_rpm_keys = tpm_keys + rpm_keys

        combined_tpm_rpm_values = await self.router_cache.async_batch_get_cache(
            keys=combined_tpm_rpm_keys
        )  # [1, 2, None, ..]

        tpm_values = combined_tpm_rpm_values[: len(tpm_keys)]
        rpm_values = combined_tpm_rpm_values[len(tpm_keys) :]

        deployment = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_keys=tpm_keys,
            tpm_values=tpm_values,
            rpm_keys=rpm_keys,
            rpm_values=rpm_values,
            messages=messages,
            input=input,
        )

        try:
            assert deployment is not None
            return deployment
        except Exception as e:
            ### GET THE DICT OF TPM / RPM + LIMITS PER DEPLOYMENT ###
            deployment_dict = {}
            for index, _deployment in enumerate(healthy_deployments):
                if isinstance(_deployment, dict):
                    id = _deployment.get("model_info", {}).get("id")
                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_tpm = None
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("litellm_params", {}).get(
                            "tpm", None
                        )
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("model_info", {}).get(
                            "tpm", None
                        )
                    if _deployment_tpm is None:
                        _deployment_tpm = float("inf")

                    ### GET CURRENT TPM ###
                    current_tpm = tpm_values[index]

                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_rpm = None
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("litellm_params", {}).get(
                            "rpm", None
                        )
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("model_info", {}).get(
                            "rpm", None
                        )
                    if _deployment_rpm is None:
                        _deployment_rpm = float("inf")

                    ### GET CURRENT RPM ###
                    current_rpm = rpm_values[index]

                    deployment_dict[id] = {
                        "current_tpm": current_tpm,
                        "tpm_limit": _deployment_tpm,
                        "current_rpm": current_rpm,
                        "rpm_limit": _deployment_rpm,
                    }
            raise ValueError(
                f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}"
            )

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

        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
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

        deployment = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_keys=tpm_keys,
            tpm_values=tpm_values,
            rpm_keys=rpm_keys,
            rpm_values=rpm_values,
            messages=messages,
            input=input,
        )

        try:
            assert deployment is not None
            return deployment
        except Exception as e:
            ### GET THE DICT OF TPM / RPM + LIMITS PER DEPLOYMENT ###
            deployment_dict = {}
            for index, _deployment in enumerate(healthy_deployments):
                if isinstance(_deployment, dict):
                    id = _deployment.get("model_info", {}).get("id")
                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_tpm = None
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("litellm_params", {}).get(
                            "tpm", None
                        )
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("model_info", {}).get(
                            "tpm", None
                        )
                    if _deployment_tpm is None:
                        _deployment_tpm = float("inf")

                    ### GET CURRENT TPM ###
                    current_tpm = tpm_values[index]

                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_rpm = None
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("litellm_params", {}).get(
                            "rpm", None
                        )
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("model_info", {}).get(
                            "rpm", None
                        )
                    if _deployment_rpm is None:
                        _deployment_rpm = float("inf")

                    ### GET CURRENT RPM ###
                    current_rpm = rpm_values[index]

                    deployment_dict[id] = {
                        "current_tpm": current_tpm,
                        "tpm_limit": _deployment_tpm,
                        "current_rpm": current_rpm,
                        "rpm_limit": _deployment_rpm,
                    }
            raise ValueError(
                f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}"
            )

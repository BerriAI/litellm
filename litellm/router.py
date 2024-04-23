# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you ! We ❤️ you! - Krrish & Ishaan

import copy, httpx
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Any, BinaryIO
import random, threading, time, traceback, uuid
import litellm, openai, hashlib, json
from litellm.caching import RedisCache, InMemoryCache, DualCache
import datetime as datetime_og
import logging, asyncio
import inspect, concurrent
from openai import AsyncOpenAI
from collections import defaultdict
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.llms.custom_httpx.azure_dall_e_2 import (
    CustomHTTPTransport,
    AsyncCustomHTTPTransport,
)
from litellm.utils import (
    ModelResponse,
    CustomStreamWrapper,
    get_utc_datetime,
    calculate_max_parallel_requests,
    check_cache_hit,
    async_check_cache_hit,
    function_setup,
)
import copy
from litellm._logging import verbose_router_logger
import logging
from litellm.types.router import Deployment, ModelInfo, LiteLLM_Params, RouterErrors
from litellm.integrations.custom_logger import CustomLogger


class Router:
    model_names: List = []
    cache_responses: Optional[bool] = False
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour
    num_retries: int = 0
    tenacity = None
    leastbusy_logger: Optional[LeastBusyLoggingHandler] = None
    lowesttpm_logger: Optional[LowestTPMLoggingHandler] = None

    def __init__(
        self,
        model_list: Optional[list] = None,
        ## CACHING ##
        redis_url: Optional[str] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        cache_responses: Optional[bool] = False,
        cache_kwargs: dict = {},  # additional kwargs to pass to RedisCache (see caching.py)
        caching_groups: Optional[
            List[tuple]
        ] = None,  # if you want to cache across model groups
        client_ttl: int = 3600,  # ttl for cached clients - will re-initialize after this time in seconds
        ## RELIABILITY ##
        num_retries: int = 0,
        timeout: Optional[float] = None,
        default_litellm_params={},  # default params for Router.chat.completion.create
        default_max_parallel_requests: Optional[int] = None,
        set_verbose: bool = False,
        debug_level: Literal["DEBUG", "INFO"] = "INFO",
        fallbacks: List = [],
        context_window_fallbacks: List = [],
        model_group_alias: Optional[dict] = {},
        enable_pre_call_checks: bool = False,
        retry_after: int = 0,  # min time to wait before retrying a failed request
        allowed_fails: Optional[
            int
        ] = None,  # Number of times a deployment can failbefore being added to cooldown
        cooldown_time: float = 1,  # (seconds) time to cooldown a deployment after failure
        routing_strategy: Literal[
            "simple-shuffle",
            "least-busy",
            "usage-based-routing",
            "latency-based-routing",
        ] = "simple-shuffle",
        routing_strategy_args: dict = {},  # just for latency-based routing
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        """
        Initialize the Router class with the given parameters for caching, reliability, and routing strategy.

        Args:
            model_list (Optional[list]): List of models to be used. Defaults to None.
            redis_url (Optional[str]): URL of the Redis server. Defaults to None.
            redis_host (Optional[str]): Hostname of the Redis server. Defaults to None.
            redis_port (Optional[int]): Port of the Redis server. Defaults to None.
            redis_password (Optional[str]): Password of the Redis server. Defaults to None.
            cache_responses (Optional[bool]): Flag to enable caching of responses. Defaults to False.
            cache_kwargs (dict): Additional kwargs to pass to RedisCache. Defaults to {}.
            caching_groups (Optional[List[tuple]]): List of model groups for caching across model groups. Defaults to None.
            client_ttl (int): Time-to-live for cached clients in seconds. Defaults to 3600.
            num_retries (int): Number of retries for failed requests. Defaults to 0.
            timeout (Optional[float]): Timeout for requests. Defaults to None.
            default_litellm_params (dict): Default parameters for Router.chat.completion.create. Defaults to {}.
            set_verbose (bool): Flag to set verbose mode. Defaults to False.
            debug_level (Literal["DEBUG", "INFO"]): Debug level for logging. Defaults to "INFO".
            fallbacks (List): List of fallback options. Defaults to [].
            context_window_fallbacks (List): List of context window fallback options. Defaults to [].
            enable_pre_call_checks (boolean): Filter out deployments which are outside context window limits for a given prompt
            model_group_alias (Optional[dict]): Alias for model groups. Defaults to {}.
            retry_after (int): Minimum time to wait before retrying a failed request. Defaults to 0.
            allowed_fails (Optional[int]): Number of allowed fails before adding to cooldown. Defaults to None.
            cooldown_time (float): Time to cooldown a deployment after failure in seconds. Defaults to 1.
            routing_strategy (Literal["simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing"]): Routing strategy. Defaults to "simple-shuffle".
            routing_strategy_args (dict): Additional args for latency-based routing. Defaults to {}.

        Returns:
            Router: An instance of the litellm.Router class.

        Example Usage:
        ```python
        from litellm import Router
        model_list = [
        {
            "model_name": "azure-gpt-3.5-turbo", # model alias
            "litellm_params": { # params for litellm completion/embedding call
                "model": "azure/<your-deployment-name-1>",
                "api_key": <your-api-key>,
                "api_version": <your-api-version>,
                "api_base": <your-api-base>
            },
        },
        {
            "model_name": "azure-gpt-3.5-turbo", # model alias
            "litellm_params": { # params for litellm completion/embedding call
                "model": "azure/<your-deployment-name-2>",
                "api_key": <your-api-key>,
                "api_version": <your-api-version>,
                "api_base": <your-api-base>
            },
        },
        {
            "model_name": "openai-gpt-3.5-turbo", # model alias
            "litellm_params": { # params for litellm completion/embedding call
                "model": "gpt-3.5-turbo",
                "api_key": <your-api-key>,
            },
        ]

        router = Router(model_list=model_list, fallbacks=[{"azure-gpt-3.5-turbo": "openai-gpt-3.5-turbo"}])
        ```
        """
        if semaphore:
            self.semaphore = semaphore
        self.set_verbose = set_verbose
        self.debug_level = debug_level
        self.enable_pre_call_checks = enable_pre_call_checks
        if self.set_verbose == True:
            if debug_level == "INFO":
                verbose_router_logger.setLevel(logging.INFO)
            elif debug_level == "DEBUG":
                verbose_router_logger.setLevel(logging.DEBUG)

        self.deployment_names: List = (
            []
        )  # names of models under litellm_params. ex. azure/chatgpt-v-2
        self.deployment_latency_map = {}
        ### CACHING ###
        cache_type: Literal["local", "redis"] = "local"  # default to an in-memory cache
        redis_cache = None
        cache_config = {}
        self.client_ttl = client_ttl
        if redis_url is not None or (
            redis_host is not None
            and redis_port is not None
            and redis_password is not None
        ):
            cache_type = "redis"

            if redis_url is not None:
                cache_config["url"] = redis_url

            if redis_host is not None:
                cache_config["host"] = redis_host

            if redis_port is not None:
                cache_config["port"] = str(redis_port)  # type: ignore

            if redis_password is not None:
                cache_config["password"] = redis_password

            # Add additional key-value pairs from cache_kwargs
            cache_config.update(cache_kwargs)
            redis_cache = RedisCache(**cache_config)

        if cache_responses:
            if litellm.cache is None:
                # the cache can be initialized on the proxy server. We should not overwrite it
                litellm.cache = litellm.Cache(type=cache_type, **cache_config)  # type: ignore
            self.cache_responses = cache_responses
        self.cache = DualCache(
            redis_cache=redis_cache, in_memory_cache=InMemoryCache()
        )  # use a dual cache (Redis+In-Memory) for tracking cooldowns, usage, etc.

        self.default_deployment = None  # use this to track the users default deployment, when they want to use model = *
        self.default_max_parallel_requests = default_max_parallel_requests

        if model_list:
            model_list = copy.deepcopy(model_list)
            self.set_model_list(model_list)
            self.healthy_deployments: List = self.model_list
            for m in model_list:
                self.deployment_latency_map[m["litellm_params"]["model"]] = 0

        self.allowed_fails = allowed_fails or litellm.allowed_fails
        self.cooldown_time = cooldown_time or 1
        self.failed_calls = (
            InMemoryCache()
        )  # cache to track failed call per deployment, if num failed calls within 1 minute > allowed fails, then add it to cooldown
        self.num_retries = num_retries or litellm.num_retries or 0
        self.timeout = timeout or litellm.request_timeout

        self.retry_after = retry_after
        self.routing_strategy = routing_strategy
        self.fallbacks = fallbacks or litellm.fallbacks
        self.context_window_fallbacks = (
            context_window_fallbacks or litellm.context_window_fallbacks
        )
        self.total_calls: defaultdict = defaultdict(
            int
        )  # dict to store total calls made to each model
        self.fail_calls: defaultdict = defaultdict(
            int
        )  # dict to store fail_calls made to each model
        self.success_calls: defaultdict = defaultdict(
            int
        )  # dict to store success_calls  made to each model
        self.previous_models: List = (
            []
        )  # list to store failed calls (passed in as metadata to next call)
        self.model_group_alias: dict = (
            model_group_alias or {}
        )  # dict to store aliases for router, ex. {"gpt-4": "gpt-3.5-turbo"}, all requests with gpt-4 -> get routed to gpt-3.5-turbo group

        # make Router.chat.completions.create compatible for openai.chat.completions.create
        self.chat = litellm.Chat(params=default_litellm_params, router_obj=self)

        # default litellm args
        self.default_litellm_params = default_litellm_params
        self.default_litellm_params.setdefault("timeout", timeout)
        self.default_litellm_params.setdefault("max_retries", 0)
        self.default_litellm_params.setdefault("metadata", {}).update(
            {"caching_groups": caching_groups}
        )

        self.deployment_stats: dict = {}  # used for debugging load balancing
        """
        deployment_stats = {
            "122999-2828282-277:
            {
                "model": "gpt-3",
                "api_base": "http://localhost:4000",
                "num_requests": 20,
                "avg_latency": 0.001,
                "num_failures": 0,
                "num_successes": 20
            }
        }
        """
        ### ROUTING SETUP ###
        if routing_strategy == "least-busy":
            self.leastbusy_logger = LeastBusyLoggingHandler(
                router_cache=self.cache, model_list=self.model_list
            )
            ## add callback
            if isinstance(litellm.input_callback, list):
                litellm.input_callback.append(self.leastbusy_logger)  # type: ignore
            else:
                litellm.input_callback = [self.leastbusy_logger]  # type: ignore
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.leastbusy_logger)  # type: ignore
        elif routing_strategy == "usage-based-routing":
            self.lowesttpm_logger = LowestTPMLoggingHandler(
                router_cache=self.cache, model_list=self.model_list
            )
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.lowesttpm_logger)  # type: ignore
        elif routing_strategy == "usage-based-routing-v2":
            self.lowesttpm_logger_v2 = LowestTPMLoggingHandler_v2(
                router_cache=self.cache, model_list=self.model_list
            )
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.lowesttpm_logger_v2)  # type: ignore
        elif routing_strategy == "latency-based-routing":
            self.lowestlatency_logger = LowestLatencyLoggingHandler(
                router_cache=self.cache,
                model_list=self.model_list,
                routing_args=routing_strategy_args,
            )
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.lowestlatency_logger)  # type: ignore
        ## COOLDOWNS ##
        if isinstance(litellm.failure_callback, list):
            litellm.failure_callback.append(self.deployment_callback_on_failure)
        else:
            litellm.failure_callback = [self.deployment_callback_on_failure]
        verbose_router_logger.info(
            f"Intialized router with Routing strategy: {self.routing_strategy}\n\nRouting fallbacks: {self.fallbacks}\n\nRouting context window fallbacks: {self.context_window_fallbacks}\n\nRouter Redis Caching={self.cache.redis_cache}"
        )
        self.routing_strategy_args = routing_strategy_args

    def print_deployment(self, deployment: dict):
        """
        returns a copy of the deployment with the api key masked
        """
        try:
            _deployment_copy = copy.deepcopy(deployment)
            litellm_params: dict = _deployment_copy["litellm_params"]
            if "api_key" in litellm_params:
                litellm_params["api_key"] = litellm_params["api_key"][:2] + "*" * 10
            return _deployment_copy
        except Exception as e:
            verbose_router_logger.debug(
                f"Error occurred while printing deployment - {str(e)}"
            )
            raise e

    ### COMPLETION, EMBEDDING, IMG GENERATION FUNCTIONS

    def completion(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Example usage:
        response = router.completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey, how's it going?"}]
        """
        try:
            verbose_router_logger.debug(f"router.completion(model={model},..)")
            kwargs["model"] = model
            kwargs["messages"] = messages

            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            kwargs.setdefault("metadata", {}).update({"model_group": model})

            ### CHECK CACHE HIT ### - prevents unnecessary calls
            current_time = get_utc_datetime()
            kwargs["litellm_call_id"] = str(uuid.uuid4())
            logging_obj, kwargs = function_setup(
                original_function="completion",
                rules_obj=litellm.utils.Rules(),
                start_time=current_time,
                **kwargs,
            )
            response = check_cache_hit(
                function_name="completion",
                model=model,
                logging_obj=logging_obj,
                start_time=current_time,
                args=(),
                kwargs=kwargs,
            )
            if response is not None:
                return response

            ### ENTER CALLING LOGIC ###
            kwargs["original_function"] = self._completion
            response = self.function_with_fallbacks(**kwargs)
            return response
        except Exception as e:
            raise e

    def _completion(self, model: str, messages: List[Dict[str, str]], **kwargs):
        model_name = None
        try:
            # pick the one that is available (lowest TPM/RPM)
            deployment = self.get_available_deployment(
                model=model,
                messages=messages,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            kwargs.setdefault("metadata", {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "api_base": deployment.get("litellm_params", {}).get("api_base"),
                    "model_info": deployment.get("model_info", {}),
                }
            )
            data = deployment["litellm_params"].copy()
            kwargs["model_info"] = deployment.get("model_info", {})
            model_name = data["model"]
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)
            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs
            )
            # check if provided keys == client keys #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client

            ### DEPLOYMENT-SPECIFIC PRE-CALL CHECKS ### (e.g. update rpm pre-call. Raise error, if deployment over limit)
            self.routing_strategy_pre_call_checks(deployment=deployment)

            response = litellm.completion(
                **{
                    **data,
                    "messages": messages,
                    "caching": self.cache_responses,
                    "client": model_client,
                    **kwargs,
                }
            )
            verbose_router_logger.info(
                f"litellm.completion(model={model_name})\033[32m 200 OK\033[0m"
            )

            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.completion(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            raise e

    async def acompletion(self, model: str, messages: List[Dict[str, str]], **kwargs):
        try:
            kwargs["model"] = model
            kwargs["messages"] = messages

            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})

            ### CHECK CACHE HIT ### - prevents unnecessary calls
            current_time = get_utc_datetime()
            kwargs["litellm_call_id"] = str(uuid.uuid4())
            logging_obj, kwargs = function_setup(
                original_function="completion",
                rules_obj=litellm.utils.Rules(),
                start_time=current_time,
                **kwargs,
            )
            response = await async_check_cache_hit(
                function_name="acompletion",
                model=model,
                logging_obj=logging_obj,
                start_time=current_time,
                args=(),
                kwargs=kwargs,
            )
            if response is not None:
                return response

            ### MAKE CALL ###
            kwargs["original_function"] = self._acompletion
            response = await self.async_function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            raise e

    async def _acompletion(self, model: str, messages: List[Dict[str, str]], **kwargs):
        """
        - Get an available deployment
        - call it with a semaphore over the call
        - semaphore specific to it's rpm
        - in the semaphore,  make a check against it's local rpm before running
        """
        model_name = None
        try:
            verbose_router_logger.debug(
                f"Inside _acompletion()- model: {model}; kwargs: {kwargs}"
            )

            deployment = await self.async_get_available_deployment(
                model=model,
                messages=messages,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )

            # debug how often this deployment picked
            self._track_deployment_metrics(deployment=deployment)

            kwargs.setdefault("metadata", {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "model_info": deployment.get("model_info", {}),
                    "api_base": deployment.get("litellm_params", {}).get("api_base"),
                }
            )
            kwargs["model_info"] = deployment.get("model_info", {})
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs and v is not None
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs, client_type="async"
            )

            # check if provided keys == client keys #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client
            self.total_calls[model_name] += 1

            timeout = (
                data.get(
                    "timeout", None
                )  # timeout set on litellm_params for this deployment
                or self.timeout  # timeout set on router
                or kwargs.get(
                    "timeout", None
                )  # this uses default_litellm_params when nothing is set
            )

            _response = litellm.acompletion(
                **{
                    **data,
                    "messages": messages,
                    "caching": self.cache_responses,
                    "client": model_client,
                    "timeout": timeout,
                    **kwargs,
                }
            )

            rpm_semaphore = self._get_client(
                deployment=deployment,
                kwargs=kwargs,
                client_type="max_parallel_requests",
            )

            if rpm_semaphore is not None and isinstance(
                rpm_semaphore, asyncio.Semaphore
            ):
                async with rpm_semaphore:
                    """
                    - Check rpm limits before making the call
                    - If allowed, increment the rpm limit (allows global value to be updated, concurrency-safe)
                    """
                    await self.async_routing_strategy_pre_call_checks(
                        deployment=deployment
                    )
                    response = await _response
            else:
                await self.async_routing_strategy_pre_call_checks(deployment=deployment)
                response = await _response

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.acompletion(model={model_name})\033[32m 200 OK\033[0m"
            )
            # debug how often this deployment picked
            self._track_deployment_metrics(deployment=deployment, response=response)

            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.acompletion(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    def image_generation(self, prompt: str, model: str, **kwargs):
        try:
            kwargs["model"] = model
            kwargs["prompt"] = prompt
            kwargs["original_function"] = self._image_generation
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = self.function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            raise e

    def _image_generation(self, prompt: str, model: str, **kwargs):
        try:
            verbose_router_logger.debug(
                f"Inside _image_generation()- model: {model}; kwargs: {kwargs}"
            )
            deployment = self.get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "prompt"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            kwargs.setdefault("metadata", {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "model_info": deployment.get("model_info", {}),
                }
            )
            kwargs["model_info"] = deployment.get("model_info", {})
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs, client_type="async"
            )
            # check if provided keys == client keys #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client

            self.total_calls[model_name] += 1

            ### DEPLOYMENT-SPECIFIC PRE-CALL CHECKS ### (e.g. update rpm pre-call. Raise error, if deployment over limit)
            self.routing_strategy_pre_call_checks(deployment=deployment)

            response = litellm.image_generation(
                **{
                    **data,
                    "prompt": prompt,
                    "caching": self.cache_responses,
                    "client": model_client,
                    **kwargs,
                }
            )
            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.image_generation(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.image_generation(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    async def aimage_generation(self, prompt: str, model: str, **kwargs):
        try:
            kwargs["model"] = model
            kwargs["prompt"] = prompt
            kwargs["original_function"] = self._aimage_generation
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = await self.async_function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            raise e

    async def _aimage_generation(self, prompt: str, model: str, **kwargs):
        try:
            verbose_router_logger.debug(
                f"Inside _image_generation()- model: {model}; kwargs: {kwargs}"
            )
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "prompt"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            kwargs.setdefault("metadata", {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "model_info": deployment.get("model_info", {}),
                }
            )
            kwargs["model_info"] = deployment.get("model_info", {})
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs, client_type="async"
            )
            # check if provided keys == client keys #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client

            self.total_calls[model_name] += 1
            response = litellm.aimage_generation(
                **{
                    **data,
                    "prompt": prompt,
                    "caching": self.cache_responses,
                    "client": model_client,
                    **kwargs,
                }
            )

            ### CONCURRENCY-SAFE RPM CHECKS ###
            rpm_semaphore = self._get_client(
                deployment=deployment,
                kwargs=kwargs,
                client_type="max_parallel_requests",
            )

            if rpm_semaphore is not None and isinstance(
                rpm_semaphore, asyncio.Semaphore
            ):
                async with rpm_semaphore:
                    """
                    - Check rpm limits before making the call
                    - If allowed, increment the rpm limit (allows global value to be updated, concurrency-safe)
                    """
                    await self.async_routing_strategy_pre_call_checks(
                        deployment=deployment
                    )
                    response = await response
            else:
                await self.async_routing_strategy_pre_call_checks(deployment=deployment)
                response = await response

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.aimage_generation(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.aimage_generation(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    async def atranscription(self, file: BinaryIO, model: str, **kwargs):
        """
        Example Usage:

        ```
        from litellm import Router
        client = Router(model_list = [
            {
                "model_name": "whisper",
                "litellm_params": {
                    "model": "whisper-1",
                },
            },
        ])

        audio_file = open("speech.mp3", "rb")
        transcript = await client.atranscription(
        model="whisper",
        file=audio_file
        )

        ```
        """
        try:
            kwargs["model"] = model
            kwargs["file"] = file
            kwargs["original_function"] = self._atranscription
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = await self.async_function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            raise e

    async def _atranscription(self, file: BinaryIO, model: str, **kwargs):
        try:
            verbose_router_logger.debug(
                f"Inside _atranscription()- model: {model}; kwargs: {kwargs}"
            )
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "prompt"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            kwargs.setdefault("metadata", {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "model_info": deployment.get("model_info", {}),
                }
            )
            kwargs["model_info"] = deployment.get("model_info", {})
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs, client_type="async"
            )
            # check if provided keys == client keys #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client

            self.total_calls[model_name] += 1
            response = litellm.atranscription(
                **{
                    **data,
                    "file": file,
                    "caching": self.cache_responses,
                    "client": model_client,
                    **kwargs,
                }
            )

            ### CONCURRENCY-SAFE RPM CHECKS ###
            rpm_semaphore = self._get_client(
                deployment=deployment,
                kwargs=kwargs,
                client_type="max_parallel_requests",
            )

            if rpm_semaphore is not None and isinstance(
                rpm_semaphore, asyncio.Semaphore
            ):
                async with rpm_semaphore:
                    """
                    - Check rpm limits before making the call
                    - If allowed, increment the rpm limit (allows global value to be updated, concurrency-safe)
                    """
                    await self.async_routing_strategy_pre_call_checks(
                        deployment=deployment
                    )
                    response = await response
            else:
                await self.async_routing_strategy_pre_call_checks(deployment=deployment)
                response = await response

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.atranscription(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.atranscription(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    async def amoderation(self, model: str, input: str, **kwargs):
        try:
            kwargs["model"] = model
            kwargs["input"] = input
            kwargs["original_function"] = self._amoderation
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})

            response = await self.async_function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            raise e

    async def _amoderation(self, model: str, input: str, **kwargs):
        model_name = None
        try:
            verbose_router_logger.debug(
                f"Inside _moderation()- model: {model}; kwargs: {kwargs}"
            )
            deployment = await self.async_get_available_deployment(
                model=model,
                input=input,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            kwargs.setdefault("metadata", {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "model_info": deployment.get("model_info", {}),
                }
            )
            kwargs["model_info"] = deployment.get("model_info", {})
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs and v is not None
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs, client_type="async"
            )
            # check if provided keys == client keys #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client
            self.total_calls[model_name] += 1

            timeout = (
                data.get(
                    "timeout", None
                )  # timeout set on litellm_params for this deployment
                or self.timeout  # timeout set on router
                or kwargs.get(
                    "timeout", None
                )  # this uses default_litellm_params when nothing is set
            )

            response = await litellm.amoderation(
                **{
                    **data,
                    "input": input,
                    "caching": self.cache_responses,
                    "client": model_client,
                    "timeout": timeout,
                    **kwargs,
                }
            )

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.amoderation(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.amoderation(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    def text_completion(
        self,
        model: str,
        prompt: str,
        is_retry: Optional[bool] = False,
        is_fallback: Optional[bool] = False,
        is_async: Optional[bool] = False,
        **kwargs,
    ):
        try:
            kwargs["model"] = model
            kwargs["prompt"] = prompt
            kwargs["original_function"] = self._acompletion
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})

            messages = [{"role": "user", "content": prompt}]
            # pick the one that is available (lowest TPM/RPM)
            deployment = self.get_available_deployment(
                model=model,
                messages=messages,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )

            data = deployment["litellm_params"].copy()
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            # call via litellm.completion()
            return litellm.text_completion(**{**data, "prompt": prompt, "caching": self.cache_responses, **kwargs})  # type: ignore
        except Exception as e:
            if self.num_retries > 0:
                kwargs["model"] = model
                kwargs["messages"] = messages
                kwargs["original_function"] = self.completion
                return self.function_with_retries(**kwargs)
            else:
                raise e

    async def atext_completion(
        self,
        model: str,
        prompt: str,
        is_retry: Optional[bool] = False,
        is_fallback: Optional[bool] = False,
        is_async: Optional[bool] = False,
        **kwargs,
    ):
        try:
            kwargs["model"] = model
            kwargs["prompt"] = prompt
            kwargs["original_function"] = self._atext_completion
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = await self.async_function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            raise e

    async def _atext_completion(self, model: str, prompt: str, **kwargs):
        try:
            verbose_router_logger.debug(
                f"Inside _atext_completion()- model: {model}; kwargs: {kwargs}"
            )
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            kwargs.setdefault("metadata", {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "model_info": deployment.get("model_info", {}),
                    "api_base": deployment.get("litellm_params", {}).get("api_base"),
                }
            )
            kwargs["model_info"] = deployment.get("model_info", {})
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs, client_type="async"
            )
            # check if provided keys == client keys #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client
            self.total_calls[model_name] += 1

            response = litellm.atext_completion(
                **{
                    **data,
                    "prompt": prompt,
                    "caching": self.cache_responses,
                    "client": model_client,
                    "timeout": self.timeout,
                    **kwargs,
                }
            )

            rpm_semaphore = self._get_client(
                deployment=deployment,
                kwargs=kwargs,
                client_type="max_parallel_requests",
            )

            if rpm_semaphore is not None and isinstance(
                rpm_semaphore, asyncio.Semaphore
            ):
                async with rpm_semaphore:
                    """
                    - Check rpm limits before making the call
                    - If allowed, increment the rpm limit (allows global value to be updated, concurrency-safe)
                    """
                    await self.async_routing_strategy_pre_call_checks(
                        deployment=deployment
                    )
                    response = await response
            else:
                await self.async_routing_strategy_pre_call_checks(deployment=deployment)
                response = await response

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.atext_completion(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.atext_completion(model={model})\033[31m Exception {str(e)}\033[0m"
            )
            if model is not None:
                self.fail_calls[model] += 1
            raise e

    def embedding(
        self,
        model: str,
        input: Union[str, List],
        is_async: Optional[bool] = False,
        **kwargs,
    ) -> Union[List[float], None]:
        try:
            kwargs["model"] = model
            kwargs["input"] = input
            kwargs["original_function"] = self._embedding
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = self.function_with_fallbacks(**kwargs)
            return response
        except Exception as e:
            raise e

    def _embedding(self, input: Union[str, List], model: str, **kwargs):
        try:
            verbose_router_logger.debug(
                f"Inside embedding()- model: {model}; kwargs: {kwargs}"
            )
            deployment = self.get_available_deployment(
                model=model,
                input=input,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            kwargs.setdefault("metadata", {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "model_info": deployment.get("model_info", {}),
                }
            )
            kwargs["model_info"] = deployment.get("model_info", {})
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs, client_type="sync"
            )
            # check if provided keys == client keys #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client

            self.total_calls[model_name] += 1

            ### DEPLOYMENT-SPECIFIC PRE-CALL CHECKS ### (e.g. update rpm pre-call. Raise error, if deployment over limit)
            self.routing_strategy_pre_call_checks(deployment=deployment)

            response = litellm.embedding(
                **{
                    **data,
                    "input": input,
                    "caching": self.cache_responses,
                    "client": model_client,
                    **kwargs,
                }
            )
            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.embedding(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.embedding(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    async def aembedding(
        self,
        model: str,
        input: Union[str, List],
        is_async: Optional[bool] = True,
        **kwargs,
    ) -> Union[List[float], None]:
        try:
            kwargs["model"] = model
            kwargs["input"] = input
            kwargs["original_function"] = self._aembedding
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = await self.async_function_with_fallbacks(**kwargs)
            return response
        except Exception as e:
            raise e

    async def _aembedding(self, input: Union[str, List], model: str, **kwargs):
        model_name = None
        try:
            verbose_router_logger.debug(
                f"Inside _aembedding()- model: {model}; kwargs: {kwargs}"
            )
            deployment = await self.async_get_available_deployment(
                model=model,
                input=input,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            kwargs.setdefault("metadata", {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "model_info": deployment.get("model_info", {}),
                    "api_base": deployment.get("litellm_params", {}).get("api_base"),
                }
            )
            kwargs["model_info"] = deployment.get("model_info", {})
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs
                ):  # prioritize model-specific params > default router params
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs, client_type="async"
            )
            # check if provided keys == client keys #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client

            self.total_calls[model_name] += 1
            response = litellm.aembedding(
                **{
                    **data,
                    "input": input,
                    "caching": self.cache_responses,
                    "client": model_client,
                    **kwargs,
                }
            )

            ### CONCURRENCY-SAFE RPM CHECKS ###
            rpm_semaphore = self._get_client(
                deployment=deployment,
                kwargs=kwargs,
                client_type="max_parallel_requests",
            )

            if rpm_semaphore is not None and isinstance(
                rpm_semaphore, asyncio.Semaphore
            ):
                async with rpm_semaphore:
                    """
                    - Check rpm limits before making the call
                    - If allowed, increment the rpm limit (allows global value to be updated, concurrency-safe)
                    """
                    await self.async_routing_strategy_pre_call_checks(
                        deployment=deployment
                    )
                    response = await response
            else:
                await self.async_routing_strategy_pre_call_checks(deployment=deployment)
                response = await response

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.aembedding(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.aembedding(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    async def async_function_with_fallbacks(self, *args, **kwargs):
        """
        Try calling the function_with_retries
        If it fails after num_retries, fall back to another model group
        """
        model_group = kwargs.get("model")
        fallbacks = kwargs.get("fallbacks", self.fallbacks)
        context_window_fallbacks = kwargs.get(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        try:
            response = await self.async_function_with_retries(*args, **kwargs)
            verbose_router_logger.debug(f"Async Response: {response}")
            return response
        except Exception as e:
            verbose_router_logger.debug(f"Traceback{traceback.format_exc()}")
            original_exception = e
            fallback_model_group = None
            try:
                verbose_router_logger.debug(f"Trying to fallback b/w models")
                if (
                    hasattr(e, "status_code")
                    and e.status_code == 400
                    and not isinstance(e, litellm.ContextWindowExceededError)
                ):  # don't retry a malformed request
                    raise e
                if (
                    isinstance(e, litellm.ContextWindowExceededError)
                    and context_window_fallbacks is not None
                ):
                    fallback_model_group = None
                    for (
                        item
                    ) in context_window_fallbacks:  # [{"gpt-3.5-turbo": ["gpt-4"]}]
                        if list(item.keys())[0] == model_group:
                            fallback_model_group = item[model_group]
                            break

                    if fallback_model_group is None:
                        raise original_exception

                    for mg in fallback_model_group:
                        """
                        Iterate through the model groups and try calling that deployment
                        """
                        try:
                            kwargs["model"] = mg
                            kwargs.setdefault("metadata", {}).update(
                                {"model_group": mg}
                            )  # update model_group used, if fallbacks are done
                            response = await self.async_function_with_retries(
                                *args, **kwargs
                            )
                            return response
                        except Exception as e:
                            pass
                elif fallbacks is not None:
                    verbose_router_logger.debug(f"inside model fallbacks: {fallbacks}")
                    for item in fallbacks:
                        if list(item.keys())[0] == model_group:
                            fallback_model_group = item[model_group]
                            break
                    if fallback_model_group is None:
                        verbose_router_logger.info(
                            f"No fallback model group found for original model_group={model_group}. Fallbacks={fallbacks}"
                        )
                        raise original_exception
                    for mg in fallback_model_group:
                        """
                        Iterate through the model groups and try calling that deployment
                        """
                        try:
                            ## LOGGING
                            kwargs = self.log_retry(kwargs=kwargs, e=original_exception)
                            verbose_router_logger.info(
                                f"Falling back to model_group = {mg}"
                            )
                            kwargs["model"] = mg
                            kwargs.setdefault("metadata", {}).update(
                                {"model_group": mg}
                            )  # update model_group used, if fallbacks are done
                            response = await self.async_function_with_fallbacks(
                                *args, **kwargs
                            )
                            return response
                        except Exception as e:
                            raise e
            except Exception as e:
                verbose_router_logger.debug(f"An exception occurred - {str(e)}")
                traceback.print_exc()
            raise original_exception

    async def async_function_with_retries(self, *args, **kwargs):
        verbose_router_logger.debug(
            f"Inside async function with retries: args - {args}; kwargs - {kwargs}"
        )
        original_function = kwargs.pop("original_function")
        fallbacks = kwargs.pop("fallbacks", self.fallbacks)
        context_window_fallbacks = kwargs.pop(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        verbose_router_logger.debug(
            f"async function w/ retries: original_function - {original_function}"
        )
        num_retries = kwargs.pop("num_retries")
        try:
            # if the function call is successful, no exception will be raised and we'll break out of the loop
            response = await original_function(*args, **kwargs)
            return response
        except Exception as e:
            original_exception = e
            ### CHECK IF RATE LIMIT / CONTEXT WINDOW ERROR w/ fallbacks available / Bad Request Error
            if (
                isinstance(original_exception, litellm.ContextWindowExceededError)
                and context_window_fallbacks is not None
            ) or (
                isinstance(original_exception, openai.RateLimitError)
                and fallbacks is not None
            ):
                raise original_exception
            ### RETRY
            #### check if it should retry + back-off if required
            if "No models available" in str(e):
                timeout = litellm._calculate_retry_after(
                    remaining_retries=num_retries,
                    max_retries=num_retries,
                    min_timeout=self.retry_after,
                )
                await asyncio.sleep(timeout)
            elif RouterErrors.user_defined_ratelimit_error.value in str(e):
                raise e  # don't wait to retry if deployment hits user-defined rate-limit
            elif hasattr(original_exception, "status_code") and litellm._should_retry(
                status_code=original_exception.status_code
            ):
                if hasattr(original_exception, "response") and hasattr(
                    original_exception.response, "headers"
                ):
                    timeout = litellm._calculate_retry_after(
                        remaining_retries=num_retries,
                        max_retries=num_retries,
                        response_headers=original_exception.response.headers,
                        min_timeout=self.retry_after,
                    )
                else:
                    timeout = litellm._calculate_retry_after(
                        remaining_retries=num_retries,
                        max_retries=num_retries,
                        min_timeout=self.retry_after,
                    )
                await asyncio.sleep(timeout)
            else:
                raise original_exception

            ## LOGGING
            if num_retries > 0:
                kwargs = self.log_retry(kwargs=kwargs, e=original_exception)

            for current_attempt in range(num_retries):
                verbose_router_logger.debug(
                    f"retrying request. Current attempt - {current_attempt}; num retries: {num_retries}"
                )
                try:
                    # if the function call is successful, no exception will be raised and we'll break out of the loop
                    response = await original_function(*args, **kwargs)
                    if inspect.iscoroutinefunction(
                        response
                    ):  # async errors are often returned as coroutines
                        response = await response
                    return response

                except Exception as e:
                    ## LOGGING
                    kwargs = self.log_retry(kwargs=kwargs, e=e)
                    remaining_retries = num_retries - current_attempt
                    if "No models available" in str(e):
                        timeout = litellm._calculate_retry_after(
                            remaining_retries=remaining_retries,
                            max_retries=num_retries,
                            min_timeout=self.retry_after,
                        )
                        await asyncio.sleep(timeout)
                    elif (
                        hasattr(e, "status_code")
                        and hasattr(e, "response")
                        and litellm._should_retry(status_code=e.status_code)
                    ):
                        if hasattr(e.response, "headers"):
                            timeout = litellm._calculate_retry_after(
                                remaining_retries=remaining_retries,
                                max_retries=num_retries,
                                response_headers=e.response.headers,
                                min_timeout=self.retry_after,
                            )
                        else:
                            timeout = litellm._calculate_retry_after(
                                remaining_retries=remaining_retries,
                                max_retries=num_retries,
                                min_timeout=self.retry_after,
                            )
                        await asyncio.sleep(timeout)
                    else:
                        raise e
            raise original_exception

    def function_with_fallbacks(self, *args, **kwargs):
        """
        Try calling the function_with_retries
        If it fails after num_retries, fall back to another model group
        """
        model_group = kwargs.get("model")
        fallbacks = kwargs.get("fallbacks", self.fallbacks)
        context_window_fallbacks = kwargs.get(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        try:
            response = self.function_with_retries(*args, **kwargs)
            return response
        except Exception as e:
            original_exception = e
            verbose_router_logger.debug(f"An exception occurs {original_exception}")
            try:
                if (
                    hasattr(e, "status_code")
                    and e.status_code == 400
                    and not isinstance(e, litellm.ContextWindowExceededError)
                ):  # don't retry a malformed request
                    raise e

                verbose_router_logger.debug(
                    f"Trying to fallback b/w models. Initial model group: {model_group}"
                )
                if (
                    isinstance(e, litellm.ContextWindowExceededError)
                    and context_window_fallbacks is not None
                ):
                    fallback_model_group = None

                    for (
                        item
                    ) in context_window_fallbacks:  # [{"gpt-3.5-turbo": ["gpt-4"]}]
                        if list(item.keys())[0] == model_group:
                            fallback_model_group = item[model_group]
                            break

                    if fallback_model_group is None:
                        raise original_exception

                    for mg in fallback_model_group:
                        """
                        Iterate through the model groups and try calling that deployment
                        """
                        try:
                            ## LOGGING
                            kwargs = self.log_retry(kwargs=kwargs, e=original_exception)
                            kwargs["model"] = mg
                            kwargs.setdefault("metadata", {}).update(
                                {"model_group": mg}
                            )  # update model_group used, if fallbacks are done
                            response = self.function_with_fallbacks(*args, **kwargs)
                            return response
                        except Exception as e:
                            pass
                elif fallbacks is not None:
                    verbose_router_logger.debug(f"inside model fallbacks: {fallbacks}")
                    fallback_model_group = None
                    for item in fallbacks:
                        if list(item.keys())[0] == model_group:
                            fallback_model_group = item[model_group]
                            break

                    if fallback_model_group is None:
                        raise original_exception

                    for mg in fallback_model_group:
                        """
                        Iterate through the model groups and try calling that deployment
                        """
                        try:
                            ## LOGGING
                            kwargs = self.log_retry(kwargs=kwargs, e=original_exception)
                            kwargs["model"] = mg
                            kwargs.setdefault("metadata", {}).update(
                                {"model_group": mg}
                            )  # update model_group used, if fallbacks are done
                            response = self.function_with_fallbacks(*args, **kwargs)
                            return response
                        except Exception as e:
                            raise e
            except Exception as e:
                raise e
            raise original_exception

    def function_with_retries(self, *args, **kwargs):
        """
        Try calling the model 3 times. Shuffle between available deployments.
        """
        verbose_router_logger.debug(
            f"Inside function with retries: args - {args}; kwargs - {kwargs}"
        )
        original_function = kwargs.pop("original_function")
        num_retries = kwargs.pop("num_retries")
        fallbacks = kwargs.pop("fallbacks", self.fallbacks)
        context_window_fallbacks = kwargs.pop(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        try:
            # if the function call is successful, no exception will be raised and we'll break out of the loop
            response = original_function(*args, **kwargs)
            return response
        except Exception as e:
            original_exception = e
            verbose_router_logger.debug(
                f"num retries in function with retries: {num_retries}"
            )
            ### CHECK IF RATE LIMIT / CONTEXT WINDOW ERROR
            if (
                isinstance(original_exception, litellm.ContextWindowExceededError)
                and context_window_fallbacks is not None
            ) or (
                isinstance(original_exception, openai.RateLimitError)
                and fallbacks is not None
            ):
                raise original_exception
            ## LOGGING
            if num_retries > 0:
                kwargs = self.log_retry(kwargs=kwargs, e=original_exception)
            ### RETRY
            for current_attempt in range(num_retries):
                verbose_router_logger.debug(
                    f"retrying request. Current attempt - {current_attempt}; retries left: {num_retries}"
                )
                try:
                    # if the function call is successful, no exception will be raised and we'll break out of the loop
                    response = original_function(*args, **kwargs)
                    return response

                except Exception as e:
                    ## LOGGING
                    kwargs = self.log_retry(kwargs=kwargs, e=e)
                    remaining_retries = num_retries - current_attempt
                    if "No models available" in str(e):
                        timeout = litellm._calculate_retry_after(
                            remaining_retries=remaining_retries,
                            max_retries=num_retries,
                            min_timeout=self.retry_after,
                        )
                        time.sleep(timeout)
                    elif (
                        hasattr(e, "status_code")
                        and hasattr(e, "response")
                        and litellm._should_retry(status_code=e.status_code)
                    ):
                        if hasattr(e.response, "headers"):
                            timeout = litellm._calculate_retry_after(
                                remaining_retries=remaining_retries,
                                max_retries=num_retries,
                                response_headers=e.response.headers,
                                min_timeout=self.retry_after,
                            )
                        else:
                            timeout = litellm._calculate_retry_after(
                                remaining_retries=remaining_retries,
                                max_retries=num_retries,
                                min_timeout=self.retry_after,
                            )
                        time.sleep(timeout)
                    else:
                        raise e
            raise original_exception

    ### HELPER FUNCTIONS

    def deployment_callback_on_failure(
        self,
        kwargs,  # kwargs to completion
        completion_response,  # response from completion
        start_time,
        end_time,  # start/end time
    ):
        try:
            exception = kwargs.get("exception", None)
            exception_type = type(exception)
            exception_status = getattr(exception, "status_code", "")
            exception_cause = getattr(exception, "__cause__", "")
            exception_message = getattr(exception, "message", "")
            exception_str = (
                str(exception_type)
                + "Status: "
                + str(exception_status)
                + "Message: "
                + str(exception_cause)
                + str(exception_message)
                + "Full exception"
                + str(exception)
            )
            model_name = kwargs.get("model", None)  # i.e. gpt35turbo
            custom_llm_provider = kwargs.get("litellm_params", {}).get(
                "custom_llm_provider", None
            )  # i.e. azure
            metadata = kwargs.get("litellm_params", {}).get("metadata", None)
            _model_info = kwargs.get("litellm_params", {}).get("model_info", {})
            if isinstance(_model_info, dict):
                deployment_id = _model_info.get("id", None)
                self._set_cooldown_deployments(
                    deployment_id
                )  # setting deployment_id in cooldown deployments
            if custom_llm_provider:
                model_name = f"{custom_llm_provider}/{model_name}"

        except Exception as e:
            raise e

    def log_retry(self, kwargs: dict, e: Exception) -> dict:
        """
        When a retry or fallback happens, log the details of the just failed model call - similar to Sentry breadcrumbing
        """
        try:
            # Log failed model as the previous model
            previous_model = {
                "exception_type": type(e).__name__,
                "exception_string": str(e),
            }
            for (
                k,
                v,
            ) in (
                kwargs.items()
            ):  # log everything in kwargs except the old previous_models value - prevent nesting
                if k not in ["metadata", "messages", "original_function"]:
                    previous_model[k] = v
                elif k == "metadata" and isinstance(v, dict):
                    previous_model["metadata"] = {}  # type: ignore
                    for metadata_k, metadata_v in kwargs["metadata"].items():
                        if metadata_k != "previous_models":
                            previous_model[k][metadata_k] = metadata_v  # type: ignore

            # check current size of self.previous_models, if it's larger than 3, remove the first element
            if len(self.previous_models) > 3:
                self.previous_models.pop(0)

            self.previous_models.append(previous_model)
            kwargs["metadata"]["previous_models"] = self.previous_models
            return kwargs
        except Exception as e:
            raise e

    def _update_usage(self, deployment_id: str):
        """
        Update deployment rpm for that minute
        """
        rpm_key = deployment_id

        request_count = self.cache.get_cache(key=rpm_key, local_only=True)
        if request_count is None:
            request_count = 1
            self.cache.set_cache(
                key=rpm_key, value=request_count, local_only=True, ttl=60
            )  # only store for 60s
        else:
            request_count += 1
            self.cache.set_cache(
                key=rpm_key, value=request_count, local_only=True
            )  # don't change existing ttl

    def _set_cooldown_deployments(self, deployment: Optional[str] = None):
        """
        Add a model to the list of models being cooled down for that minute, if it exceeds the allowed fails / minute
        """
        if deployment is None:
            return

        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        # get current fails for deployment
        # update the number of failed calls
        # if it's > allowed fails
        # cooldown deployment
        current_fails = self.failed_calls.get_cache(key=deployment) or 0
        updated_fails = current_fails + 1
        verbose_router_logger.debug(
            f"Attempting to add {deployment} to cooldown list. updated_fails: {updated_fails}; self.allowed_fails: {self.allowed_fails}"
        )
        cooldown_time = self.cooldown_time or 1
        if updated_fails > self.allowed_fails:
            # get the current cooldown list for that minute
            cooldown_key = f"{current_minute}:cooldown_models"  # group cooldown models by minute to reduce number of redis calls
            cached_value = self.cache.get_cache(key=cooldown_key)

            verbose_router_logger.debug(f"adding {deployment} to cooldown models")
            # update value
            try:
                if deployment in cached_value:
                    pass
                else:
                    cached_value = cached_value + [deployment]
                    # save updated value
                    self.cache.set_cache(
                        value=cached_value, key=cooldown_key, ttl=cooldown_time
                    )
            except:
                cached_value = [deployment]
                # save updated value
                self.cache.set_cache(
                    value=cached_value, key=cooldown_key, ttl=cooldown_time
                )
        else:
            self.failed_calls.set_cache(
                key=deployment, value=updated_fails, ttl=cooldown_time
            )

    async def _async_get_cooldown_deployments(self):
        """
        Async implementation of '_get_cooldown_deployments'
        """
        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        # get the current cooldown list for that minute
        cooldown_key = f"{current_minute}:cooldown_models"

        # ----------------------
        # Return cooldown models
        # ----------------------
        cooldown_models = await self.cache.async_get_cache(key=cooldown_key) or []

        verbose_router_logger.debug(f"retrieve cooldown models: {cooldown_models}")
        return cooldown_models

    def _get_cooldown_deployments(self):
        """
        Get the list of models being cooled down for this minute
        """
        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        # get the current cooldown list for that minute
        cooldown_key = f"{current_minute}:cooldown_models"

        # ----------------------
        # Return cooldown models
        # ----------------------
        cooldown_models = self.cache.get_cache(key=cooldown_key) or []

        verbose_router_logger.debug(f"retrieve cooldown models: {cooldown_models}")
        return cooldown_models

    def routing_strategy_pre_call_checks(self, deployment: dict):
        """
        Mimics 'async_routing_strategy_pre_call_checks'

        Ensures consistent update rpm implementation for 'usage-based-routing-v2'

        Returns:
        - None

        Raises:
        - Rate Limit Exception - If the deployment is over it's tpm/rpm limits
        """
        for _callback in litellm.callbacks:
            if isinstance(_callback, CustomLogger):
                response = _callback.pre_call_check(deployment)

    async def async_routing_strategy_pre_call_checks(self, deployment: dict):
        """
        For usage-based-routing-v2, enables running rpm checks before the call is made, inside the semaphore.

        -> makes the calls concurrency-safe, when rpm limits are set for a deployment

        Returns:
        - None

        Raises:
        - Rate Limit Exception - If the deployment is over it's tpm/rpm limits
        """
        for _callback in litellm.callbacks:
            if isinstance(_callback, CustomLogger):
                response = await _callback.async_pre_call_check(deployment)

    def set_client(self, model: dict):
        """
        - Initializes Azure/OpenAI clients. Stores them in cache, b/c of this - https://github.com/BerriAI/litellm/issues/1278
        - Initializes Semaphore for client w/ rpm. Stores them in cache. b/c of this - https://github.com/BerriAI/litellm/issues/2994
        """
        client_ttl = self.client_ttl
        litellm_params = model.get("litellm_params", {})
        model_name = litellm_params.get("model")
        model_id = model["model_info"]["id"]
        # ### IF RPM SET - initialize a semaphore ###
        rpm = litellm_params.get("rpm", None)
        tpm = litellm_params.get("tpm", None)
        max_parallel_requests = litellm_params.get("max_parallel_requests", None)
        calculated_max_parallel_requests = calculate_max_parallel_requests(
            rpm=rpm,
            max_parallel_requests=max_parallel_requests,
            tpm=tpm,
            default_max_parallel_requests=self.default_max_parallel_requests,
        )
        if calculated_max_parallel_requests:
            semaphore = asyncio.Semaphore(calculated_max_parallel_requests)
            cache_key = f"{model_id}_max_parallel_requests_client"
            self.cache.set_cache(
                key=cache_key,
                value=semaphore,
                local_only=True,
            )

        ####  for OpenAI / Azure we need to initalize the Client for High Traffic ########
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        custom_llm_provider = custom_llm_provider or model_name.split("/", 1)[0] or ""
        default_api_base = None
        default_api_key = None
        if custom_llm_provider in litellm.openai_compatible_providers:
            _, custom_llm_provider, api_key, api_base = litellm.get_llm_provider(
                model=model_name
            )
            default_api_base = api_base
            default_api_key = api_key
        if (
            model_name in litellm.open_ai_chat_completion_models
            or custom_llm_provider in litellm.openai_compatible_providers
            or custom_llm_provider == "azure"
            or custom_llm_provider == "azure_text"
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "openai"
            or custom_llm_provider == "text-completion-openai"
            or "ft:gpt-3.5-turbo" in model_name
            or model_name in litellm.open_ai_embedding_models
        ):
            if custom_llm_provider == "azure":
                if litellm.utils._is_non_openai_azure_model(model_name):
                    custom_llm_provider = "openai"
                    # remove azure prefx from model_name
                    model_name = model_name.replace("azure/", "")
            # glorified / complicated reading of configs
            # user can pass vars directly or they can pas os.environ/AZURE_API_KEY, in which case we will read the env
            # we do this here because we init clients for Azure, OpenAI and we need to set the right key
            api_key = litellm_params.get("api_key") or default_api_key
            if api_key and api_key.startswith("os.environ/"):
                api_key_env_name = api_key.replace("os.environ/", "")
                api_key = litellm.get_secret(api_key_env_name)
                litellm_params["api_key"] = api_key

            api_base = litellm_params.get("api_base")
            base_url = litellm_params.get("base_url")
            api_base = (
                api_base or base_url or default_api_base
            )  # allow users to pass in `api_base` or `base_url` for azure
            if api_base and api_base.startswith("os.environ/"):
                api_base_env_name = api_base.replace("os.environ/", "")
                api_base = litellm.get_secret(api_base_env_name)
                litellm_params["api_base"] = api_base

            api_version = litellm_params.get("api_version")
            if api_version and api_version.startswith("os.environ/"):
                api_version_env_name = api_version.replace("os.environ/", "")
                api_version = litellm.get_secret(api_version_env_name)
                litellm_params["api_version"] = api_version

            timeout = litellm_params.pop("timeout", None)
            if isinstance(timeout, str) and timeout.startswith("os.environ/"):
                timeout_env_name = timeout.replace("os.environ/", "")
                timeout = litellm.get_secret(timeout_env_name)
                litellm_params["timeout"] = timeout

            stream_timeout = litellm_params.pop(
                "stream_timeout", timeout
            )  # if no stream_timeout is set, default to timeout
            if isinstance(stream_timeout, str) and stream_timeout.startswith(
                "os.environ/"
            ):
                stream_timeout_env_name = stream_timeout.replace("os.environ/", "")
                stream_timeout = litellm.get_secret(stream_timeout_env_name)
                litellm_params["stream_timeout"] = stream_timeout

            max_retries = litellm_params.pop("max_retries", 2)
            if isinstance(max_retries, str) and max_retries.startswith("os.environ/"):
                max_retries_env_name = max_retries.replace("os.environ/", "")
                max_retries = litellm.get_secret(max_retries_env_name)
                litellm_params["max_retries"] = max_retries

            # proxy support
            import os
            import httpx

            # Check if the HTTP_PROXY and HTTPS_PROXY environment variables are set and use them accordingly.
            http_proxy = os.getenv("HTTP_PROXY", None)
            https_proxy = os.getenv("HTTPS_PROXY", None)
            no_proxy = os.getenv("NO_PROXY", None)

            # Create the proxies dictionary only if the environment variables are set.
            sync_proxy_mounts = None
            async_proxy_mounts = None
            if http_proxy is not None and https_proxy is not None:
                sync_proxy_mounts = {
                    "http://": httpx.HTTPTransport(proxy=httpx.Proxy(url=http_proxy)),
                    "https://": httpx.HTTPTransport(proxy=httpx.Proxy(url=https_proxy)),
                }
                async_proxy_mounts = {
                    "http://": httpx.AsyncHTTPTransport(
                        proxy=httpx.Proxy(url=http_proxy)
                    ),
                    "https://": httpx.AsyncHTTPTransport(
                        proxy=httpx.Proxy(url=https_proxy)
                    ),
                }

                # assume no_proxy is a list of comma separated urls
                if no_proxy is not None and isinstance(no_proxy, str):
                    no_proxy_urls = no_proxy.split(",")

                    for url in no_proxy_urls:  # set no-proxy support for specific urls
                        sync_proxy_mounts[url] = None  # type: ignore
                        async_proxy_mounts[url] = None  # type: ignore

            organization = litellm_params.get("organization", None)
            if isinstance(organization, str) and organization.startswith("os.environ/"):
                organization_env_name = organization.replace("os.environ/", "")
                organization = litellm.get_secret(organization_env_name)
                litellm_params["organization"] = organization

            if "azure" in model_name:
                if api_base is None:
                    raise ValueError(
                        f"api_base is required for Azure OpenAI. Set it on your config. Model - {model}"
                    )
                if api_version is None:
                    api_version = "2023-07-01-preview"
                if "gateway.ai.cloudflare.com" in api_base:
                    if not api_base.endswith("/"):
                        api_base += "/"
                    azure_model = model_name.replace("azure/", "")
                    api_base += f"{azure_model}"
                    cache_key = f"{model_id}_async_client"
                    _client = openai.AsyncAzureOpenAI(
                        api_key=api_key,
                        base_url=api_base,
                        api_version=api_version,
                        timeout=timeout,
                        max_retries=max_retries,
                        http_client=httpx.AsyncClient(
                            transport=AsyncCustomHTTPTransport(),
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            mounts=async_proxy_mounts,
                        ),  # type: ignore
                    )
                    self.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr

                    cache_key = f"{model_id}_client"
                    _client = openai.AzureOpenAI(  # type: ignore
                        api_key=api_key,
                        base_url=api_base,
                        api_version=api_version,
                        timeout=timeout,
                        max_retries=max_retries,
                        http_client=httpx.Client(
                            transport=CustomHTTPTransport(),
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            mounts=sync_proxy_mounts,
                        ),  # type: ignore
                    )
                    self.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr
                    # streaming clients can have diff timeouts
                    cache_key = f"{model_id}_stream_async_client"
                    _client = openai.AsyncAzureOpenAI(  # type: ignore
                        api_key=api_key,
                        base_url=api_base,
                        api_version=api_version,
                        timeout=stream_timeout,
                        max_retries=max_retries,
                        http_client=httpx.AsyncClient(
                            transport=AsyncCustomHTTPTransport(),
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            mounts=async_proxy_mounts,
                        ),  # type: ignore
                    )
                    self.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr

                    cache_key = f"{model_id}_stream_client"
                    _client = openai.AzureOpenAI(  # type: ignore
                        api_key=api_key,
                        base_url=api_base,
                        api_version=api_version,
                        timeout=stream_timeout,
                        max_retries=max_retries,
                        http_client=httpx.Client(
                            transport=CustomHTTPTransport(),
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            mounts=sync_proxy_mounts,
                        ),  # type: ignore
                    )
                    self.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr
                else:
                    _api_key = api_key
                    if _api_key is not None and isinstance(_api_key, str):
                        # only show first 5 chars of api_key
                        _api_key = _api_key[:8] + "*" * 15
                    verbose_router_logger.debug(
                        f"Initializing Azure OpenAI Client for {model_name}, Api Base: {str(api_base)}, Api Key:{_api_key}"
                    )
                    azure_client_params = {
                        "api_key": api_key,
                        "azure_endpoint": api_base,
                        "api_version": api_version,
                    }
                    from litellm.llms.azure import select_azure_base_url_or_endpoint

                    # this decides if we should set azure_endpoint or base_url on Azure OpenAI Client
                    # required to support GPT-4 vision enhancements, since base_url needs to be set on Azure OpenAI Client
                    azure_client_params = select_azure_base_url_or_endpoint(
                        azure_client_params
                    )

                    cache_key = f"{model_id}_async_client"
                    _client = openai.AsyncAzureOpenAI(  # type: ignore
                        **azure_client_params,
                        timeout=timeout,
                        max_retries=max_retries,
                        http_client=httpx.AsyncClient(
                            transport=AsyncCustomHTTPTransport(),
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            mounts=async_proxy_mounts,
                        ),  # type: ignore
                    )
                    self.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr

                    cache_key = f"{model_id}_client"
                    _client = openai.AzureOpenAI(  # type: ignore
                        **azure_client_params,
                        timeout=timeout,
                        max_retries=max_retries,
                        http_client=httpx.Client(
                            transport=CustomHTTPTransport(),
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            mounts=sync_proxy_mounts,
                        ),  # type: ignore
                    )
                    self.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr

                    # streaming clients should have diff timeouts
                    cache_key = f"{model_id}_stream_async_client"
                    _client = openai.AsyncAzureOpenAI(  # type: ignore
                        **azure_client_params,
                        timeout=stream_timeout,
                        max_retries=max_retries,
                        http_client=httpx.AsyncClient(
                            transport=AsyncCustomHTTPTransport(),
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            mounts=async_proxy_mounts,
                        ),
                    )
                    self.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr

                    cache_key = f"{model_id}_stream_client"
                    _client = openai.AzureOpenAI(  # type: ignore
                        **azure_client_params,
                        timeout=stream_timeout,
                        max_retries=max_retries,
                        http_client=httpx.Client(
                            transport=CustomHTTPTransport(),
                            limits=httpx.Limits(
                                max_connections=1000, max_keepalive_connections=100
                            ),
                            mounts=sync_proxy_mounts,
                        ),
                    )
                    self.cache.set_cache(
                        key=cache_key,
                        value=_client,
                        ttl=client_ttl,
                        local_only=True,
                    )  # cache for 1 hr

            else:
                _api_key = api_key
                if _api_key is not None and isinstance(_api_key, str):
                    # only show first 5 chars of api_key
                    _api_key = _api_key[:8] + "*" * 15
                verbose_router_logger.debug(
                    f"Initializing OpenAI Client for {model_name}, Api Base:{str(api_base)}, Api Key:{_api_key}"
                )
                cache_key = f"{model_id}_async_client"
                _client = openai.AsyncOpenAI(  # type: ignore
                    api_key=api_key,
                    base_url=api_base,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    http_client=httpx.AsyncClient(
                        transport=AsyncCustomHTTPTransport(),
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        mounts=async_proxy_mounts,
                    ),  # type: ignore
                )
                self.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr

                cache_key = f"{model_id}_client"
                _client = openai.OpenAI(  # type: ignore
                    api_key=api_key,
                    base_url=api_base,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    http_client=httpx.Client(
                        transport=CustomHTTPTransport(),
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        mounts=sync_proxy_mounts,
                    ),  # type: ignore
                )
                self.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr

                # streaming clients should have diff timeouts
                cache_key = f"{model_id}_stream_async_client"
                _client = openai.AsyncOpenAI(  # type: ignore
                    api_key=api_key,
                    base_url=api_base,
                    timeout=stream_timeout,
                    max_retries=max_retries,
                    organization=organization,
                    http_client=httpx.AsyncClient(
                        transport=AsyncCustomHTTPTransport(),
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        mounts=async_proxy_mounts,
                    ),  # type: ignore
                )
                self.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr

                # streaming clients should have diff timeouts
                cache_key = f"{model_id}_stream_client"
                _client = openai.OpenAI(  # type: ignore
                    api_key=api_key,
                    base_url=api_base,
                    timeout=stream_timeout,
                    max_retries=max_retries,
                    organization=organization,
                    http_client=httpx.Client(
                        transport=CustomHTTPTransport(),
                        limits=httpx.Limits(
                            max_connections=1000, max_keepalive_connections=100
                        ),
                        mounts=sync_proxy_mounts,
                    ),  # type: ignore
                )
                self.cache.set_cache(
                    key=cache_key,
                    value=_client,
                    ttl=client_ttl,
                    local_only=True,
                )  # cache for 1 hr

    def _generate_model_id(self, model_group: str, litellm_params: dict):
        """
        Helper function to consistently generate the same id for a deployment

        - create a string from all the litellm params
        - hash
        - use hash as id
        """
        concat_str = model_group
        for k, v in litellm_params.items():
            if isinstance(k, str):
                concat_str += k
            elif isinstance(k, dict):
                concat_str += json.dumps(k)
            else:
                concat_str += str(k)

            if isinstance(v, str):
                concat_str += v
            elif isinstance(v, dict):
                concat_str += json.dumps(v)
            else:
                concat_str += str(v)

        hash_object = hashlib.sha256(concat_str.encode())

        return hash_object.hexdigest()

    def set_model_list(self, model_list: list):
        original_model_list = copy.deepcopy(model_list)
        self.model_list = []
        # we add api_base/api_key each model so load balancing between azure/gpt on api_base1 and api_base2 works
        import os

        for model in original_model_list:
            _model_name = model.pop("model_name")
            _litellm_params = model.pop("litellm_params")
            ## check if litellm params in os.environ
            if isinstance(_litellm_params, dict):
                for k, v in _litellm_params.items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        _litellm_params[k] = litellm.get_secret(v)

            _model_info: dict = model.pop("model_info", {})

            # check if model info has id
            if "id" not in _model_info:
                _id = self._generate_model_id(_model_name, _litellm_params)
                _model_info["id"] = _id

            deployment = Deployment(
                **model,
                model_name=_model_name,
                litellm_params=_litellm_params,
                model_info=_model_info,
            )

            deployment = self._add_deployment(deployment=deployment)

            model = deployment.to_json(exclude_none=True)

            self.model_list.append(model)

        verbose_router_logger.debug(f"\nInitialized Model List {self.model_list}")
        self.model_names = [m["model_name"] for m in model_list]

    def _add_deployment(self, deployment: Deployment) -> Deployment:
        import os

        #### DEPLOYMENT NAMES INIT ########
        self.deployment_names.append(deployment.litellm_params.model)
        ############ Users can either pass tpm/rpm as a litellm_param or a router param ###########
        # for get_available_deployment, we use the litellm_param["rpm"]
        # in this snippet we also set rpm to be a litellm_param
        if (
            deployment.litellm_params.rpm is None
            and getattr(deployment, "rpm", None) is not None
        ):
            deployment.litellm_params.rpm = getattr(deployment, "rpm")

        if (
            deployment.litellm_params.tpm is None
            and getattr(deployment, "tpm", None) is not None
        ):
            deployment.litellm_params.tpm = getattr(deployment, "tpm")

        #### VALIDATE MODEL ########
        # check if model provider in supported providers
        (
            _model,
            custom_llm_provider,
            dynamic_api_key,
            api_base,
        ) = litellm.get_llm_provider(
            model=deployment.litellm_params.model,
            custom_llm_provider=deployment.litellm_params.get(
                "custom_llm_provider", None
            ),
        )

        # Check if user is trying to use model_name == "*"
        # this is a catch all model for their specific api key
        if deployment.model_name == "*":
            self.default_deployment = deployment.to_json(exclude_none=True)

        # Azure GPT-Vision Enhancements, users can pass os.environ/
        data_sources = deployment.litellm_params.get("dataSources", [])

        for data_source in data_sources:
            params = data_source.get("parameters", {})
            for param_key in ["endpoint", "key"]:
                # if endpoint or key set for Azure GPT Vision Enhancements, check if it's an env var
                if param_key in params and params[param_key].startswith("os.environ/"):
                    env_name = params[param_key].replace("os.environ/", "")
                    params[param_key] = os.environ.get(env_name, "")

        # done reading model["litellm_params"]
        if custom_llm_provider not in litellm.provider_list:
            raise Exception(f"Unsupported provider - {custom_llm_provider}")

        # init OpenAI, Azure clients
        self.set_client(model=deployment.to_json(exclude_none=True))

        return deployment

    def add_deployment(self, deployment: Deployment) -> Optional[Deployment]:
        """
        Parameters:
        - deployment: Deployment - the deployment to be added to the Router

        Returns:
        - The added deployment
        - OR None (if deployment already exists)
        """
        # check if deployment already exists

        if deployment.model_info.id in self.get_model_ids():
            return None

        # add to model list
        _deployment = deployment.to_json(exclude_none=True)
        self.model_list.append(_deployment)

        # initialize client
        self._add_deployment(deployment=deployment)

        # add to model names
        self.model_names.append(deployment.model_name)
        return deployment

    def delete_deployment(self, id: str) -> Optional[Deployment]:
        """
        Parameters:
        - id: str - the id of the deployment to be deleted

        Returns:
        - The deleted deployment
        - OR None (if deleted deployment not found)
        """
        deployment_idx = None
        for idx, m in enumerate(self.model_list):
            if m["model_info"]["id"] == id:
                deployment_idx = idx

        try:
            if deployment_idx is not None:
                item = self.model_list.pop(deployment_idx)
                return item
            else:
                return None
        except:
            return None

    def get_deployment(self, model_id: str):
        for model in self.model_list:
            if "model_info" in model and "id" in model["model_info"]:
                if model_id == model["model_info"]["id"]:
                    return model
        return None

    def get_model_ids(self):
        ids = []
        for model in self.model_list:
            if "model_info" in model and "id" in model["model_info"]:
                id = model["model_info"]["id"]
                ids.append(id)
        return ids

    def get_model_names(self):
        return self.model_names

    def get_model_list(self):
        if hasattr(self, "model_list"):
            return self.model_list
        return None

    def get_settings(self):
        """
        Get router settings method, returns a dictionary of the settings and their values.
        For example get the set values for routing_strategy_args, routing_strategy, allowed_fails, cooldown_time, num_retries, timeout, max_retries, retry_after
        """
        _all_vars = vars(self)
        _settings_to_return = {}
        vars_to_include = [
            "routing_strategy_args",
            "routing_strategy",
            "allowed_fails",
            "cooldown_time",
            "num_retries",
            "timeout",
            "max_retries",
            "retry_after",
        ]

        for var in vars_to_include:
            if var in _all_vars:
                _settings_to_return[var] = _all_vars[var]
        return _settings_to_return

    def update_settings(self, **kwargs):
        # only the following settings are allowed to be configured
        _allowed_settings = [
            "routing_strategy_args",
            "routing_strategy",
            "allowed_fails",
            "cooldown_time",
            "num_retries",
            "timeout",
            "max_retries",
            "retry_after",
        ]

        for var in kwargs:
            if var in _allowed_settings:
                setattr(self, var, kwargs[var])
            else:
                verbose_router_logger.debug("Setting {} is not allowed".format(var))

    def _get_client(self, deployment, kwargs, client_type=None):
        """
        Returns the appropriate client based on the given deployment, kwargs, and client_type.

        Parameters:
            deployment (dict): The deployment dictionary containing the clients.
            kwargs (dict): The keyword arguments passed to the function.
            client_type (str): The type of client to return.

        Returns:
            The appropriate client based on the given client_type and kwargs.
        """
        model_id = deployment["model_info"]["id"]
        if client_type == "max_parallel_requests":
            cache_key = "{}_max_parallel_requests_client".format(model_id)
            client = self.cache.get_cache(key=cache_key, local_only=True)
            return client
        elif client_type == "async":
            if kwargs.get("stream") == True:
                cache_key = f"{model_id}_stream_async_client"
                client = self.cache.get_cache(key=cache_key, local_only=True)
                if client is None:
                    """
                    Re-initialize the client
                    """
                    self.set_client(model=deployment)
                    client = self.cache.get_cache(key=cache_key, local_only=True)
                return client
            else:
                cache_key = f"{model_id}_async_client"
                client = self.cache.get_cache(key=cache_key, local_only=True)
                if client is None:
                    """
                    Re-initialize the client
                    """
                    self.set_client(model=deployment)
                    client = self.cache.get_cache(key=cache_key, local_only=True)
                return client
        else:
            if kwargs.get("stream") == True:
                cache_key = f"{model_id}_stream_client"
                client = self.cache.get_cache(key=cache_key)
                if client is None:
                    """
                    Re-initialize the client
                    """
                    self.set_client(model=deployment)
                    client = self.cache.get_cache(key=cache_key)
                return client
            else:
                cache_key = f"{model_id}_client"
                client = self.cache.get_cache(key=cache_key)
                if client is None:
                    """
                    Re-initialize the client
                    """
                    self.set_client(model=deployment)
                    client = self.cache.get_cache(key=cache_key)
                return client

    def _pre_call_checks(
        self,
        model: str,
        healthy_deployments: List,
        messages: List[Dict[str, str]],
    ):
        """
        Filter out model in model group, if:

        - model context window < message length
        - filter models above rpm limits
        - [TODO] function call and model doesn't support function calling
        """
        verbose_router_logger.debug(
            f"Starting Pre-call checks for deployments in model={model}"
        )

        _returned_deployments = copy.deepcopy(healthy_deployments)

        invalid_model_indices = []

        try:
            input_tokens = litellm.token_counter(messages=messages)
        except Exception as e:
            return _returned_deployments

        _context_window_error = False
        _rate_limit_error = False

        ## get model group RPM ##
        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        rpm_key = f"{model}:rpm:{current_minute}"
        model_group_cache = (
            self.cache.get_cache(key=rpm_key, local_only=True) or {}
        )  # check the in-memory cache used by lowest_latency and usage-based routing. Only check the local cache.
        for idx, deployment in enumerate(_returned_deployments):
            # see if we have the info for this model
            try:
                base_model = deployment.get("model_info", {}).get("base_model", None)
                if base_model is None:
                    base_model = deployment.get("litellm_params", {}).get(
                        "base_model", None
                    )
                model = base_model or deployment.get("litellm_params", {}).get(
                    "model", None
                )
                model_info = litellm.get_model_info(model=model)

                if (
                    isinstance(model_info, dict)
                    and model_info.get("max_input_tokens", None) is not None
                ):
                    if (
                        isinstance(model_info["max_input_tokens"], int)
                        and input_tokens > model_info["max_input_tokens"]
                    ):
                        invalid_model_indices.append(idx)
                        _context_window_error = True
                        continue
            except Exception as e:
                verbose_router_logger.debug("An error occurs - {}".format(str(e)))

            ## RPM CHECK ##
            _litellm_params = deployment.get("litellm_params", {})
            model_id = deployment.get("model_info", {}).get("id", "")
            ### get local router cache ###
            current_request_cache_local = (
                self.cache.get_cache(key=model_id, local_only=True) or 0
            )
            ### get usage based cache ###
            if isinstance(model_group_cache, dict):
                model_group_cache[model_id] = model_group_cache.get(model_id, 0)

                current_request = max(
                    current_request_cache_local, model_group_cache[model_id]
                )

                if (
                    isinstance(_litellm_params, dict)
                    and _litellm_params.get("rpm", None) is not None
                ):
                    if (
                        isinstance(_litellm_params["rpm"], int)
                        and _litellm_params["rpm"] <= current_request
                    ):
                        invalid_model_indices.append(idx)
                        _rate_limit_error = True
                        continue

        if len(invalid_model_indices) == len(_returned_deployments):
            """
            - no healthy deployments available b/c context window checks or rate limit error

            - First check for rate limit errors (if this is true, it means the model passed the context window check but failed the rate limit check)
            """

            if _rate_limit_error == True:  # allow generic fallback logic to take place
                raise ValueError(
                    f"No deployments available for selected model, passed model={model}"
                )
            elif _context_window_error == True:
                raise litellm.ContextWindowExceededError(
                    message="Context Window exceeded for given call",
                    model=model,
                    llm_provider="",
                    response=httpx.Response(
                        status_code=400,
                        request=httpx.Request("GET", "https://example.com"),
                    ),
                )
        if len(invalid_model_indices) > 0:
            for idx in reversed(invalid_model_indices):
                _returned_deployments.pop(idx)

        return _returned_deployments

    def _common_checks_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ):
        """
        Common checks for 'get_available_deployment' across sync + async call.

        If 'healthy_deployments' returned is None, this means the user chose a specific deployment
        """
        # check if aliases set on litellm model alias map
        if specific_deployment == True:
            # users can also specify a specific deployment name. At this point we should check if they are just trying to call a specific deployment
            for deployment in self.model_list:
                deployment_model = deployment.get("litellm_params").get("model")
                if deployment_model == model:
                    # User Passed a specific deployment name on their config.yaml, example azure/chat-gpt-v-2
                    # return the first deployment where the `model` matches the specificed deployment name
                    return deployment, None
            raise ValueError(
                f"LiteLLM Router: Trying to call specific deployment, but Model:{model} does not exist in Model List: {self.model_list}"
            )

        if model in self.model_group_alias:
            verbose_router_logger.debug(
                f"Using a model alias. Got Request for {model}, sending requests to {self.model_group_alias.get(model)}"
            )
            model = self.model_group_alias[model]

        if model not in self.model_names and self.default_deployment is not None:
            updated_deployment = copy.deepcopy(
                self.default_deployment
            )  # self.default_deployment
            updated_deployment["litellm_params"]["model"] = model
            return updated_deployment, None

        ## get healthy deployments
        ### get all deployments
        healthy_deployments = [m for m in self.model_list if m["model_name"] == model]
        if len(healthy_deployments) == 0:
            # check if the user sent in a deployment name instead
            healthy_deployments = [
                m for m in self.model_list if m["litellm_params"]["model"] == model
            ]

        verbose_router_logger.debug(
            f"initial list of deployments: {healthy_deployments}"
        )

        verbose_router_logger.debug(
            f"healthy deployments: length {len(healthy_deployments)} {healthy_deployments}"
        )
        if len(healthy_deployments) == 0:
            raise ValueError(f"No healthy deployment available, passed model={model}")
        if litellm.model_alias_map and model in litellm.model_alias_map:
            model = litellm.model_alias_map[
                model
            ]  # update the model to the actual value if an alias has been passed in

        return model, healthy_deployments

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ):
        """
        Async implementation of 'get_available_deployments'.

        Allows all cache calls to be made async => 10x perf impact (8rps -> 100 rps).
        """
        if (
            self.routing_strategy != "usage-based-routing-v2"
            and self.routing_strategy != "simple-shuffle"
        ):  # prevent regressions for other routing strategies, that don't have async get available deployments implemented.
            return self.get_available_deployment(
                model=model,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
            )

        model, healthy_deployments = self._common_checks_available_deployment(
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )

        if healthy_deployments is None:
            return model

        # filter out the deployments currently cooling down
        deployments_to_remove = []
        # cooldown_deployments is a list of model_id's cooling down, cooldown_deployments = ["16700539-b3cd-42f4-b426-6a12a1bb706a", "16700539-b3cd-42f4-b426-7899"]
        cooldown_deployments = await self._async_get_cooldown_deployments()
        verbose_router_logger.debug(
            f"async cooldown deployments: {cooldown_deployments}"
        )
        # Find deployments in model_list whose model_id is cooling down
        for deployment in healthy_deployments:
            deployment_id = deployment["model_info"]["id"]
            if deployment_id in cooldown_deployments:
                deployments_to_remove.append(deployment)
        # remove unhealthy deployments from healthy deployments
        for deployment in deployments_to_remove:
            healthy_deployments.remove(deployment)

        # filter pre-call checks
        if self.enable_pre_call_checks and messages is not None:
            healthy_deployments = self._pre_call_checks(
                model=model, healthy_deployments=healthy_deployments, messages=messages
            )

        if (
            self.routing_strategy == "usage-based-routing-v2"
            and self.lowesttpm_logger_v2 is not None
        ):
            deployment = await self.lowesttpm_logger_v2.async_get_available_deployments(
                model_group=model,
                healthy_deployments=healthy_deployments,
                messages=messages,
                input=input,
            )
        elif self.routing_strategy == "simple-shuffle":
            # if users pass rpm or tpm, we do a random weighted pick - based on rpm/tpm
            ############## Check if we can do a RPM/TPM based weighted pick #################
            rpm = healthy_deployments[0].get("litellm_params").get("rpm", None)
            if rpm is not None:
                # use weight-random pick if rpms provided
                rpms = [m["litellm_params"].get("rpm", 0) for m in healthy_deployments]
                verbose_router_logger.debug(f"\nrpms {rpms}")
                total_rpm = sum(rpms)
                weights = [rpm / total_rpm for rpm in rpms]
                verbose_router_logger.debug(f"\n weights {weights}")
                # Perform weighted random pick
                selected_index = random.choices(range(len(rpms)), weights=weights)[0]
                verbose_router_logger.debug(f"\n selected index, {selected_index}")
                deployment = healthy_deployments[selected_index]
                verbose_router_logger.info(
                    f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment) or deployment[0]} for model: {model}"
                )
                return deployment or deployment[0]
            ############## Check if we can do a RPM/TPM based weighted pick #################
            tpm = healthy_deployments[0].get("litellm_params").get("tpm", None)
            if tpm is not None:
                # use weight-random pick if rpms provided
                tpms = [m["litellm_params"].get("tpm", 0) for m in healthy_deployments]
                verbose_router_logger.debug(f"\ntpms {tpms}")
                total_tpm = sum(tpms)
                weights = [tpm / total_tpm for tpm in tpms]
                verbose_router_logger.debug(f"\n weights {weights}")
                # Perform weighted random pick
                selected_index = random.choices(range(len(tpms)), weights=weights)[0]
                verbose_router_logger.debug(f"\n selected index, {selected_index}")
                deployment = healthy_deployments[selected_index]
                verbose_router_logger.info(
                    f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment) or deployment[0]} for model: {model}"
                )
                return deployment or deployment[0]

            ############## No RPM/TPM passed, we do a random pick #################
            item = random.choice(healthy_deployments)
            return item or item[0]
        if deployment is None:
            verbose_router_logger.info(
                f"get_available_deployment for model: {model}, No deployment available"
            )
            raise ValueError(
                f"No deployments available for selected model, passed model={model}"
            )
        verbose_router_logger.info(
            f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment)} for model: {model}"
        )

        return deployment

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ):
        """
        Returns the deployment based on routing strategy
        """
        # users need to explicitly call a specific deployment, by setting `specific_deployment = True` as completion()/embedding() kwarg
        # When this was no explicit we had several issues with fallbacks timing out

        model, healthy_deployments = self._common_checks_available_deployment(
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )

        if healthy_deployments is None:
            return model

        # filter out the deployments currently cooling down
        deployments_to_remove = []
        # cooldown_deployments is a list of model_id's cooling down, cooldown_deployments = ["16700539-b3cd-42f4-b426-6a12a1bb706a", "16700539-b3cd-42f4-b426-7899"]
        cooldown_deployments = self._get_cooldown_deployments()
        verbose_router_logger.debug(f"cooldown deployments: {cooldown_deployments}")
        # Find deployments in model_list whose model_id is cooling down
        for deployment in healthy_deployments:
            deployment_id = deployment["model_info"]["id"]
            if deployment_id in cooldown_deployments:
                deployments_to_remove.append(deployment)
        # remove unhealthy deployments from healthy deployments
        for deployment in deployments_to_remove:
            healthy_deployments.remove(deployment)

        # filter pre-call checks
        if self.enable_pre_call_checks and messages is not None:
            healthy_deployments = self._pre_call_checks(
                model=model, healthy_deployments=healthy_deployments, messages=messages
            )

        if self.routing_strategy == "least-busy" and self.leastbusy_logger is not None:
            deployment = self.leastbusy_logger.get_available_deployments(
                model_group=model, healthy_deployments=healthy_deployments
            )
        elif self.routing_strategy == "simple-shuffle":
            # if users pass rpm or tpm, we do a random weighted pick - based on rpm/tpm
            ############## Check if we can do a RPM/TPM based weighted pick #################
            rpm = healthy_deployments[0].get("litellm_params").get("rpm", None)
            if rpm is not None:
                # use weight-random pick if rpms provided
                rpms = [m["litellm_params"].get("rpm", 0) for m in healthy_deployments]
                verbose_router_logger.debug(f"\nrpms {rpms}")
                total_rpm = sum(rpms)
                weights = [rpm / total_rpm for rpm in rpms]
                verbose_router_logger.debug(f"\n weights {weights}")
                # Perform weighted random pick
                selected_index = random.choices(range(len(rpms)), weights=weights)[0]
                verbose_router_logger.debug(f"\n selected index, {selected_index}")
                deployment = healthy_deployments[selected_index]
                verbose_router_logger.info(
                    f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment) or deployment[0]} for model: {model}"
                )
                return deployment or deployment[0]
            ############## Check if we can do a RPM/TPM based weighted pick #################
            tpm = healthy_deployments[0].get("litellm_params").get("tpm", None)
            if tpm is not None:
                # use weight-random pick if rpms provided
                tpms = [m["litellm_params"].get("tpm", 0) for m in healthy_deployments]
                verbose_router_logger.debug(f"\ntpms {tpms}")
                total_tpm = sum(tpms)
                weights = [tpm / total_tpm for tpm in tpms]
                verbose_router_logger.debug(f"\n weights {weights}")
                # Perform weighted random pick
                selected_index = random.choices(range(len(tpms)), weights=weights)[0]
                verbose_router_logger.debug(f"\n selected index, {selected_index}")
                deployment = healthy_deployments[selected_index]
                verbose_router_logger.info(
                    f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment) or deployment[0]} for model: {model}"
                )
                return deployment or deployment[0]

            ############## No RPM/TPM passed, we do a random pick #################
            item = random.choice(healthy_deployments)
            return item or item[0]
        elif (
            self.routing_strategy == "latency-based-routing"
            and self.lowestlatency_logger is not None
        ):
            deployment = self.lowestlatency_logger.get_available_deployments(
                model_group=model, healthy_deployments=healthy_deployments
            )
        elif (
            self.routing_strategy == "usage-based-routing"
            and self.lowesttpm_logger is not None
        ):
            deployment = self.lowesttpm_logger.get_available_deployments(
                model_group=model,
                healthy_deployments=healthy_deployments,
                messages=messages,
                input=input,
            )
        elif (
            self.routing_strategy == "usage-based-routing-v2"
            and self.lowesttpm_logger_v2 is not None
        ):
            deployment = self.lowesttpm_logger_v2.get_available_deployments(
                model_group=model,
                healthy_deployments=healthy_deployments,
                messages=messages,
                input=input,
            )
        if deployment is None:
            verbose_router_logger.info(
                f"get_available_deployment for model: {model}, No deployment available"
            )
            raise ValueError(
                f"No deployments available for selected model, passed model={model}"
            )
        verbose_router_logger.info(
            f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment)} for model: {model}"
        )
        return deployment

    def _track_deployment_metrics(self, deployment, response=None):
        try:
            litellm_params = deployment["litellm_params"]
            api_base = litellm_params.get("api_base", "")
            model = litellm_params.get("model", "")

            model_id = deployment.get("model_info", {}).get("id", None)
            if response is None:

                # update self.deployment_stats
                if model_id is not None:
                    self._update_usage(model_id)  # update in-memory cache for tracking
                    if model_id in self.deployment_stats:
                        # only update num_requests
                        self.deployment_stats[model_id]["num_requests"] += 1
                    else:
                        self.deployment_stats[model_id] = {
                            "api_base": api_base,
                            "model": model,
                            "num_requests": 1,
                        }
            else:
                # check response_ms and update num_successes
                if isinstance(response, dict):
                    response_ms = response.get("_response_ms", 0)
                else:
                    response_ms = 0
                if model_id is not None:
                    if model_id in self.deployment_stats:
                        # check if avg_latency exists
                        if "avg_latency" in self.deployment_stats[model_id]:
                            # update avg_latency
                            self.deployment_stats[model_id]["avg_latency"] = (
                                self.deployment_stats[model_id]["avg_latency"]
                                + response_ms
                            ) / self.deployment_stats[model_id]["num_successes"]
                        else:
                            self.deployment_stats[model_id]["avg_latency"] = response_ms

                        # check if num_successes exists
                        if "num_successes" in self.deployment_stats[model_id]:
                            self.deployment_stats[model_id]["num_successes"] += 1
                        else:
                            self.deployment_stats[model_id]["num_successes"] = 1
                    else:
                        self.deployment_stats[model_id] = {
                            "api_base": api_base,
                            "model": model,
                            "num_successes": 1,
                            "avg_latency": response_ms,
                        }
            if self.set_verbose == True and self.debug_level == "DEBUG":
                from pprint import pformat

                # Assuming self.deployment_stats is your dictionary
                formatted_stats = pformat(self.deployment_stats)

                # Assuming verbose_router_logger is your logger
                verbose_router_logger.info(
                    "self.deployment_stats: \n%s", formatted_stats
                )
        except Exception as e:
            verbose_router_logger.error(f"Error in _track_deployment_metrics: {str(e)}")

    def flush_cache(self):
        litellm.cache = None
        self.cache.flush_cache()

    def reset(self):
        ## clean up on close
        litellm.success_callback = []
        litellm.__async_success_callback = []
        litellm.failure_callback = []
        litellm._async_failure_callback = []
        self.flush_cache()

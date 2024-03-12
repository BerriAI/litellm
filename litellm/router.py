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
import litellm, openai
from litellm.caching import RedisCache, InMemoryCache, DualCache

import logging, asyncio
import inspect, concurrent
from openai import AsyncOpenAI
from collections import defaultdict
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.llms.custom_httpx.azure_dall_e_2 import (
    CustomHTTPTransport,
    AsyncCustomHTTPTransport,
)
from litellm.utils import ModelResponse, CustomStreamWrapper
import copy
from litellm._logging import verbose_router_logger
import logging


class Router:
    """
    Example usage:
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
        set_verbose: bool = False,
        debug_level: Literal["DEBUG", "INFO"] = "INFO",
        fallbacks: List = [],
        context_window_fallbacks: List = [],
        model_group_alias: Optional[dict] = {},
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
            model_group_alias (Optional[dict]): Alias for model groups. Defaults to {}.
            retry_after (int): Minimum time to wait before retrying a failed request. Defaults to 0.
            allowed_fails (Optional[int]): Number of allowed fails before adding to cooldown. Defaults to None.
            cooldown_time (float): Time to cooldown a deployment after failure in seconds. Defaults to 1.
            routing_strategy (Literal["simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing"]): Routing strategy. Defaults to "simple-shuffle".
            routing_strategy_args (dict): Additional args for latency-based routing. Defaults to {}.

        Returns:
            Router: An instance of the litellm.Router class.
        """
        self.set_verbose = set_verbose
        self.debug_level = debug_level
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
        self.chat = litellm.Chat(params=default_litellm_params)

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
        verbose_router_logger.debug(
            f"Intialized router with Routing strategy: {self.routing_strategy}\n"
        )

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
            kwargs["original_function"] = self._completion
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            kwargs.setdefault("metadata", {}).update({"model_group": model})
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
            kwargs["original_function"] = self._acompletion
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})

            response = await self.async_function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            raise e

    async def _acompletion(self, model: str, messages: List[Dict[str, str]], **kwargs):
        model_name = None
        try:
            verbose_router_logger.debug(
                f"Inside _acompletion()- model: {model}; kwargs: {kwargs}"
            )
            deployment = self.get_available_deployment(
                model=model,
                messages=messages,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            if self.set_verbose == True and self.debug_level == "DEBUG":
                # debug how often this deployment picked
                self._print_deployment_metrics(deployment=deployment)

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

            response = await litellm.acompletion(
                **{
                    **data,
                    "messages": messages,
                    "caching": self.cache_responses,
                    "client": model_client,
                    "timeout": timeout,
                    **kwargs,
                }
            )
            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.acompletion(model={model_name})\033[32m 200 OK\033[0m"
            )
            if self.set_verbose == True and self.debug_level == "DEBUG":
                # debug how often this deployment picked
                self._print_deployment_metrics(deployment=deployment, response=response)
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
            response = await litellm.aimage_generation(
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
            response = await litellm.atranscription(
                **{
                    **data,
                    "file": file,
                    "caching": self.cache_responses,
                    "client": model_client,
                    **kwargs,
                }
            )
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
            deployment = self.get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": prompt}],
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
            response = await litellm.atext_completion(
                **{
                    **data,
                    "prompt": prompt,
                    "caching": self.cache_responses,
                    "client": model_client,
                    "timeout": self.timeout,
                    **kwargs,
                }
            )
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
        try:
            verbose_router_logger.debug(
                f"Inside _aembedding()- model: {model}; kwargs: {kwargs}"
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
            response = await litellm.aembedding(
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
                if (
                    hasattr(e, "status_code") and e.status_code == 400
                ):  # don't retry a malformed request
                    raise e
                verbose_router_logger.debug(f"Trying to fallback b/w models")
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
                and context_window_fallbacks is None
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
                and context_window_fallbacks is None
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

    def _set_cooldown_deployments(self, deployment: Optional[str] = None):
        """
        Add a model to the list of models being cooled down for that minute, if it exceeds the allowed fails / minute
        """
        if deployment is None:
            return

        current_minute = datetime.now().strftime("%H-%M")
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

    def _get_cooldown_deployments(self):
        """
        Get the list of models being cooled down for this minute
        """
        current_minute = datetime.now().strftime("%H-%M")
        # get the current cooldown list for that minute
        cooldown_key = f"{current_minute}:cooldown_models"

        # ----------------------
        # Return cooldown models
        # ----------------------
        cooldown_models = self.cache.get_cache(key=cooldown_key) or []

        verbose_router_logger.debug(f"retrieve cooldown models: {cooldown_models}")
        return cooldown_models

    def set_client(self, model: dict):
        """
        Initializes Azure/OpenAI clients. Stores them in cache, b/c of this - https://github.com/BerriAI/litellm/issues/1278
        """
        client_ttl = self.client_ttl
        litellm_params = model.get("litellm_params", {})
        model_name = litellm_params.get("model")
        model_id = model["model_info"]["id"]
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
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "openai"
            or "ft:gpt-3.5-turbo" in model_name
            or model_name in litellm.open_ai_embedding_models
        ):
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
                    verbose_router_logger.debug(
                        f"Initializing Azure OpenAI Client for {model_name}, Api Base: {str(api_base)}, Api Key:{api_key}"
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
                verbose_router_logger.debug(
                    f"Initializing OpenAI Client for {model_name}, Api Base:{str(api_base)}, Api Key:{api_key}"
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

    def set_model_list(self, model_list: list):
        self.model_list = copy.deepcopy(model_list)
        # we add api_base/api_key each model so load balancing between azure/gpt on api_base1 and api_base2 works
        import os

        for model in self.model_list:
            #### MODEL ID INIT ########
            model_info = model.get("model_info", {})
            model_info["id"] = model_info.get("id", str(uuid.uuid4()))
            model["model_info"] = model_info
            #### DEPLOYMENT NAMES INIT ########
            self.deployment_names.append(model["litellm_params"]["model"])
            ############ Users can either pass tpm/rpm as a litellm_param or a router param ###########
            # for get_available_deployment, we use the litellm_param["rpm"]
            # in this snippet we also set rpm to be a litellm_param
            if (
                model["litellm_params"].get("rpm") is None
                and model.get("rpm") is not None
            ):
                model["litellm_params"]["rpm"] = model.get("rpm")
            if (
                model["litellm_params"].get("tpm") is None
                and model.get("tpm") is not None
            ):
                model["litellm_params"]["tpm"] = model.get("tpm")

            #### VALIDATE MODEL ########
            # check if model provider in supported providers
            (
                _model,
                custom_llm_provider,
                dynamic_api_key,
                api_base,
            ) = litellm.get_llm_provider(
                model=model["litellm_params"]["model"],
                custom_llm_provider=model["litellm_params"].get(
                    "custom_llm_provider", None
                ),
            )

            # Azure GPT-Vision Enhancements, users can pass os.environ/
            data_sources = model.get("litellm_params", {}).get("dataSources", [])

            for data_source in data_sources:
                params = data_source.get("parameters", {})
                for param_key in ["endpoint", "key"]:
                    # if endpoint or key set for Azure GPT Vision Enhancements, check if it's an env var
                    if param_key in params and params[param_key].startswith(
                        "os.environ/"
                    ):
                        env_name = params[param_key].replace("os.environ/", "")
                        params[param_key] = os.environ.get(env_name, "")

            # done reading model["litellm_params"]
            if custom_llm_provider not in litellm.provider_list:
                raise Exception(f"Unsupported provider - {custom_llm_provider}")

            # init OpenAI, Azure clients
            self.set_client(model=model)

        verbose_router_logger.debug(f"\nInitialized Model List {self.model_list}")
        self.model_names = [m["model_name"] for m in model_list]

    def get_model_names(self):
        return self.model_names

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
        if client_type == "async":
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
        if specific_deployment == True:
            # users can also specify a specific deployment name. At this point we should check if they are just trying to call a specific deployment
            for deployment in self.model_list:
                deployment_model = deployment.get("litellm_params").get("model")
                if deployment_model == model:
                    # User Passed a specific deployment name on their config.yaml, example azure/chat-gpt-v-2
                    # return the first deployment where the `model` matches the specificed deployment name
                    return deployment
            raise ValueError(
                f"LiteLLM Router: Trying to call specific deployment, but Model:{model} does not exist in Model List: {self.model_list}"
            )

        # check if aliases set on litellm model alias map
        if model in self.model_group_alias:
            verbose_router_logger.debug(
                f"Using a model alias. Got Request for {model}, sending requests to {self.model_group_alias.get(model)}"
            )
            model = self.model_group_alias[model]

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

        verbose_router_logger.debug(
            f"healthy deployments: length {len(healthy_deployments)} {healthy_deployments}"
        )
        if len(healthy_deployments) == 0:
            raise ValueError("No models available")
        if litellm.model_alias_map and model in litellm.model_alias_map:
            model = litellm.model_alias_map[
                model
            ]  # update the model to the actual value if an alias has been passed in
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

        if deployment is None:
            verbose_router_logger.info(
                f"get_available_deployment for model: {model}, No deployment available"
            )
            raise ValueError("No models available.")
        verbose_router_logger.info(
            f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment)} for model: {model}"
        )
        return deployment

    def _print_deployment_metrics(self, deployment, response=None):
        try:
            litellm_params = deployment["litellm_params"]
            api_base = litellm_params.get("api_base", "")
            model = litellm_params.get("model", "")

            model_id = deployment.get("model_info", {}).get("id", None)
            if response is None:

                # update self.deployment_stats
                if model_id is not None:
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
                response_ms = response.get("_response_ms", 0)
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
            from pprint import pformat

            # Assuming self.deployment_stats is your dictionary
            formatted_stats = pformat(self.deployment_stats)

            # Assuming verbose_router_logger is your logger
            verbose_router_logger.info("self.deployment_stats: \n%s", formatted_stats)
        except Exception as e:
            verbose_router_logger.error(f"Error in _print_deployment_metrics: {str(e)}")

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

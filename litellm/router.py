# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you ! We ❤️ you! - Krrish & Ishaan

import asyncio
import concurrent
import copy
import datetime as datetime_og
import enum
import hashlib
import inspect
import json
import logging
import random
import re
import threading
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import httpx
import openai
from openai import AsyncOpenAI
from typing_extensions import overload

import litellm
from litellm._logging import verbose_router_logger
from litellm.assistants.main import AssistantDeleted
from litellm.caching import DualCache, InMemoryCache, RedisCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.azure import get_azure_ad_token_from_oidc
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.router_strategy.tag_based_routing import get_deployments_for_tag
from litellm.router_utils.client_initalization_utils import (
    set_client,
    should_initialize_sync_client,
)
from litellm.router_utils.cooldown_callbacks import router_cooldown_handler
from litellm.router_utils.handle_error import send_llm_exception_alert
from litellm.scheduler import FlowItem, Scheduler
from litellm.types.llms.openai import (
    Assistant,
    AssistantToolParam,
    AsyncCursorPage,
    Attachment,
    OpenAIMessage,
    Run,
    Thread,
)
from litellm.types.router import (
    SPECIAL_MODEL_INFO_PARAMS,
    AlertingConfig,
    AllowedFailsPolicy,
    AssistantsTypedDict,
    CustomRoutingStrategyBase,
    Deployment,
    DeploymentTypedDict,
    LiteLLM_Params,
    ModelGroupInfo,
    ModelInfo,
    RetryPolicy,
    RouterErrors,
    RouterGeneralSettings,
    updateDeployment,
    updateLiteLLMParams,
)
from litellm.types.utils import ModelInfo as ModelMapInfo
from litellm.utils import (
    CustomStreamWrapper,
    ModelResponse,
    _is_region_eu,
    calculate_max_parallel_requests,
    create_proxy_transport_and_mounts,
    get_utc_datetime,
)


class RoutingArgs(enum.Enum):
    ttl = 60  # 1min (RPM/TPM expire key)


class Router:
    model_names: List = []
    cache_responses: Optional[bool] = False
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour
    tenacity = None
    leastbusy_logger: Optional[LeastBusyLoggingHandler] = None
    lowesttpm_logger: Optional[LowestTPMLoggingHandler] = None

    def __init__(
        self,
        model_list: Optional[
            Union[List[DeploymentTypedDict], List[Dict[str, Any]]]
        ] = None,
        ## ASSISTANTS API ##
        assistants_config: Optional[AssistantsTypedDict] = None,
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
        ## SCHEDULER ##
        polling_interval: Optional[float] = None,
        ## RELIABILITY ##
        num_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        default_litellm_params: Optional[
            dict
        ] = None,  # default params for Router.chat.completion.create
        default_max_parallel_requests: Optional[int] = None,
        set_verbose: bool = False,
        debug_level: Literal["DEBUG", "INFO"] = "INFO",
        default_fallbacks: Optional[
            List[str]
        ] = None,  # generic fallbacks, works across all deployments
        fallbacks: List = [],
        context_window_fallbacks: List = [],
        content_policy_fallbacks: List = [],
        model_group_alias: Optional[dict] = {},
        enable_pre_call_checks: bool = False,
        enable_tag_filtering: bool = False,
        retry_after: int = 0,  # min time to wait before retrying a failed request
        retry_policy: Optional[
            RetryPolicy
        ] = None,  # set custom retries for different exceptions
        model_group_retry_policy: Optional[
            Dict[str, RetryPolicy]
        ] = {},  # set custom retry policies based on model group
        allowed_fails: Optional[
            int
        ] = None,  # Number of times a deployment can failbefore being added to cooldown
        allowed_fails_policy: Optional[
            AllowedFailsPolicy
        ] = None,  # set custom allowed fails policy
        cooldown_time: Optional[
            float
        ] = None,  # (seconds) time to cooldown a deployment after failure
        disable_cooldowns: Optional[bool] = None,
        routing_strategy: Literal[
            "simple-shuffle",
            "least-busy",
            "usage-based-routing",
            "latency-based-routing",
            "cost-based-routing",
            "usage-based-routing-v2",
        ] = "simple-shuffle",
        routing_strategy_args: dict = {},  # just for latency-based routing
        semaphore: Optional[asyncio.Semaphore] = None,
        alerting_config: Optional[AlertingConfig] = None,
        router_general_settings: Optional[
            RouterGeneralSettings
        ] = RouterGeneralSettings(),
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
            polling_interval: (Optional[float]): frequency of polling queue. Only for '.scheduler_acompletion()'. Default is 3ms.
            num_retries (Optional[int]): Number of retries for failed requests. Defaults to 2.
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
            routing_strategy (Literal["simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing", "cost-based-routing"]): Routing strategy. Defaults to "simple-shuffle".
            routing_strategy_args (dict): Additional args for latency-based routing. Defaults to {}.
            alerting_config (AlertingConfig): Slack alerting configuration. Defaults to None.
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
        self.enable_tag_filtering = enable_tag_filtering
        if self.set_verbose == True:
            if debug_level == "INFO":
                verbose_router_logger.setLevel(logging.INFO)
            elif debug_level == "DEBUG":
                verbose_router_logger.setLevel(logging.DEBUG)
        self.router_general_settings: RouterGeneralSettings = (
            router_general_settings or RouterGeneralSettings()
        )

        self.assistants_config = assistants_config
        self.deployment_names: List = (
            []
        )  # names of models under litellm_params. ex. azure/chatgpt-v-2
        self.deployment_latency_map = {}
        ### CACHING ###
        cache_type: Literal["local", "redis", "redis-semantic", "s3", "disk"] = (
            "local"  # default to an in-memory cache
        )
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

        ### SCHEDULER ###
        self.scheduler = Scheduler(
            polling_interval=polling_interval, redis_cache=redis_cache
        )
        self.default_deployment = None  # use this to track the users default deployment, when they want to use model = *
        self.default_max_parallel_requests = default_max_parallel_requests
        self.provider_default_deployments: Dict[str, List] = {}

        if model_list is not None:
            model_list = copy.deepcopy(model_list)
            self.set_model_list(model_list)
            self.healthy_deployments: List = self.model_list  # type: ignore
            for m in model_list:
                self.deployment_latency_map[m["litellm_params"]["model"]] = 0
        else:
            self.model_list: List = (
                []
            )  # initialize an empty list - to allow _add_deployment and delete_deployment to work

        if allowed_fails is not None:
            self.allowed_fails = allowed_fails
        else:
            self.allowed_fails = litellm.allowed_fails
        self.cooldown_time = cooldown_time or 60
        self.disable_cooldowns = disable_cooldowns
        self.failed_calls = (
            InMemoryCache()
        )  # cache to track failed call per deployment, if num failed calls within 1 minute > allowed fails, then add it to cooldown

        if num_retries is not None:
            self.num_retries = num_retries
        elif litellm.num_retries is not None:
            self.num_retries = litellm.num_retries
        else:
            self.num_retries = openai.DEFAULT_MAX_RETRIES

        self.timeout = timeout or litellm.request_timeout

        self.retry_after = retry_after
        self.routing_strategy = routing_strategy

        ## SETTING FALLBACKS ##
        ### validate if it's set + in correct format
        _fallbacks = fallbacks or litellm.fallbacks

        self.validate_fallbacks(fallback_param=_fallbacks)
        ### set fallbacks
        self.fallbacks = _fallbacks

        if default_fallbacks is not None or litellm.default_fallbacks is not None:
            _fallbacks = default_fallbacks or litellm.default_fallbacks
            if self.fallbacks is not None:
                self.fallbacks.append({"*": _fallbacks})
            else:
                self.fallbacks = [{"*": _fallbacks}]

        self.context_window_fallbacks = (
            context_window_fallbacks or litellm.context_window_fallbacks
        )

        _content_policy_fallbacks = (
            content_policy_fallbacks or litellm.content_policy_fallbacks
        )
        self.validate_fallbacks(fallback_param=_content_policy_fallbacks)
        self.content_policy_fallbacks = _content_policy_fallbacks
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
        default_litellm_params = default_litellm_params or {}
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
        self.routing_strategy_init(
            routing_strategy=routing_strategy,
            routing_strategy_args=routing_strategy_args,
        )
        ## USAGE TRACKING ##
        if isinstance(litellm._async_success_callback, list):
            litellm._async_success_callback.append(self.deployment_callback_on_success)
        else:
            litellm._async_success_callback.append(self.deployment_callback_on_success)
        ## COOLDOWNS ##
        if isinstance(litellm.failure_callback, list):
            litellm.failure_callback.append(self.deployment_callback_on_failure)
        else:
            litellm.failure_callback = [self.deployment_callback_on_failure]
        verbose_router_logger.debug(
            f"Intialized router with Routing strategy: {self.routing_strategy}\n\n"
            f"Routing enable_pre_call_checks: {self.enable_pre_call_checks}\n\n"
            f"Routing fallbacks: {self.fallbacks}\n\n"
            f"Routing content fallbacks: {self.content_policy_fallbacks}\n\n"
            f"Routing context window fallbacks: {self.context_window_fallbacks}\n\n"
            f"Router Redis Caching={self.cache.redis_cache}\n"
        )

        self.routing_strategy_args = routing_strategy_args
        self.retry_policy: Optional[RetryPolicy] = retry_policy
        self.model_group_retry_policy: Optional[Dict[str, RetryPolicy]] = (
            model_group_retry_policy
        )
        self.allowed_fails_policy: Optional[AllowedFailsPolicy] = allowed_fails_policy
        self.alerting_config: Optional[AlertingConfig] = alerting_config
        if self.alerting_config is not None:
            self._initialize_alerting()

    def validate_fallbacks(self, fallback_param: Optional[List]):
        if fallback_param is None:
            return

        for fallback_dict in fallback_param:
            if not isinstance(fallback_dict, dict):
                raise ValueError(f"Item '{fallback_dict}' is not a dictionary.")
            if len(fallback_dict) != 1:
                raise ValueError(
                    f"Dictionary '{fallback_dict}' must have exactly one key, but has {len(fallback_dict)} keys."
                )

    def routing_strategy_init(self, routing_strategy: str, routing_strategy_args: dict):
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
                router_cache=self.cache,
                model_list=self.model_list,
                routing_args=routing_strategy_args,
            )
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.lowesttpm_logger)  # type: ignore
        elif routing_strategy == "usage-based-routing-v2":
            self.lowesttpm_logger_v2 = LowestTPMLoggingHandler_v2(
                router_cache=self.cache,
                model_list=self.model_list,
                routing_args=routing_strategy_args,
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
        elif routing_strategy == "cost-based-routing":
            self.lowestcost_logger = LowestCostLoggingHandler(
                router_cache=self.cache,
                model_list=self.model_list,
                routing_args={},
            )
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.lowestcost_logger)  # type: ignore

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

    def _completion(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Union[ModelResponse, CustomStreamWrapper]:
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

            ## CHECK CONTENT FILTER ERROR ##
            if isinstance(response, ModelResponse):
                _should_raise = self._should_raise_content_policy_error(
                    model=model, response=response, kwargs=kwargs
                )
                if _should_raise:
                    raise litellm.ContentPolicyViolationError(
                        message="Response output was blocked.",
                        model=model,
                        llm_provider="",
                    )

            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.completion(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            raise e

    # fmt: off

    @overload
    async def acompletion(
        self, model: str, messages: List[Dict[str, str]], stream: Literal[True], **kwargs
    ) -> CustomStreamWrapper: 
        ...

    @overload
    async def acompletion(
        self, model: str, messages: List[Dict[str, str]], stream: Literal[False] = False, **kwargs
    ) -> ModelResponse: 
        ...

    @overload
    async def acompletion(
        self, model: str, messages: List[Dict[str, str]], stream: Union[Literal[True], Literal[False]] = False, **kwargs
    ) -> Union[CustomStreamWrapper, ModelResponse]: 
        ...

    # fmt: on

    # The actual implementation of the function
    async def acompletion(
        self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    ):
        try:
            kwargs["model"] = model
            kwargs["messages"] = messages
            kwargs["stream"] = stream
            kwargs["original_function"] = self._acompletion
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)

            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})

            if kwargs.get("priority", None) is not None and isinstance(
                kwargs.get("priority"), int
            ):
                response = await self.schedule_acompletion(**kwargs)
            else:
                response = await self.async_function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            asyncio.create_task(
                send_llm_exception_alert(
                    litellm_router_instance=self,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
            raise e

    async def _acompletion(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Union[ModelResponse, CustomStreamWrapper]:
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
                request_kwargs=kwargs,
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
                or data.get(
                    "request_timeout", None
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

            ## CHECK CONTENT FILTER ERROR ##
            if isinstance(response, ModelResponse):
                _should_raise = self._should_raise_content_policy_error(
                    model=model, response=response, kwargs=kwargs
                )
                if _should_raise:
                    raise litellm.ContentPolicyViolationError(
                        message="Response output was blocked.",
                        model=model,
                        llm_provider="",
                    )

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

    async def abatch_completion(
        self,
        models: List[str],
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        **kwargs,
    ):
        """
        Async Batch Completion. Used for 2 scenarios:
        1. Batch Process 1 request to N models on litellm.Router. Pass messages as List[Dict[str, str]] to use this
        2. Batch Process N requests to M models on litellm.Router. Pass messages as List[List[Dict[str, str]]] to use this

        Example Request for 1 request to N models:
        ```
            response = await router.abatch_completion(
                models=["gpt-3.5-turbo", "groq-llama"],
                messages=[
                    {"role": "user", "content": "is litellm becoming a better product ?"}
                ],
                max_tokens=15,
            )
        ```


        Example Request for N requests to M models:
        ```
            response = await router.abatch_completion(
                models=["gpt-3.5-turbo", "groq-llama"],
                messages=[
                    [{"role": "user", "content": "is litellm becoming a better product ?"}],
                    [{"role": "user", "content": "who is this"}],
                ],
            )
        ```
        """
        ############## Helpers for async completion ##################

        async def _async_completion_no_exceptions(
            model: str, messages: List[Dict[str, str]], **kwargs
        ):
            """
            Wrapper around self.async_completion that catches exceptions and returns them as a result
            """
            try:
                return await self.acompletion(model=model, messages=messages, **kwargs)
            except Exception as e:
                return e

        async def _async_completion_no_exceptions_return_idx(
            model: str,
            messages: List[Dict[str, str]],
            idx: int,  # index of message this response corresponds to
            **kwargs,
        ):
            """
            Wrapper around self.async_completion that catches exceptions and returns them as a result
            """
            try:
                return (
                    await self.acompletion(model=model, messages=messages, **kwargs),
                    idx,
                )
            except Exception as e:
                return e, idx

        ############## Helpers for async completion ##################

        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            _tasks = []
            for model in models:
                # add each task but if the task fails
                _tasks.append(_async_completion_no_exceptions(model=model, messages=messages, **kwargs))  # type: ignore
            response = await asyncio.gather(*_tasks)
            return response
        elif isinstance(messages, list) and all(isinstance(m, list) for m in messages):
            _tasks = []
            for idx, message in enumerate(messages):
                for model in models:
                    # Request Number X, Model Number Y
                    _tasks.append(
                        _async_completion_no_exceptions_return_idx(
                            model=model, idx=idx, messages=message, **kwargs  # type: ignore
                        )
                    )
            responses = await asyncio.gather(*_tasks)
            final_responses: List[List[Any]] = [[] for _ in range(len(messages))]
            for response in responses:
                if isinstance(response, tuple):
                    final_responses[response[1]].append(response[0])
                else:
                    final_responses[0].append(response)
            return final_responses

    async def abatch_completion_one_model_multiple_requests(
        self, model: str, messages: List[List[Dict[str, str]]], **kwargs
    ):
        """
        Async Batch Completion - Batch Process multiple Messages to one model_group on litellm.Router

        Use this for sending multiple requests to 1 model

        Args:
            model (List[str]): model group
            messages (List[List[Dict[str, str]]]): list of messages. Each element in the list is one request
            **kwargs: additional kwargs
        Usage:
            response = await self.abatch_completion_one_model_multiple_requests(
                model="gpt-3.5-turbo",
                messages=[
                    [{"role": "user", "content": "hello"}, {"role": "user", "content": "tell me something funny"}],
                    [{"role": "user", "content": "hello good mornign"}],
                ]
            )
        """

        async def _async_completion_no_exceptions(
            model: str, messages: List[Dict[str, str]], **kwargs
        ):
            """
            Wrapper around self.async_completion that catches exceptions and returns them as a result
            """
            try:
                return await self.acompletion(model=model, messages=messages, **kwargs)
            except Exception as e:
                return e

        _tasks = []
        for message_request in messages:
            # add each task but if the task fails
            _tasks.append(
                _async_completion_no_exceptions(
                    model=model, messages=message_request, **kwargs
                )
            )

        response = await asyncio.gather(*_tasks)
        return response

    # fmt: off

    @overload
    async def abatch_completion_fastest_response(
        self, model: str, messages: List[Dict[str, str]], stream: Literal[True], **kwargs
    ) -> CustomStreamWrapper:
        ...



    @overload
    async def abatch_completion_fastest_response(
        self, model: str, messages: List[Dict[str, str]], stream: Literal[False] = False, **kwargs
    ) -> ModelResponse:
        ...

    # fmt: on

    async def abatch_completion_fastest_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ):
        """
        model - List of comma-separated model names. E.g. model="gpt-4, gpt-3.5-turbo"

        Returns fastest response from list of model names. OpenAI-compatible endpoint.
        """
        models = [m.strip() for m in model.split(",")]

        async def _async_completion_no_exceptions(
            model: str, messages: List[Dict[str, str]], stream: bool, **kwargs: Any
        ) -> Union[ModelResponse, CustomStreamWrapper, Exception]:
            """
            Wrapper around self.acompletion that catches exceptions and returns them as a result
            """
            try:
                return await self.acompletion(model=model, messages=messages, stream=stream, **kwargs)  # type: ignore
            except asyncio.CancelledError:
                verbose_router_logger.debug(
                    "Received 'task.cancel'. Cancelling call w/ model={}.".format(model)
                )
                raise
            except Exception as e:
                return e

        pending_tasks = []  # type: ignore

        async def check_response(task: asyncio.Task):
            nonlocal pending_tasks
            try:
                result = await task
                if isinstance(result, (ModelResponse, CustomStreamWrapper)):
                    verbose_router_logger.debug(
                        "Received successful response. Cancelling other LLM API calls."
                    )
                    # If a desired response is received, cancel all other pending tasks
                    for t in pending_tasks:
                        t.cancel()
                    return result
            except Exception:
                # Ignore exceptions, let the loop handle them
                pass
            finally:
                # Remove the task from pending tasks if it finishes
                try:
                    pending_tasks.remove(task)
                except KeyError:
                    pass

        for model in models:
            task = asyncio.create_task(
                _async_completion_no_exceptions(
                    model=model, messages=messages, stream=stream, **kwargs
                )
            )
            pending_tasks.append(task)

        # Await the first task to complete successfully
        while pending_tasks:
            done, pending_tasks = await asyncio.wait(  # type: ignore
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for completed_task in done:
                result = await check_response(completed_task)
                if result is not None:
                    # Return the first successful result
                    result._hidden_params["fastest_response_batch_completion"] = True
                    return result

        # If we exit the loop without returning, all tasks failed
        raise Exception("All tasks failed")

    ### SCHEDULER ###

    # fmt: off

    @overload
    async def schedule_acompletion(
        self, model: str, messages: List[Dict[str, str]], priority: int, stream: Literal[False] = False, **kwargs
    ) -> ModelResponse: 
        ...
    
    @overload
    async def schedule_acompletion(
        self, model: str, messages: List[Dict[str, str]], priority: int, stream: Literal[True], **kwargs
    ) -> CustomStreamWrapper: 
        ...

    # fmt: on

    async def schedule_acompletion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        priority: int,
        stream=False,
        **kwargs,
    ):
        ### FLOW ITEM ###
        _request_id = str(uuid.uuid4())
        item = FlowItem(
            priority=priority,  # 👈 SET PRIORITY FOR REQUEST
            request_id=_request_id,  # 👈 SET REQUEST ID
            model_name="gpt-3.5-turbo",  # 👈 SAME as 'Router'
        )
        ### [fin] ###

        ## ADDS REQUEST TO QUEUE ##
        await self.scheduler.add_request(request=item)

        ## POLL QUEUE
        end_time = time.time() + self.timeout
        curr_time = time.time()
        poll_interval = self.scheduler.polling_interval  # poll every 3ms
        make_request = False

        while curr_time < end_time:
            _healthy_deployments = await self._async_get_healthy_deployments(
                model=model
            )
            make_request = await self.scheduler.poll(  ## POLL QUEUE ## - returns 'True' if there's healthy deployments OR if request is at top of queue
                id=item.request_id,
                model_name=item.model_name,
                health_deployments=_healthy_deployments,
            )
            if make_request:  ## IF TRUE -> MAKE REQUEST
                break
            else:  ## ELSE -> loop till default_timeout
                await asyncio.sleep(poll_interval)
                curr_time = time.time()

        if make_request:
            try:
                _response = await self.acompletion(
                    model=model, messages=messages, stream=stream, **kwargs
                )
                _response._hidden_params.setdefault("additional_headers", {})
                _response._hidden_params["additional_headers"].update(
                    {"x-litellm-request-prioritization-used": True}
                )
                return _response
            except Exception as e:
                setattr(e, "priority", priority)
                raise e
        else:
            raise litellm.Timeout(
                message="Request timed out while polling queue",
                model=model,
                llm_provider="openai",
            )

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
            asyncio.create_task(
                send_llm_exception_alert(
                    litellm_router_instance=self,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
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
            asyncio.create_task(
                send_llm_exception_alert(
                    litellm_router_instance=self,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
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

    async def aspeech(self, model: str, input: str, voice: str, **kwargs):
        """
        Example Usage:

        ```
        from litellm import Router
        client = Router(model_list = [
            {
                "model_name": "tts",
                "litellm_params": {
                    "model": "tts-1",
                },
            },
        ])

        async with client.aspeech(
            model="tts",
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=None,
            api_key=None,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        ) as response:
            response.stream_to_file(speech_file_path)

        ```
        """
        try:
            kwargs["input"] = input
            kwargs["voice"] = voice

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

            response = await litellm.aspeech(**data, **kwargs)

            return response
        except Exception as e:
            asyncio.create_task(
                send_llm_exception_alert(
                    litellm_router_instance=self,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
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
            asyncio.create_task(
                send_llm_exception_alert(
                    litellm_router_instance=self,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
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
            asyncio.create_task(
                send_llm_exception_alert(
                    litellm_router_instance=self,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
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

    async def aadapter_completion(
        self,
        adapter_id: str,
        model: str,
        is_retry: Optional[bool] = False,
        is_fallback: Optional[bool] = False,
        is_async: Optional[bool] = False,
        **kwargs,
    ):
        try:
            kwargs["model"] = model
            kwargs["adapter_id"] = adapter_id
            kwargs["original_function"] = self._aadapter_completion
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            timeout = kwargs.get("request_timeout", self.timeout)
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = await self.async_function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            asyncio.create_task(
                send_llm_exception_alert(
                    litellm_router_instance=self,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
            raise e

    async def _aadapter_completion(self, adapter_id: str, model: str, **kwargs):
        try:
            verbose_router_logger.debug(
                f"Inside _aadapter_completion()- model: {model}; kwargs: {kwargs}"
            )
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "default text"}],
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

            response = litellm.aadapter_completion(
                **{
                    **data,
                    "adapter_id": adapter_id,
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
                    response = await response  # type: ignore
            else:
                await self.async_routing_strategy_pre_call_checks(deployment=deployment)
                response = await response  # type: ignore

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.aadapter_completion(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.aadapter_completion(model={model})\033[31m Exception {str(e)}\033[0m"
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
            asyncio.create_task(
                send_llm_exception_alert(
                    litellm_router_instance=self,
                    request_kwargs=kwargs,
                    error_traceback_str=traceback.format_exc(),
                    original_exception=e,
                )
            )
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

    #### ASSISTANTS API ####

    async def acreate_assistants(
        self,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> Assistant:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.acreate_assistants(
            custom_llm_provider=custom_llm_provider, client=client, **kwargs
        )

    async def adelete_assistant(
        self,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> AssistantDeleted:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.adelete_assistant(
            custom_llm_provider=custom_llm_provider, client=client, **kwargs
        )

    async def aget_assistants(
        self,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> AsyncCursorPage[Assistant]:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.aget_assistants(
            custom_llm_provider=custom_llm_provider, client=client, **kwargs
        )

    async def acreate_thread(
        self,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> Thread:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )
        return await litellm.acreate_thread(
            custom_llm_provider=custom_llm_provider, client=client, **kwargs
        )

    async def aget_thread(
        self,
        thread_id: str,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> Thread:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )
        return await litellm.aget_thread(
            custom_llm_provider=custom_llm_provider,
            thread_id=thread_id,
            client=client,
            **kwargs,
        )

    async def a_add_message(
        self,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: str,
        attachments: Optional[List[Attachment]] = None,
        metadata: Optional[dict] = None,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> OpenAIMessage:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.a_add_message(
            custom_llm_provider=custom_llm_provider,
            thread_id=thread_id,
            role=role,
            content=content,
            attachments=attachments,
            metadata=metadata,
            client=client,
            **kwargs,
        )

    async def aget_messages(
        self,
        thread_id: str,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ) -> AsyncCursorPage[OpenAIMessage]:
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )
        return await litellm.aget_messages(
            custom_llm_provider=custom_llm_provider,
            thread_id=thread_id,
            client=client,
            **kwargs,
        )

    async def arun_thread(
        self,
        thread_id: str,
        assistant_id: str,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        additional_instructions: Optional[str] = None,
        instructions: Optional[str] = None,
        metadata: Optional[dict] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        tools: Optional[Iterable[AssistantToolParam]] = None,
        client: Optional[Any] = None,
        **kwargs,
    ) -> Run:

        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )

        return await litellm.arun_thread(
            custom_llm_provider=custom_llm_provider,
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,
            model=model,
            stream=stream,
            tools=tools,
            client=client,
            **kwargs,
        )

    #### [END] ASSISTANTS API ####

    async def async_function_with_fallbacks(self, *args, **kwargs):
        """
        Try calling the function_with_retries
        If it fails after num_retries, fall back to another model group
        """
        mock_testing_fallbacks = kwargs.pop("mock_testing_fallbacks", None)
        mock_testing_context_fallbacks = kwargs.pop(
            "mock_testing_context_fallbacks", None
        )
        mock_testing_content_policy_fallbacks = kwargs.pop(
            "mock_testing_content_policy_fallbacks", None
        )
        model_group = kwargs.get("model")
        fallbacks = kwargs.get("fallbacks", self.fallbacks)
        context_window_fallbacks = kwargs.get(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        content_policy_fallbacks = kwargs.get(
            "content_policy_fallbacks", self.content_policy_fallbacks
        )
        try:
            if mock_testing_fallbacks is not None and mock_testing_fallbacks is True:
                raise litellm.InternalServerError(
                    model=model_group,
                    llm_provider="",
                    message=f"This is a mock exception for model={model_group}, to trigger a fallback. Fallbacks={fallbacks}",
                )
            elif (
                mock_testing_context_fallbacks is not None
                and mock_testing_context_fallbacks is True
            ):
                raise litellm.ContextWindowExceededError(
                    model=model_group,
                    llm_provider="",
                    message=f"This is a mock exception for model={model_group}, to trigger a fallback. \
                        Context_Window_Fallbacks={context_window_fallbacks}",
                )
            elif (
                mock_testing_content_policy_fallbacks is not None
                and mock_testing_content_policy_fallbacks is True
            ):
                raise litellm.ContentPolicyViolationError(
                    model=model_group,
                    llm_provider="",
                    message=f"This is a mock exception for model={model_group}, to trigger a fallback. \
                        Context_Policy_Fallbacks={content_policy_fallbacks}",
                )

            response = await self.async_function_with_retries(*args, **kwargs)
            verbose_router_logger.debug(f"Async Response: {response}")
            return response
        except Exception as e:
            verbose_router_logger.debug(f"Traceback{traceback.format_exc()}")
            original_exception = e
            fallback_model_group = None
            fallback_failure_exception_str = ""
            try:
                verbose_router_logger.debug("Trying to fallback b/w models")
                if isinstance(e, litellm.ContextWindowExceededError):
                    if context_window_fallbacks is not None:
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
                                verbose_router_logger.info(
                                    "Successful fallback b/w models."
                                )
                                return response
                            except Exception as e:
                                pass
                    else:
                        error_message = "model={}. context_window_fallbacks={}. fallbacks={}.\n\nSet 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks".format(
                            model_group, context_window_fallbacks, fallbacks
                        )
                        verbose_router_logger.info(
                            msg="Got 'ContextWindowExceededError'. No context_window_fallback set. Defaulting \
                            to fallbacks, if available.{}".format(
                                error_message
                            )
                        )

                        e.message += "\n{}".format(error_message)
                elif isinstance(e, litellm.ContentPolicyViolationError):
                    if content_policy_fallbacks is not None:
                        fallback_model_group = None
                        for (
                            item
                        ) in content_policy_fallbacks:  # [{"gpt-3.5-turbo": ["gpt-4"]}]
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
                                verbose_router_logger.info(
                                    "Successful fallback b/w models."
                                )
                                return response
                            except Exception as e:
                                pass
                    else:
                        error_message = "model={}. content_policy_fallback={}. fallbacks={}.\n\nSet 'content_policy_fallback' - https://docs.litellm.ai/docs/routing#fallbacks".format(
                            model_group, content_policy_fallbacks, fallbacks
                        )
                        verbose_router_logger.info(
                            msg="Got 'ContentPolicyViolationError'. No content_policy_fallback set. Defaulting \
                            to fallbacks, if available.{}".format(
                                error_message
                            )
                        )

                        e.message += "\n{}".format(error_message)
                if fallbacks is not None:
                    verbose_router_logger.debug(f"inside model fallbacks: {fallbacks}")
                    generic_fallback_idx: Optional[int] = None
                    ## check for specific model group-specific fallbacks
                    for idx, item in enumerate(fallbacks):
                        if isinstance(item, dict):
                            if list(item.keys())[0] == model_group:
                                fallback_model_group = item[model_group]
                                break
                            elif list(item.keys())[0] == "*":
                                generic_fallback_idx = idx
                        elif isinstance(item, str):
                            fallback_model_group = [fallbacks.pop(idx)]
                    ## if none, check for generic fallback
                    if (
                        fallback_model_group is None
                        and generic_fallback_idx is not None
                    ):
                        fallback_model_group = fallbacks[generic_fallback_idx]["*"]

                    if fallback_model_group is None:
                        verbose_router_logger.info(
                            f"No fallback model group found for original model_group={model_group}. Fallbacks={fallbacks}"
                        )
                        if hasattr(original_exception, "message"):
                            original_exception.message += f"No fallback model group found for original model_group={model_group}. Fallbacks={fallbacks}"
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
                            verbose_router_logger.info(
                                "Successful fallback b/w models."
                            )
                            return response
                        except Exception as e:
                            raise e
            except Exception as new_exception:
                verbose_router_logger.error(
                    "litellm.router.py::async_function_with_fallbacks() - Error occurred while trying to do fallbacks - {}\n{}\n\nDebug Information:\nCooldown Deployments={}".format(
                        str(new_exception),
                        traceback.format_exc(),
                        await self._async_get_cooldown_deployments_with_debug_info(),
                    )
                )
                fallback_failure_exception_str = str(new_exception)

            if hasattr(original_exception, "message"):
                # add the available fallbacks to the exception
                original_exception.message += "\nReceived Model Group={}\nAvailable Model Group Fallbacks={}".format(
                    model_group,
                    fallback_model_group,
                )
                if len(fallback_failure_exception_str) > 0:
                    original_exception.message += (
                        "\nError doing the fallback: {}".format(
                            fallback_failure_exception_str
                        )
                    )

            raise original_exception

    async def async_function_with_retries(self, *args, **kwargs):
        verbose_router_logger.debug(
            f"Inside async function with retries: args - {args}; kwargs - {kwargs}"
        )
        original_function = kwargs.pop("original_function")
        mock_testing_rate_limit_error = kwargs.pop(
            "mock_testing_rate_limit_error", None
        )
        fallbacks = kwargs.pop("fallbacks", self.fallbacks)
        context_window_fallbacks = kwargs.pop(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        content_policy_fallbacks = kwargs.pop(
            "content_policy_fallbacks", self.content_policy_fallbacks
        )
        model_group = kwargs.get("model")
        num_retries = kwargs.pop("num_retries")

        verbose_router_logger.debug(
            f"async function w/ retries: original_function - {original_function}, num_retries - {num_retries}"
        )
        try:
            if (
                mock_testing_rate_limit_error is not None
                and mock_testing_rate_limit_error is True
            ):
                verbose_router_logger.info(
                    "litellm.router.py::async_function_with_retries() - mock_testing_rate_limit_error=True. Raising litellm.RateLimitError."
                )
                raise litellm.RateLimitError(
                    model=model_group,
                    llm_provider="",
                    message=f"This is a mock exception for model={model_group}, to trigger a rate limit error.",
                )
            # if the function call is successful, no exception will be raised and we'll break out of the loop
            response = await original_function(*args, **kwargs)
            return response
        except Exception as e:
            current_attempt = None
            original_exception = e
            """
            Retry Logic
             
            """
            _healthy_deployments = await self._async_get_healthy_deployments(
                model=kwargs.get("model") or "",
            )

            # raises an exception if this error should not be retries
            self.should_retry_this_error(
                error=e,
                healthy_deployments=_healthy_deployments,
                context_window_fallbacks=context_window_fallbacks,
                regular_fallbacks=fallbacks,
                content_policy_fallbacks=content_policy_fallbacks,
            )

            # decides how long to sleep before retry
            _timeout = self._time_to_sleep_before_retry(
                e=original_exception,
                remaining_retries=num_retries,
                num_retries=num_retries,
                healthy_deployments=_healthy_deployments,
            )
            # sleeps for the length of the timeout
            await asyncio.sleep(_timeout)

            if (
                self.retry_policy is not None
                or self.model_group_retry_policy is not None
            ):
                # get num_retries from retry policy
                _retry_policy_retries = self.get_num_retries_from_retry_policy(
                    exception=original_exception, model_group=kwargs.get("model")
                )
                if _retry_policy_retries is not None:
                    num_retries = _retry_policy_retries
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
                    _healthy_deployments = await self._async_get_healthy_deployments(
                        model=kwargs.get("model"),
                    )
                    _timeout = self._time_to_sleep_before_retry(
                        e=original_exception,
                        remaining_retries=remaining_retries,
                        num_retries=num_retries,
                        healthy_deployments=_healthy_deployments,
                    )
                    await asyncio.sleep(_timeout)

            if type(original_exception) in litellm.LITELLM_EXCEPTION_TYPES:
                original_exception.max_retries = num_retries
                original_exception.num_retries = current_attempt

            raise original_exception

    def should_retry_this_error(
        self,
        error: Exception,
        healthy_deployments: Optional[List] = None,
        context_window_fallbacks: Optional[List] = None,
        content_policy_fallbacks: Optional[List] = None,
        regular_fallbacks: Optional[List] = None,
    ):
        """
        1. raise an exception for ContextWindowExceededError if context_window_fallbacks is not None
        2. raise an exception for ContentPolicyViolationError if content_policy_fallbacks is not None

        2. raise an exception for RateLimitError if
            - there are no fallbacks
            - there are no healthy deployments in the same model group
        """
        _num_healthy_deployments = 0
        if healthy_deployments is not None and isinstance(healthy_deployments, list):
            _num_healthy_deployments = len(healthy_deployments)

        ### CHECK IF RATE LIMIT / CONTEXT WINDOW ERROR / CONTENT POLICY VIOLATION ERROR w/ fallbacks available / Bad Request Error
        if (
            isinstance(error, litellm.ContextWindowExceededError)
            and context_window_fallbacks is not None
        ):
            raise error

        if (
            isinstance(error, litellm.ContentPolicyViolationError)
            and content_policy_fallbacks is not None
        ):
            raise error

        # Error we should only retry if there are other deployments
        if isinstance(error, openai.RateLimitError):
            if (
                _num_healthy_deployments <= 0  # if no healthy deployments
                and regular_fallbacks is not None  # and fallbacks available
                and len(regular_fallbacks) > 0
            ):
                raise error  # then raise the error

        if isinstance(error, openai.AuthenticationError):
            """
            - if other deployments available -> retry
            - else -> raise error
            """
            if _num_healthy_deployments <= 0:  # if no healthy deployments
                raise error  # then raise error

        return True

    def function_with_fallbacks(self, *args, **kwargs):
        """
        Try calling the function_with_retries
        If it fails after num_retries, fall back to another model group
        """
        mock_testing_fallbacks = kwargs.pop("mock_testing_fallbacks", None)
        model_group = kwargs.get("model")
        fallbacks = kwargs.get("fallbacks", self.fallbacks)
        context_window_fallbacks = kwargs.get(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        content_policy_fallbacks = kwargs.get(
            "content_policy_fallbacks", self.content_policy_fallbacks
        )
        try:
            if mock_testing_fallbacks is not None and mock_testing_fallbacks == True:
                raise Exception(
                    f"This is a mock exception for model={model_group}, to trigger a fallback. Fallbacks={fallbacks}"
                )

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
                elif (
                    isinstance(e, litellm.ContentPolicyViolationError)
                    and content_policy_fallbacks is not None
                ):
                    fallback_model_group = None

                    for (
                        item
                    ) in content_policy_fallbacks:  # [{"gpt-3.5-turbo": ["gpt-4"]}]
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
                    generic_fallback_idx: Optional[int] = None
                    for idx, item in enumerate(fallbacks):
                        if isinstance(item, dict):
                            if list(item.keys())[0] == model_group:
                                fallback_model_group = item[model_group]
                                break
                            elif list(item.keys())[0] == "*":
                                generic_fallback_idx = idx
                        elif isinstance(item, str):
                            fallback_model_group = [fallbacks.pop(idx)]
                    ## if none, check for generic fallback
                    if (
                        fallback_model_group is None
                        and generic_fallback_idx is not None
                    ):
                        fallback_model_group = fallbacks[generic_fallback_idx]["*"]

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

    def _time_to_sleep_before_retry(
        self,
        e: Exception,
        remaining_retries: int,
        num_retries: int,
        healthy_deployments: Optional[List] = None,
    ) -> Union[int, float]:
        """
        Calculate back-off, then retry

        It should instantly retry only when:
            1. there are healthy deployments in the same model group
            2. there are fallbacks for the completion call
        """
        if (
            healthy_deployments is not None
            and isinstance(healthy_deployments, list)
            and len(healthy_deployments) > 0
        ):
            return 0

        if hasattr(e, "response") and hasattr(e.response, "headers"):
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
        return timeout

    def function_with_retries(self, *args, **kwargs):
        """
        Try calling the model 3 times. Shuffle-between available deployments.
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
        content_policy_fallbacks = kwargs.pop(
            "content_policy_fallbacks", self.content_policy_fallbacks
        )

        try:
            # if the function call is successful, no exception will be raised and we'll break out of the loop
            response = original_function(*args, **kwargs)
            return response
        except Exception as e:
            current_attempt = None
            original_exception = e
            ### CHECK IF RATE LIMIT / CONTEXT WINDOW ERROR
            _healthy_deployments = self._get_healthy_deployments(
                model=kwargs.get("model"),
            )

            # raises an exception if this error should not be retries
            self.should_retry_this_error(
                error=e,
                healthy_deployments=_healthy_deployments,
                context_window_fallbacks=context_window_fallbacks,
                regular_fallbacks=fallbacks,
                content_policy_fallbacks=content_policy_fallbacks,
            )

            # decides how long to sleep before retry
            _timeout = self._time_to_sleep_before_retry(
                e=original_exception,
                remaining_retries=num_retries,
                num_retries=num_retries,
                healthy_deployments=_healthy_deployments,
            )

            ## LOGGING
            if num_retries > 0:
                kwargs = self.log_retry(kwargs=kwargs, e=original_exception)

            time.sleep(_timeout)
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
                    _healthy_deployments = self._get_healthy_deployments(
                        model=kwargs.get("model"),
                    )
                    remaining_retries = num_retries - current_attempt
                    _timeout = self._time_to_sleep_before_retry(
                        e=e,
                        remaining_retries=remaining_retries,
                        num_retries=num_retries,
                        healthy_deployments=_healthy_deployments,
                    )
                    time.sleep(_timeout)

            if type(original_exception) in litellm.LITELLM_EXCEPTION_TYPES:
                setattr(original_exception, "max_retries", num_retries)
                setattr(original_exception, "num_retries", current_attempt)

            raise original_exception

    ### HELPER FUNCTIONS

    async def deployment_callback_on_success(
        self,
        kwargs,  # kwargs to completion
        completion_response,  # response from completion
        start_time,
        end_time,  # start/end time
    ):
        """
        Track remaining tpm/rpm quota for model in model_list
        """
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
                model_info = kwargs["litellm_params"].get("model_info", {}) or {}
                id = model_info.get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                total_tokens = completion_response["usage"].get("total_tokens", 0)

                # ------------
                # Setup values
                # ------------
                dt = get_utc_datetime()
                current_minute = dt.strftime(
                    "%H-%M"
                )  # use the same timezone regardless of system clock

                tpm_key = f"global_router:{id}:tpm:{current_minute}"
                # ------------
                # Update usage
                # ------------
                # update cache

                ## TPM
                await self.cache.async_increment_cache(
                    key=tpm_key, value=total_tokens, ttl=RoutingArgs.ttl.value
                )

        except Exception as e:
            verbose_router_logger.error(
                "litellm.proxy.hooks.prompt_injection_detection.py::async_pre_call_hook(): Exception occured - {}\n{}".format(
                    str(e), traceback.format_exc()
                )
            )
            pass

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

            exception_response = getattr(exception, "response", {})
            exception_headers = getattr(exception_response, "headers", None)
            _time_to_cooldown = kwargs.get("litellm_params", {}).get(
                "cooldown_time", self.cooldown_time
            )

            if exception_headers is not None:

                _time_to_cooldown = (
                    litellm.utils._get_retry_after_from_exception_header(
                        response_headers=exception_headers
                    )
                )

                if _time_to_cooldown is None or _time_to_cooldown < 0:
                    # if the response headers did not read it -> set to default cooldown time
                    _time_to_cooldown = self.cooldown_time

            if isinstance(_model_info, dict):
                deployment_id = _model_info.get("id", None)
                self._set_cooldown_deployments(
                    exception_status=exception_status,
                    original_exception=exception,
                    deployment=deployment_id,
                    time_to_cooldown=_time_to_cooldown,
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

    def _is_cooldown_required(self, exception_status: Union[str, int]):
        """
        A function to determine if a cooldown is required based on the exception status.

        Parameters:
            exception_status (Union[str, int]): The status of the exception.

        Returns:
            bool: True if a cooldown is required, False otherwise.
        """
        try:

            if isinstance(exception_status, str):
                exception_status = int(exception_status)

            if exception_status >= 400 and exception_status < 500:
                if exception_status == 429:
                    # Cool down 429 Rate Limit Errors
                    return True

                elif exception_status == 401:
                    # Cool down 401 Auth Errors
                    return True

                elif exception_status == 408:
                    return True

                elif exception_status == 404:
                    return True

                else:
                    # Do NOT cool down all other 4XX Errors
                    return False

            else:
                # should cool down for all other errors
                return True

        except:
            # Catch all - if any exceptions default to cooling down
            return True

    def _should_raise_content_policy_error(
        self, model: str, response: ModelResponse, kwargs: dict
    ) -> bool:
        """
        Determines if a content policy error should be raised.

        Only raised if a fallback is available.

        Else, original response is returned.
        """
        if response.choices[0].finish_reason != "content_filter":
            return False

        content_policy_fallbacks = kwargs.get(
            "content_policy_fallbacks", self.content_policy_fallbacks
        )
        ### ONLY RAISE ERROR IF CP FALLBACK AVAILABLE ###
        if content_policy_fallbacks is not None:
            fallback_model_group = None
            for item in content_policy_fallbacks:  # [{"gpt-3.5-turbo": ["gpt-4"]}]
                if list(item.keys())[0] == model:
                    fallback_model_group = item[model]
                    break

            if fallback_model_group is not None:
                return True

        verbose_router_logger.info(
            "Content Policy Error occurred. No available fallbacks. Returning original response. model={}, content_policy_fallbacks={}".format(
                model, content_policy_fallbacks
            )
        )
        return False

    def _set_cooldown_deployments(
        self,
        original_exception: Any,
        exception_status: Union[str, int],
        deployment: Optional[str] = None,
        time_to_cooldown: Optional[float] = None,
    ):
        """
        Add a model to the list of models being cooled down for that minute, if it exceeds the allowed fails / minute

        or

        the exception is not one that should be immediately retried (e.g. 401)
        """
        if self.disable_cooldowns is True:
            return

        if deployment is None:
            return

        if self._is_cooldown_required(exception_status=exception_status) == False:
            return

        _allowed_fails = self.get_allowed_fails_from_policy(
            exception=original_exception,
        )

        allowed_fails = _allowed_fails or self.allowed_fails

        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        # get current fails for deployment
        # update the number of failed calls
        # if it's > allowed fails
        # cooldown deployment
        current_fails = self.failed_calls.get_cache(key=deployment) or 0
        updated_fails = current_fails + 1
        verbose_router_logger.debug(
            f"Attempting to add {deployment} to cooldown list. updated_fails: {updated_fails}; self.allowed_fails: {allowed_fails}"
        )
        cooldown_time = self.cooldown_time or 1
        if time_to_cooldown is not None:
            cooldown_time = time_to_cooldown

        if isinstance(exception_status, str):
            try:
                exception_status = int(exception_status)
            except Exception as e:
                verbose_router_logger.debug(
                    "Unable to cast exception status to int {}. Defaulting to status=500.".format(
                        exception_status
                    )
                )
                exception_status = 500
        _should_retry = litellm._should_retry(status_code=exception_status)

        if updated_fails > allowed_fails or _should_retry is False:
            # get the current cooldown list for that minute
            cooldown_key = f"{current_minute}:cooldown_models"  # group cooldown models by minute to reduce number of redis calls
            cached_value = self.cache.get_cache(
                key=cooldown_key
            )  # [(deployment_id, {last_error_str, last_error_status_code})]

            cached_value_deployment_ids = []
            if (
                cached_value is not None
                and isinstance(cached_value, list)
                and len(cached_value) > 0
                and isinstance(cached_value[0], tuple)
            ):
                cached_value_deployment_ids = [cv[0] for cv in cached_value]
            verbose_router_logger.debug(f"adding {deployment} to cooldown models")
            # update value
            if cached_value is not None and len(cached_value_deployment_ids) > 0:
                if deployment in cached_value_deployment_ids:
                    pass
                else:
                    cached_value = cached_value + [
                        (
                            deployment,
                            {
                                "Exception Received": str(original_exception),
                                "Status Code": str(exception_status),
                            },
                        )
                    ]
                    # save updated value
                    self.cache.set_cache(
                        value=cached_value, key=cooldown_key, ttl=cooldown_time
                    )
            else:
                cached_value = [
                    (
                        deployment,
                        {
                            "Exception Received": str(original_exception),
                            "Status Code": str(exception_status),
                        },
                    )
                ]
                # save updated value
                self.cache.set_cache(
                    value=cached_value, key=cooldown_key, ttl=cooldown_time
                )

            # Trigger cooldown handler
            asyncio.create_task(
                router_cooldown_handler(
                    litellm_router_instance=self,
                    deployment_id=deployment,
                    exception_status=exception_status,
                    cooldown_time=cooldown_time,
                )
            )
        else:
            self.failed_calls.set_cache(
                key=deployment, value=updated_fails, ttl=cooldown_time
            )

    async def _async_get_cooldown_deployments(self) -> List[str]:
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

        cached_value_deployment_ids = []
        if (
            cooldown_models is not None
            and isinstance(cooldown_models, list)
            and len(cooldown_models) > 0
            and isinstance(cooldown_models[0], tuple)
        ):
            cached_value_deployment_ids = [cv[0] for cv in cooldown_models]

        verbose_router_logger.debug(f"retrieve cooldown models: {cooldown_models}")
        return cached_value_deployment_ids

    async def _async_get_cooldown_deployments_with_debug_info(self) -> List[tuple]:
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

    def _get_cooldown_deployments(self) -> List[str]:
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

        cached_value_deployment_ids = []
        if (
            cooldown_models is not None
            and isinstance(cooldown_models, list)
            and len(cooldown_models) > 0
            and isinstance(cooldown_models[0], tuple)
        ):
            cached_value_deployment_ids = [cv[0] for cv in cooldown_models]

        verbose_router_logger.debug(f"retrieve cooldown models: {cooldown_models}")
        return cached_value_deployment_ids

    def _get_healthy_deployments(self, model: str):
        _all_deployments: list = []
        try:
            _, _all_deployments = self._common_checks_available_deployment(  # type: ignore
                model=model,
            )
            if type(_all_deployments) == dict:
                return []
        except:
            pass

        unhealthy_deployments = self._get_cooldown_deployments()
        healthy_deployments: list = []
        for deployment in _all_deployments:
            if deployment["model_info"]["id"] in unhealthy_deployments:
                continue
            else:
                healthy_deployments.append(deployment)

        return healthy_deployments

    async def _async_get_healthy_deployments(self, model: str):
        _all_deployments: list = []
        try:
            _, _all_deployments = self._common_checks_available_deployment(  # type: ignore
                model=model,
            )
            if type(_all_deployments) == dict:
                return []
        except:
            pass

        unhealthy_deployments = await self._async_get_cooldown_deployments()
        healthy_deployments: list = []
        for deployment in _all_deployments:
            if deployment["model_info"]["id"] in unhealthy_deployments:
                continue
            else:
                healthy_deployments.append(deployment)
        return healthy_deployments

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
                try:
                    response = await _callback.async_pre_call_check(deployment)
                except litellm.RateLimitError as e:
                    self._set_cooldown_deployments(
                        exception_status=e.status_code,
                        original_exception=e,
                        deployment=deployment["model_info"]["id"],
                        time_to_cooldown=self.cooldown_time,
                    )
                    raise e
                except Exception as e:
                    raise e

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

    def _create_deployment(
        self, model: dict, _model_name: str, _litellm_params: dict, _model_info: dict
    ):
        deployment = Deployment(
            **model,
            model_name=_model_name,
            litellm_params=LiteLLM_Params(**_litellm_params),
            model_info=_model_info,
        )

        ## REGISTER MODEL INFO IN LITELLM MODEL COST MAP
        _model_name = deployment.litellm_params.model
        if deployment.litellm_params.custom_llm_provider is not None:
            _model_name = (
                deployment.litellm_params.custom_llm_provider + "/" + _model_name
            )
        litellm.register_model(
            model_cost={
                _model_name: _model_info,
            }
        )

        deployment = self._add_deployment(deployment=deployment)

        model = deployment.to_json(exclude_none=True)

        self.model_list.append(model)

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

            if _litellm_params.get("organization", None) is not None and isinstance(
                _litellm_params["organization"], list
            ):  # Addresses https://github.com/BerriAI/litellm/issues/3949
                for org in _litellm_params["organization"]:
                    _litellm_params["organization"] = org
                    self._create_deployment(
                        model=model,
                        _model_name=_model_name,
                        _litellm_params=_litellm_params,
                        _model_info=_model_info,
                    )
            else:
                self._create_deployment(
                    model=model,
                    _model_name=_model_name,
                    _litellm_params=_litellm_params,
                    _model_info=_model_info,
                )

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

        provider_specific_deployment = re.match(
            rf"{custom_llm_provider}/\*$", deployment.model_name
        )

        # Check if user is trying to use model_name == "*"
        # this is a catch all model for their specific api key
        if deployment.model_name == "*":
            if deployment.litellm_params.model == "*":
                # user wants to pass through all requests to litellm.acompletion for unknown deployments
                self.router_general_settings.pass_through_all_models = True
            else:
                self.default_deployment = deployment.to_json(exclude_none=True)
        # Check if user is using provider specific wildcard routing
        # example model_name = "databricks/*" or model_name = "anthropic/*"
        elif provider_specific_deployment:
            if custom_llm_provider in self.provider_default_deployments:
                self.provider_default_deployments[custom_llm_provider].append(
                    deployment.to_json(exclude_none=True)
                )
            else:
                self.provider_default_deployments[custom_llm_provider] = [
                    deployment.to_json(exclude_none=True)
                ]

        # Azure GPT-Vision Enhancements, users can pass os.environ/
        data_sources = deployment.litellm_params.get("dataSources", []) or []

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
        set_client(
            litellm_router_instance=self, model=deployment.to_json(exclude_none=True)
        )

        # set region (if azure model) ## PREVIEW FEATURE ##
        if litellm.enable_preview_features == True:
            print("Auto inferring region")  # noqa
            """
            Hiding behind a feature flag
            When there is a large amount of LLM deployments this makes startup times blow up
            """
            try:
                if (
                    "azure" in deployment.litellm_params.model
                    and deployment.litellm_params.region_name is None
                ):
                    region = litellm.utils.get_model_region(
                        litellm_params=deployment.litellm_params, mode=None
                    )

                    deployment.litellm_params.region_name = region
            except Exception as e:
                verbose_router_logger.debug(
                    "Unable to get the region for azure model - {}, {}".format(
                        deployment.litellm_params.model, str(e)
                    )
                )
                pass  # [NON-BLOCKING]

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

    def upsert_deployment(self, deployment: Deployment) -> Optional[Deployment]:
        """
        Add or update deployment
        Parameters:
        - deployment: Deployment - the deployment to be added to the Router

        Returns:
        - The added/updated deployment
        """
        # check if deployment already exists
        _deployment_model_id = deployment.model_info.id or ""
        _deployment_on_router: Optional[Deployment] = self.get_deployment(
            model_id=_deployment_model_id
        )
        if _deployment_on_router is not None:
            # deployment with this model_id exists on the router
            if deployment.litellm_params == _deployment_on_router.litellm_params:
                # No need to update
                return None

            # if there is a new litellm param -> then update the deployment
            # remove the previous deployment
            removal_idx: Optional[int] = None
            for idx, model in enumerate(self.model_list):
                if model["model_info"]["id"] == deployment.model_info.id:
                    removal_idx = idx

            if removal_idx is not None:
                self.model_list.pop(removal_idx)
        else:
            # if the model_id is not in router
            self.add_deployment(deployment=deployment)
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

    def get_deployment(self, model_id: str) -> Optional[Deployment]:
        """
        Returns -> Deployment or None

        Raise Exception -> if model found in invalid format
        """
        for model in self.model_list:
            if "model_info" in model and "id" in model["model_info"]:
                if model_id == model["model_info"]["id"]:
                    if isinstance(model, dict):
                        return Deployment(**model)
                    elif isinstance(model, Deployment):
                        return model
                    else:
                        raise Exception("Model invalid format - {}".format(type(model)))
        return None

    def get_deployment_by_model_group_name(
        self, model_group_name: str
    ) -> Optional[Deployment]:
        """
        Returns -> Deployment or None

        Raise Exception -> if model found in invalid format
        """
        for model in self.model_list:
            if model["model_name"] == model_group_name:
                if isinstance(model, dict):
                    return Deployment(**model)
                elif isinstance(model, Deployment):
                    return model
                else:
                    raise Exception("Model Name invalid - {}".format(type(model)))
        return None

    def get_router_model_info(self, deployment: dict) -> ModelMapInfo:
        """
        For a given model id, return the model info (max tokens, input cost, output cost, etc.).

        Augment litellm info with additional params set in `model_info`.

        For azure models, ignore the `model:`. Only set max tokens, cost values if base_model is set.

        Returns
        - ModelInfo - If found -> typed dict with max tokens, input cost, etc.

        Raises:
        - ValueError -> If model is not mapped yet
        """
        ## GET BASE MODEL
        base_model = deployment.get("model_info", {}).get("base_model", None)
        if base_model is None:
            base_model = deployment.get("litellm_params", {}).get("base_model", None)

        model = base_model

        ## GET PROVIDER
        _model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=deployment.get("litellm_params", {}).get("model", ""),
            litellm_params=LiteLLM_Params(**deployment.get("litellm_params", {})),
        )

        ## SET MODEL TO 'model=' - if base_model is None + not azure
        if custom_llm_provider == "azure" and base_model is None:
            verbose_router_logger.error(
                "Could not identify azure model. Set azure 'base_model' for accurate max tokens, cost tracking, etc.- https://docs.litellm.ai/docs/proxy/cost_tracking#spend-tracking-for-azure-openai-models"
            )
        elif custom_llm_provider != "azure":
            model = _model

        ## GET LITELLM MODEL INFO - raises exception, if model is not mapped
        model_info = litellm.get_model_info(model=model)

        ## CHECK USER SET MODEL INFO
        user_model_info = deployment.get("model_info", {})

        model_info.update(user_model_info)

        return model_info

    def get_model_info(self, id: str) -> Optional[dict]:
        """
        For a given model id, return the model info

        Returns
        - dict: the model in list with 'model_name', 'litellm_params', Optional['model_info']
        - None: could not find deployment in list
        """
        for model in self.model_list:
            if "model_info" in model and "id" in model["model_info"]:
                if id == model["model_info"]["id"]:
                    return model
        return None

    def get_model_group_info(self, model_group: str) -> Optional[ModelGroupInfo]:
        """
        For a given model group name, return the combined model info

        Returns:
        - ModelGroupInfo if able to construct a model group
        - None if error constructing model group info
        """

        model_group_info: Optional[ModelGroupInfo] = None

        total_tpm: Optional[int] = None
        total_rpm: Optional[int] = None

        for model in self.model_list:
            if "model_name" in model and model["model_name"] == model_group:
                # model in model group found #
                litellm_params = LiteLLM_Params(**model["litellm_params"])
                # get model tpm
                _deployment_tpm: Optional[int] = None
                if _deployment_tpm is None:
                    _deployment_tpm = model.get("tpm", None)
                if _deployment_tpm is None:
                    _deployment_tpm = model.get("litellm_params", {}).get("tpm", None)
                if _deployment_tpm is None:
                    _deployment_tpm = model.get("model_info", {}).get("tpm", None)

                if _deployment_tpm is not None:
                    if total_tpm is None:
                        total_tpm = 0
                    total_tpm += _deployment_tpm  # type: ignore
                # get model rpm
                _deployment_rpm: Optional[int] = None
                if _deployment_rpm is None:
                    _deployment_rpm = model.get("rpm", None)
                if _deployment_rpm is None:
                    _deployment_rpm = model.get("litellm_params", {}).get("rpm", None)
                if _deployment_rpm is None:
                    _deployment_rpm = model.get("model_info", {}).get("rpm", None)

                if _deployment_rpm is not None:
                    if total_rpm is None:
                        total_rpm = 0
                    total_rpm += _deployment_rpm  # type: ignore
                # get model info
                try:
                    model_info = litellm.get_model_info(model=litellm_params.model)
                except Exception:
                    model_info = None
                # get llm provider
                model, llm_provider = "", ""
                try:
                    model, llm_provider, _, _ = litellm.get_llm_provider(
                        model=litellm_params.model,
                        custom_llm_provider=litellm_params.custom_llm_provider,
                    )
                except litellm.exceptions.BadRequestError as e:
                    verbose_router_logger.error(
                        "litellm.router.py::get_model_group_info() - {}".format(str(e))
                    )

                if model_info is None:
                    supported_openai_params = litellm.get_supported_openai_params(
                        model=model, custom_llm_provider=llm_provider
                    )
                    if supported_openai_params is None:
                        supported_openai_params = []
                    model_info = ModelMapInfo(
                        key=model_group,
                        max_tokens=None,
                        max_input_tokens=None,
                        max_output_tokens=None,
                        input_cost_per_token=0,
                        output_cost_per_token=0,
                        litellm_provider=llm_provider,
                        mode="chat",
                        supported_openai_params=supported_openai_params,
                        supports_system_messages=None,
                    )

                if model_group_info is None:
                    model_group_info = ModelGroupInfo(
                        model_group=model_group, providers=[llm_provider], **model_info  # type: ignore
                    )
                else:
                    # if max_input_tokens > curr
                    # if max_output_tokens > curr
                    # if input_cost_per_token > curr
                    # if output_cost_per_token > curr
                    # supports_parallel_function_calling == True
                    # supports_vision == True
                    # supports_function_calling == True
                    if llm_provider not in model_group_info.providers:
                        model_group_info.providers.append(llm_provider)
                    if (
                        model_info.get("max_input_tokens", None) is not None
                        and model_info["max_input_tokens"] is not None
                        and (
                            model_group_info.max_input_tokens is None
                            or model_info["max_input_tokens"]
                            > model_group_info.max_input_tokens
                        )
                    ):
                        model_group_info.max_input_tokens = model_info[
                            "max_input_tokens"
                        ]
                    if (
                        model_info.get("max_output_tokens", None) is not None
                        and model_info["max_output_tokens"] is not None
                        and (
                            model_group_info.max_output_tokens is None
                            or model_info["max_output_tokens"]
                            > model_group_info.max_output_tokens
                        )
                    ):
                        model_group_info.max_output_tokens = model_info[
                            "max_output_tokens"
                        ]
                    if model_info.get("input_cost_per_token", None) is not None and (
                        model_group_info.input_cost_per_token is None
                        or model_info["input_cost_per_token"]
                        > model_group_info.input_cost_per_token
                    ):
                        model_group_info.input_cost_per_token = model_info[
                            "input_cost_per_token"
                        ]
                    if model_info.get("output_cost_per_token", None) is not None and (
                        model_group_info.output_cost_per_token is None
                        or model_info["output_cost_per_token"]
                        > model_group_info.output_cost_per_token
                    ):
                        model_group_info.output_cost_per_token = model_info[
                            "output_cost_per_token"
                        ]
                    if (
                        model_info.get("supports_parallel_function_calling", None)
                        is not None
                        and model_info["supports_parallel_function_calling"] is True  # type: ignore
                    ):
                        model_group_info.supports_parallel_function_calling = True
                    if (
                        model_info.get("supports_vision", None) is not None
                        and model_info["supports_vision"] is True  # type: ignore
                    ):
                        model_group_info.supports_vision = True
                    if (
                        model_info.get("supports_function_calling", None) is not None
                        and model_info["supports_function_calling"] is True  # type: ignore
                    ):
                        model_group_info.supports_function_calling = True
                    if (
                        model_info.get("supported_openai_params", None) is not None
                        and model_info["supported_openai_params"] is not None
                    ):
                        model_group_info.supported_openai_params = model_info[
                            "supported_openai_params"
                        ]

        ## UPDATE WITH TOTAL TPM/RPM FOR MODEL GROUP
        if total_tpm is not None and model_group_info is not None:
            model_group_info.tpm = total_tpm

        if total_rpm is not None and model_group_info is not None:
            model_group_info.rpm = total_rpm

        return model_group_info

    async def get_model_group_usage(
        self, model_group: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns remaining tpm/rpm quota for model group

        Returns:
        - usage: Tuple[tpm, rpm]
        """
        dt = get_utc_datetime()
        current_minute = dt.strftime(
            "%H-%M"
        )  # use the same timezone regardless of system clock
        tpm_keys: List[str] = []
        rpm_keys: List[str] = []
        for model in self.model_list:
            if "model_name" in model and model["model_name"] == model_group:
                tpm_keys.append(
                    f"global_router:{model['model_info']['id']}:tpm:{current_minute}"
                )
                rpm_keys.append(
                    f"global_router:{model['model_info']['id']}:rpm:{current_minute}"
                )
        combined_tpm_rpm_keys = tpm_keys + rpm_keys

        combined_tpm_rpm_values = await self.cache.async_batch_get_cache(
            keys=combined_tpm_rpm_keys
        )

        if combined_tpm_rpm_values is None:
            return None, None

        tpm_usage_list: Optional[List] = combined_tpm_rpm_values[: len(tpm_keys)]
        rpm_usage_list: Optional[List] = combined_tpm_rpm_values[len(tpm_keys) :]

        ## TPM
        tpm_usage: Optional[int] = None
        if tpm_usage_list is not None:
            for t in tpm_usage_list:
                if isinstance(t, int):
                    if tpm_usage is None:
                        tpm_usage = 0
                    tpm_usage += t
        ## RPM
        rpm_usage: Optional[int] = None
        if rpm_usage_list is not None:
            for t in rpm_usage_list:
                if isinstance(t, int):
                    if rpm_usage is None:
                        rpm_usage = 0
                    rpm_usage += t
        return tpm_usage, rpm_usage

    def get_model_ids(self) -> List[str]:
        """
        Returns list of model id's.
        """
        ids = []
        for model in self.model_list:
            if "model_info" in model and "id" in model["model_info"]:
                id = model["model_info"]["id"]
                ids.append(id)
        return ids

    def get_model_names(self) -> List[str]:
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
            "fallbacks",
            "context_window_fallbacks",
            "model_group_retry_policy",
        ]

        for var in vars_to_include:
            if var in _all_vars:
                _settings_to_return[var] = _all_vars[var]
            if (
                var == "routing_strategy_args"
                and self.routing_strategy == "latency-based-routing"
            ):
                _settings_to_return[var] = self.lowestlatency_logger.routing_args.json()
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
            "fallbacks",
            "context_window_fallbacks",
            "model_group_retry_policy",
        ]

        _int_settings = [
            "timeout",
            "num_retries",
            "retry_after",
            "allowed_fails",
            "cooldown_time",
        ]

        _existing_router_settings = self.get_settings()
        for var in kwargs:
            if var in _allowed_settings:
                if var in _int_settings:
                    _casted_value = int(kwargs[var])
                    setattr(self, var, _casted_value)
                else:
                    # only run routing strategy init if it has changed
                    if (
                        var == "routing_strategy"
                        and _existing_router_settings["routing_strategy"] != kwargs[var]
                    ):
                        self.routing_strategy_init(
                            routing_strategy=kwargs[var],
                            routing_strategy_args=kwargs.get(
                                "routing_strategy_args", {}
                            ),
                        )
                    setattr(self, var, kwargs[var])
            else:
                verbose_router_logger.debug("Setting {} is not allowed".format(var))
        verbose_router_logger.debug(f"Updated Router settings: {self.get_settings()}")

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
                    set_client(litellm_router_instance=self, model=deployment)
                    client = self.cache.get_cache(key=cache_key, local_only=True)
                return client
            else:
                cache_key = f"{model_id}_async_client"
                client = self.cache.get_cache(key=cache_key, local_only=True)
                if client is None:
                    """
                    Re-initialize the client
                    """
                    set_client(litellm_router_instance=self, model=deployment)
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
                    set_client(litellm_router_instance=self, model=deployment)
                    client = self.cache.get_cache(key=cache_key)
                return client
            else:
                cache_key = f"{model_id}_client"
                client = self.cache.get_cache(key=cache_key)
                if client is None:
                    """
                    Re-initialize the client
                    """
                    set_client(litellm_router_instance=self, model=deployment)
                    client = self.cache.get_cache(key=cache_key)
                return client

    def _pre_call_checks(
        self,
        model: str,
        healthy_deployments: List,
        messages: List[Dict[str, str]],
        request_kwargs: Optional[dict] = None,
    ):
        """
        Filter out model in model group, if:

        - model context window < message length. For azure openai models, requires 'base_model' is set. - https://docs.litellm.ai/docs/proxy/cost_tracking#spend-tracking-for-azure-openai-models
        - filter models above rpm limits
        - if region given, filter out models not in that region / unknown region
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
            verbose_router_logger.error(
                "litellm.router.py::_pre_call_checks: failed to count tokens. Returning initial list of deployments. Got - {}".format(
                    str(e)
                )
            )
            return _returned_deployments

        _context_window_error = False
        _potential_error_str = ""
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
                model_info = self.get_router_model_info(deployment=deployment)

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
                        _potential_error_str += (
                            "Model={}, Max Input Tokens={}, Got={}".format(
                                model, model_info["max_input_tokens"], input_tokens
                            )
                        )
                        continue
            except Exception as e:
                verbose_router_logger.error("An error occurs - {}".format(str(e)))

            _litellm_params = deployment.get("litellm_params", {})
            model_id = deployment.get("model_info", {}).get("id", "")
            ## RPM CHECK ##
            ### get local router cache ###
            current_request_cache_local = (
                self.cache.get_cache(key=model_id, local_only=True) or 0
            )
            ### get usage based cache ###
            if (
                isinstance(model_group_cache, dict)
                and self.routing_strategy != "usage-based-routing-v2"
            ):
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

            ## REGION CHECK ##
            if (
                request_kwargs is not None
                and request_kwargs.get("allowed_model_region") is not None
                and request_kwargs["allowed_model_region"] == "eu"
            ):
                if _litellm_params.get("region_name") is not None and isinstance(
                    _litellm_params["region_name"], str
                ):
                    # check if in allowed_model_region
                    if (
                        _is_region_eu(litellm_params=LiteLLM_Params(**_litellm_params))
                        == False
                    ):
                        invalid_model_indices.append(idx)
                        continue
                else:
                    verbose_router_logger.debug(
                        "Filtering out model - {}, as model_region=None, and allowed_model_region={}".format(
                            model_id, request_kwargs.get("allowed_model_region")
                        )
                    )
                    # filter out since region unknown, and user wants to filter for specific region
                    invalid_model_indices.append(idx)
                    continue

            ## INVALID PARAMS ## -> catch 'gpt-3.5-turbo-16k' not supporting 'response_format' param
            if request_kwargs is not None and litellm.drop_params == False:
                # get supported params
                model, custom_llm_provider, _, _ = litellm.get_llm_provider(
                    model=model, litellm_params=LiteLLM_Params(**_litellm_params)
                )

                supported_openai_params = litellm.get_supported_openai_params(
                    model=model, custom_llm_provider=custom_llm_provider
                )

                if supported_openai_params is None:
                    continue
                else:
                    # check the non-default openai params in request kwargs
                    non_default_params = litellm.utils.get_non_default_params(
                        passed_params=request_kwargs
                    )
                    special_params = ["response_format"]
                    # check if all params are supported
                    for k, v in non_default_params.items():
                        if k not in supported_openai_params and k in special_params:
                            # if not -> invalid model
                            verbose_router_logger.debug(
                                f"INVALID MODEL INDEX @ REQUEST KWARG FILTERING, k={k}"
                            )
                            invalid_model_indices.append(idx)

        if len(invalid_model_indices) == len(_returned_deployments):
            """
            - no healthy deployments available b/c context window checks or rate limit error

            - First check for rate limit errors (if this is true, it means the model passed the context window check but failed the rate limit check)
            """

            if _rate_limit_error == True:  # allow generic fallback logic to take place
                raise ValueError(
                    f"{RouterErrors.no_deployments_available.value}, Try again in {self.cooldown_time} seconds. Passed model={model}. Try again in {self.cooldown_time} seconds."
                )
            elif _context_window_error is True:
                raise litellm.ContextWindowExceededError(
                    message="litellm._pre_call_checks: Context Window exceeded for given call. No models have context window large enough for this call.\n{}".format(
                        _potential_error_str
                    ),
                    model=model,
                    llm_provider="",
                )
        if len(invalid_model_indices) > 0:
            for idx in reversed(invalid_model_indices):
                _returned_deployments.pop(idx)

        ## ORDER FILTERING ## -> if user set 'order' in deployments, return deployments with lowest order (e.g. order=1 > order=2)
        if len(_returned_deployments) > 0:
            _returned_deployments = litellm.utils._get_order_filtered_deployments(
                _returned_deployments
            )

        return _returned_deployments

    def _common_checks_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Tuple[str, Union[list, dict]]:
        """
        Common checks for 'get_available_deployment' across sync + async call.

        If 'healthy_deployments' returned is None, this means the user chose a specific deployment

        Returns
        - Dict, if specific model chosen
        - List, if multiple models chosen
        """
        # check if aliases set on litellm model alias map
        if specific_deployment == True:
            # users can also specify a specific deployment name. At this point we should check if they are just trying to call a specific deployment
            for deployment in self.model_list:
                deployment_model = deployment.get("litellm_params").get("model")
                if deployment_model == model:
                    # User Passed a specific deployment name on their config.yaml, example azure/chat-gpt-v-2
                    # return the first deployment where the `model` matches the specificed deployment name
                    return deployment_model, deployment
            raise ValueError(
                f"LiteLLM Router: Trying to call specific deployment, but Model:{model} does not exist in Model List: {self.model_list}"
            )
        elif model in self.get_model_ids():
            deployment = self.get_model_info(id=model)
            if deployment is not None:
                deployment_model = deployment.get("litellm_params", {}).get("model")
                return deployment_model, deployment
            raise ValueError(
                f"LiteLLM Router: Trying to call specific deployment, but Model ID :{model} does not exist in \
                    Model ID List: {self.get_model_ids}"
            )

        if model in self.model_group_alias:
            verbose_router_logger.debug(
                f"Using a model alias. Got Request for {model}, sending requests to {self.model_group_alias.get(model)}"
            )
            model = self.model_group_alias[model]

        if model not in self.model_names:
            # check if provider/ specific wildcard routing
            try:
                (
                    _,
                    custom_llm_provider,
                    _,
                    _,
                ) = litellm.get_llm_provider(model=model)
                # check if custom_llm_provider
                if custom_llm_provider in self.provider_default_deployments:
                    _provider_deployments = self.provider_default_deployments[
                        custom_llm_provider
                    ]
                    provider_deployments = []
                    for deployment in _provider_deployments:
                        dep = copy.deepcopy(deployment)
                        dep["litellm_params"]["model"] = model
                        provider_deployments.append(dep)
                    return model, provider_deployments
            except:
                # get_llm_provider raises exception when provider is unknown
                pass

            # check if default deployment is set
            if self.default_deployment is not None:
                updated_deployment = copy.deepcopy(
                    self.default_deployment
                )  # self.default_deployment
                updated_deployment["litellm_params"]["model"] = model
                return model, updated_deployment

        ## get healthy deployments
        ### get all deployments
        healthy_deployments = [m for m in self.model_list if m["model_name"] == model]
        if len(healthy_deployments) == 0:
            # check if the user sent in a deployment name instead
            healthy_deployments = [
                m for m in self.model_list if m["litellm_params"]["model"] == model
            ]

        litellm.print_verbose(f"initial list of deployments: {healthy_deployments}")

        if len(healthy_deployments) == 0:
            raise ValueError(
                f"No healthy deployment available, passed model={model}. Try again in {self.cooldown_time} seconds"
            )
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
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Async implementation of 'get_available_deployments'.

        Allows all cache calls to be made async => 10x perf impact (8rps -> 100 rps).
        """
        if (
            self.routing_strategy != "usage-based-routing-v2"
            and self.routing_strategy != "simple-shuffle"
            and self.routing_strategy != "cost-based-routing"
        ):  # prevent regressions for other routing strategies, that don't have async get available deployments implemented.
            return self.get_available_deployment(
                model=model,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
                request_kwargs=request_kwargs,
            )
        try:
            model, healthy_deployments = self._common_checks_available_deployment(
                model=model,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
            )  # type: ignore

            if isinstance(healthy_deployments, dict):
                return healthy_deployments

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
            _allowed_model_region = (
                request_kwargs.get("allowed_model_region")
                if request_kwargs is not None
                else None
            )

            if self.enable_pre_call_checks and messages is not None:
                healthy_deployments = self._pre_call_checks(
                    model=model,
                    healthy_deployments=healthy_deployments,
                    messages=messages,
                    request_kwargs=request_kwargs,
                )

            # check if user wants to do tag based routing
            healthy_deployments = await get_deployments_for_tag(
                llm_router_instance=self,
                request_kwargs=request_kwargs,
                healthy_deployments=healthy_deployments,
            )

            if len(healthy_deployments) == 0:
                if _allowed_model_region is None:
                    _allowed_model_region = "n/a"
                raise ValueError(
                    f"{RouterErrors.no_deployments_available.value}, Try again in {self.cooldown_time} seconds. Passed model={model}. pre-call-checks={self.enable_pre_call_checks}, allowed_model_region={_allowed_model_region}, cooldown_list={await self._async_get_cooldown_deployments_with_debug_info()}"
                )

            if (
                self.routing_strategy == "usage-based-routing-v2"
                and self.lowesttpm_logger_v2 is not None
            ):
                deployment = (
                    await self.lowesttpm_logger_v2.async_get_available_deployments(
                        model_group=model,
                        healthy_deployments=healthy_deployments,  # type: ignore
                        messages=messages,
                        input=input,
                    )
                )
            if (
                self.routing_strategy == "cost-based-routing"
                and self.lowestcost_logger is not None
            ):
                deployment = (
                    await self.lowestcost_logger.async_get_available_deployments(
                        model_group=model,
                        healthy_deployments=healthy_deployments,  # type: ignore
                        messages=messages,
                        input=input,
                    )
                )
            elif self.routing_strategy == "simple-shuffle":
                # if users pass rpm or tpm, we do a random weighted pick - based on rpm/tpm
                ############## Check if we can do a RPM/TPM based weighted pick #################
                rpm = healthy_deployments[0].get("litellm_params").get("rpm", None)
                if rpm is not None:
                    # use weight-random pick if rpms provided
                    rpms = [
                        m["litellm_params"].get("rpm", 0) for m in healthy_deployments
                    ]
                    verbose_router_logger.debug(f"\nrpms {rpms}")
                    total_rpm = sum(rpms)
                    weights = [rpm / total_rpm for rpm in rpms]
                    verbose_router_logger.debug(f"\n weights {weights}")
                    # Perform weighted random pick
                    selected_index = random.choices(range(len(rpms)), weights=weights)[
                        0
                    ]
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
                    tpms = [
                        m["litellm_params"].get("tpm", 0) for m in healthy_deployments
                    ]
                    verbose_router_logger.debug(f"\ntpms {tpms}")
                    total_tpm = sum(tpms)
                    weights = [tpm / total_tpm for tpm in tpms]
                    verbose_router_logger.debug(f"\n weights {weights}")
                    # Perform weighted random pick
                    selected_index = random.choices(range(len(tpms)), weights=weights)[
                        0
                    ]
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
                    f"{RouterErrors.no_deployments_available.value}, Try again in {self.cooldown_time} seconds. Passed model={model}"
                )
            verbose_router_logger.info(
                f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment)} for model: {model}"
            )

            return deployment
        except Exception as e:
            traceback_exception = traceback.format_exc()
            # if router rejects call -> log to langfuse/otel/etc.
            if request_kwargs is not None:
                logging_obj = request_kwargs.get("litellm_logging_obj", None)
                if logging_obj is not None:
                    ## LOGGING
                    threading.Thread(
                        target=logging_obj.failure_handler,
                        args=(e, traceback_exception),
                    ).start()  # log response
                    # Handle any exceptions that might occur during streaming
                    asyncio.create_task(
                        logging_obj.async_failure_handler(e, traceback_exception)  # type: ignore
                    )
            raise e

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
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

        if isinstance(healthy_deployments, dict):
            return healthy_deployments

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
                model_group=model, healthy_deployments=healthy_deployments  # type: ignore
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
                model_group=model,
                healthy_deployments=healthy_deployments,  # type: ignore
                request_kwargs=request_kwargs,
            )
        elif (
            self.routing_strategy == "usage-based-routing"
            and self.lowesttpm_logger is not None
        ):
            deployment = self.lowesttpm_logger.get_available_deployments(
                model_group=model,
                healthy_deployments=healthy_deployments,  # type: ignore
                messages=messages,
                input=input,
            )
        elif (
            self.routing_strategy == "usage-based-routing-v2"
            and self.lowesttpm_logger_v2 is not None
        ):
            deployment = self.lowesttpm_logger_v2.get_available_deployments(
                model_group=model,
                healthy_deployments=healthy_deployments,  # type: ignore
                messages=messages,
                input=input,
            )
        if deployment is None:
            verbose_router_logger.info(
                f"get_available_deployment for model: {model}, No deployment available"
            )
            raise ValueError(
                f"{RouterErrors.no_deployments_available.value}, Try again in {self.cooldown_time} seconds. Passed model={model}"
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

    def get_num_retries_from_retry_policy(
        self, exception: Exception, model_group: Optional[str] = None
    ):
        """
        BadRequestErrorRetries: Optional[int] = None
        AuthenticationErrorRetries: Optional[int] = None
        TimeoutErrorRetries: Optional[int] = None
        RateLimitErrorRetries: Optional[int] = None
        ContentPolicyViolationErrorRetries: Optional[int] = None
        """
        # if we can find the exception then in the retry policy -> return the number of retries
        retry_policy = self.retry_policy
        if (
            self.model_group_retry_policy is not None
            and model_group is not None
            and model_group in self.model_group_retry_policy
        ):
            retry_policy = self.model_group_retry_policy.get(model_group, None)

        if retry_policy is None:
            return None
        if isinstance(retry_policy, dict):
            retry_policy = RetryPolicy(**retry_policy)
        if (
            isinstance(exception, litellm.BadRequestError)
            and retry_policy.BadRequestErrorRetries is not None
        ):
            return retry_policy.BadRequestErrorRetries
        if (
            isinstance(exception, litellm.AuthenticationError)
            and retry_policy.AuthenticationErrorRetries is not None
        ):
            return retry_policy.AuthenticationErrorRetries
        if (
            isinstance(exception, litellm.Timeout)
            and retry_policy.TimeoutErrorRetries is not None
        ):
            return retry_policy.TimeoutErrorRetries
        if (
            isinstance(exception, litellm.RateLimitError)
            and retry_policy.RateLimitErrorRetries is not None
        ):
            return retry_policy.RateLimitErrorRetries
        if (
            isinstance(exception, litellm.ContentPolicyViolationError)
            and retry_policy.ContentPolicyViolationErrorRetries is not None
        ):
            return retry_policy.ContentPolicyViolationErrorRetries

    def get_allowed_fails_from_policy(self, exception: Exception):
        """
        BadRequestErrorRetries: Optional[int] = None
        AuthenticationErrorRetries: Optional[int] = None
        TimeoutErrorRetries: Optional[int] = None
        RateLimitErrorRetries: Optional[int] = None
        ContentPolicyViolationErrorRetries: Optional[int] = None
        """
        # if we can find the exception then in the retry policy -> return the number of retries
        allowed_fails_policy: Optional[AllowedFailsPolicy] = self.allowed_fails_policy

        if allowed_fails_policy is None:
            return None

        if (
            isinstance(exception, litellm.BadRequestError)
            and allowed_fails_policy.BadRequestErrorAllowedFails is not None
        ):
            return allowed_fails_policy.BadRequestErrorAllowedFails
        if (
            isinstance(exception, litellm.AuthenticationError)
            and allowed_fails_policy.AuthenticationErrorAllowedFails is not None
        ):
            return allowed_fails_policy.AuthenticationErrorAllowedFails
        if (
            isinstance(exception, litellm.Timeout)
            and allowed_fails_policy.TimeoutErrorAllowedFails is not None
        ):
            return allowed_fails_policy.TimeoutErrorAllowedFails
        if (
            isinstance(exception, litellm.RateLimitError)
            and allowed_fails_policy.RateLimitErrorAllowedFails is not None
        ):
            return allowed_fails_policy.RateLimitErrorAllowedFails
        if (
            isinstance(exception, litellm.ContentPolicyViolationError)
            and allowed_fails_policy.ContentPolicyViolationErrorAllowedFails is not None
        ):
            return allowed_fails_policy.ContentPolicyViolationErrorAllowedFails

    def _initialize_alerting(self):
        from litellm.integrations.slack_alerting import SlackAlerting

        router_alerting_config: AlertingConfig = self.alerting_config

        _slack_alerting_logger = SlackAlerting(
            alerting_threshold=router_alerting_config.alerting_threshold,
            alerting=["slack"],
            default_webhook_url=router_alerting_config.webhook_url,
        )

        self.slack_alerting_logger = _slack_alerting_logger

        litellm.callbacks.append(_slack_alerting_logger)
        litellm.success_callback.append(
            _slack_alerting_logger.response_taking_too_long_callback
        )
        print("\033[94m\nInitialized Alerting for litellm.Router\033[0m\n")  # noqa

    def set_custom_routing_strategy(
        self, CustomRoutingStrategy: CustomRoutingStrategyBase
    ):
        """
        Sets get_available_deployment and async_get_available_deployment on an instanced of litellm.Router

        Use this to set your custom routing strategy

        Args:
            CustomRoutingStrategy: litellm.router.CustomRoutingStrategyBase
        """

        setattr(
            self,
            "get_available_deployment",
            CustomRoutingStrategy.get_available_deployment,
        )
        setattr(
            self,
            "async_get_available_deployment",
            CustomRoutingStrategy.async_get_available_deployment,
        )

    def flush_cache(self):
        litellm.cache = None
        self.cache.flush_cache()

    def reset(self):
        ## clean up on close
        litellm.success_callback = []
        litellm._async_success_callback = []
        litellm.failure_callback = []
        litellm._async_failure_callback = []
        self.retry_policy = None
        self.flush_cache()

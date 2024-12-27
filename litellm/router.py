# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you ! We ❤️ you! - Krrish & Ishaan

import asyncio
import copy
import enum
import hashlib
import inspect
import json
import logging
import threading
import time
import traceback
import uuid
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing_extensions import overload

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.exception_mapping_utils
from litellm import get_secret_str
from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache, InMemoryCache, RedisCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.router_strategy.simple_shuffle import simple_shuffle
from litellm.router_strategy.tag_based_routing import get_deployments_for_tag
from litellm.router_utils.batch_utils import (
    _get_router_metadata_variable_name,
    replace_model_in_jsonl,
)
from litellm.router_utils.client_initalization_utils import InitalizeOpenAISDKClient
from litellm.router_utils.cooldown_cache import CooldownCache
from litellm.router_utils.cooldown_handlers import (
    DEFAULT_COOLDOWN_TIME_SECONDS,
    _async_get_cooldown_deployments,
    _async_get_cooldown_deployments_with_debug_info,
    _get_cooldown_deployments,
    _set_cooldown_deployments,
)
from litellm.router_utils.fallback_event_handlers import (
    _check_non_standard_fallback_format,
    get_fallback_model_group,
    run_async_fallback,
)
from litellm.router_utils.get_retry_from_policy import (
    get_num_retries_from_retry_policy as _get_num_retries_from_retry_policy,
)
from litellm.router_utils.handle_error import (
    async_raise_no_deployment_exception,
    send_llm_exception_alert,
)
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
    PromptCachingDeploymentCheck,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    increment_deployment_failures_for_current_minute,
    increment_deployment_successes_for_current_minute,
)
from litellm.scheduler import FlowItem, Scheduler
from litellm.types.llms.openai import AllMessageValues, Batch, FileObject, FileTypes
from litellm.types.router import (
    CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS,
    VALID_LITELLM_ENVIRONMENTS,
    AlertingConfig,
    AllowedFailsPolicy,
    AssistantsTypedDict,
    CustomRoutingStrategyBase,
    Deployment,
    DeploymentTypedDict,
    LiteLLM_Params,
    ModelGroupInfo,
    OptionalPreCallChecks,
    RetryPolicy,
    RouterCacheEnum,
    RouterGeneralSettings,
    RouterModelGroupAliasItem,
    RouterRateLimitError,
    RouterRateLimitErrorBasic,
    RoutingStrategy,
)
from litellm.types.services import ServiceTypes
from litellm.types.utils import GenericBudgetConfigType
from litellm.types.utils import ModelInfo as ModelMapInfo
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import (
    CustomStreamWrapper,
    EmbeddingResponse,
    ModelResponse,
    get_llm_provider,
    get_secret,
    get_utc_datetime,
    is_region_allowed,
)

from .router_utils.pattern_match_deployments import PatternMatchRouter

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


class RoutingArgs(enum.Enum):
    ttl = 60  # 1min (RPM/TPM expire key)


class Router:
    model_names: List = []
    cache_responses: Optional[bool] = False
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour
    tenacity = None
    leastbusy_logger: Optional[LeastBusyLoggingHandler] = None
    lowesttpm_logger: Optional[LowestTPMLoggingHandler] = None

    def __init__(  # noqa: PLR0915
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
        default_priority: Optional[int] = None,
        ## RELIABILITY ##
        num_retries: Optional[int] = None,
        max_fallbacks: Optional[
            int
        ] = None,  # max fallbacks to try before exiting the call. Defaults to 5.
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
        model_group_alias: Optional[
            Dict[str, Union[str, RouterModelGroupAliasItem]]
        ] = {},
        enable_pre_call_checks: bool = False,
        enable_tag_filtering: bool = False,
        retry_after: int = 0,  # min time to wait before retrying a failed request
        retry_policy: Optional[
            Union[RetryPolicy, dict]
        ] = None,  # set custom retries for different exceptions
        model_group_retry_policy: Dict[
            str, RetryPolicy
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
        optional_pre_call_checks: Optional[OptionalPreCallChecks] = None,
        routing_strategy_args: dict = {},  # just for latency-based
        provider_budget_config: Optional[GenericBudgetConfigType] = None,
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
            default_priority: (Optional[int]): the default priority for a request. Only for '.scheduler_acompletion()'. Default is None.
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
            provider_budget_config (ProviderBudgetConfig): Provider budget configuration. Use this to set llm_provider budget limits. example $100/day to OpenAI, $100/day to Azure, etc. Defaults to None.
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

        from litellm._service_logger import ServiceLogging

        self.set_verbose = set_verbose
        self.debug_level = debug_level
        self.enable_pre_call_checks = enable_pre_call_checks
        self.enable_tag_filtering = enable_tag_filtering
        if self.set_verbose is True:
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
        cache_config: Dict[str, Any] = {}

        self.client_ttl = client_ttl
        if redis_url is not None or (redis_host is not None and redis_port is not None):
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
        self.default_priority = default_priority
        self.default_deployment = None  # use this to track the users default deployment, when they want to use model = *
        self.default_max_parallel_requests = default_max_parallel_requests
        self.provider_default_deployment_ids: List[str] = []
        self.pattern_router = PatternMatchRouter()

        if model_list is not None:
            model_list = copy.deepcopy(model_list)
            self.set_model_list(model_list)
            self.healthy_deployments: List = self.model_list  # type: ignore
            for m in model_list:
                if "model" in m["litellm_params"]:
                    self.deployment_latency_map[m["litellm_params"]["model"]] = 0
        else:
            self.model_list: List = (
                []
            )  # initialize an empty list - to allow _add_deployment and delete_deployment to work

        if allowed_fails is not None:
            self.allowed_fails = allowed_fails
        else:
            self.allowed_fails = litellm.allowed_fails
        self.cooldown_time = cooldown_time or DEFAULT_COOLDOWN_TIME_SECONDS
        self.cooldown_cache = CooldownCache(
            cache=self.cache, default_cooldown_time=self.cooldown_time
        )
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

        if max_fallbacks is not None:
            self.max_fallbacks = max_fallbacks
        elif litellm.max_fallbacks is not None:
            self.max_fallbacks = litellm.max_fallbacks
        else:
            self.max_fallbacks = litellm.ROUTER_MAX_FALLBACKS

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
        self.model_group_alias: Dict[str, Union[str, RouterModelGroupAliasItem]] = (
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
        self.access_groups = None
        ## USAGE TRACKING ##
        if isinstance(litellm._async_success_callback, list):
            litellm._async_success_callback.append(self.deployment_callback_on_success)
        else:
            litellm._async_success_callback.append(self.deployment_callback_on_success)
        if isinstance(litellm.success_callback, list):
            litellm.success_callback.append(self.sync_deployment_callback_on_success)
        else:
            litellm.success_callback = [self.sync_deployment_callback_on_success]
        if isinstance(litellm._async_failure_callback, list):
            litellm._async_failure_callback.append(
                self.async_deployment_callback_on_failure
            )
        else:
            litellm._async_failure_callback = [
                self.async_deployment_callback_on_failure
            ]
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
        self.service_logger_obj = ServiceLogging()
        self.routing_strategy_args = routing_strategy_args
        self.provider_budget_config = provider_budget_config
        self.router_budget_logger: Optional[RouterBudgetLimiting] = None
        if RouterBudgetLimiting.should_init_router_budget_limiter(
            model_list=model_list, provider_budget_config=self.provider_budget_config
        ):
            if optional_pre_call_checks is not None:
                optional_pre_call_checks.append("router_budget_limiting")
            else:
                optional_pre_call_checks = ["router_budget_limiting"]
        self.retry_policy: Optional[RetryPolicy] = None
        if retry_policy is not None:
            if isinstance(retry_policy, dict):
                self.retry_policy = RetryPolicy(**retry_policy)
            elif isinstance(retry_policy, RetryPolicy):
                self.retry_policy = retry_policy
            verbose_router_logger.info(
                "\033[32mRouter Custom Retry Policy Set:\n{}\033[0m".format(
                    self.retry_policy.model_dump(exclude_none=True)
                )
            )

        self.model_group_retry_policy: Optional[Dict[str, RetryPolicy]] = (
            model_group_retry_policy
        )

        self.allowed_fails_policy: Optional[AllowedFailsPolicy] = None
        if allowed_fails_policy is not None:
            if isinstance(allowed_fails_policy, dict):
                self.allowed_fails_policy = AllowedFailsPolicy(**allowed_fails_policy)
            elif isinstance(allowed_fails_policy, AllowedFailsPolicy):
                self.allowed_fails_policy = allowed_fails_policy

            verbose_router_logger.info(
                "\033[32mRouter Custom Allowed Fails Policy Set:\n{}\033[0m".format(
                    self.allowed_fails_policy.model_dump(exclude_none=True)
                )
            )

        self.alerting_config: Optional[AlertingConfig] = alerting_config

        if optional_pre_call_checks is not None:
            self.add_optional_pre_call_checks(optional_pre_call_checks)

        if self.alerting_config is not None:
            self._initialize_alerting()

        self.initialize_assistants_endpoint()

        self.amoderation = self.factory_function(
            litellm.amoderation, call_type="moderation"
        )

    def initialize_assistants_endpoint(self):
        ## INITIALIZE PASS THROUGH ASSISTANTS ENDPOINT ##
        self.acreate_assistants = self.factory_function(litellm.acreate_assistants)
        self.adelete_assistant = self.factory_function(litellm.adelete_assistant)
        self.aget_assistants = self.factory_function(litellm.aget_assistants)
        self.acreate_thread = self.factory_function(litellm.acreate_thread)
        self.aget_thread = self.factory_function(litellm.aget_thread)
        self.a_add_message = self.factory_function(litellm.a_add_message)
        self.aget_messages = self.factory_function(litellm.aget_messages)
        self.arun_thread = self.factory_function(litellm.arun_thread)

    def validate_fallbacks(self, fallback_param: Optional[List]):
        """
        Validate the fallbacks parameter.
        """
        if fallback_param is None:
            return

        for fallback_dict in fallback_param:
            if not isinstance(fallback_dict, dict):
                raise ValueError(f"Item '{fallback_dict}' is not a dictionary.")
            if len(fallback_dict) != 1:
                raise ValueError(
                    f"Dictionary '{fallback_dict}' must have exactly one key, but has {len(fallback_dict)} keys."
                )

    def add_optional_pre_call_checks(
        self, optional_pre_call_checks: Optional[OptionalPreCallChecks]
    ):
        if optional_pre_call_checks is not None:
            for pre_call_check in optional_pre_call_checks:
                _callback: Optional[CustomLogger] = None
                if pre_call_check == "prompt_caching":
                    _callback = PromptCachingDeploymentCheck(cache=self.cache)
                elif pre_call_check == "router_budget_limiting":
                    _callback = RouterBudgetLimiting(
                        dual_cache=self.cache,
                        provider_budget_config=self.provider_budget_config,
                        model_list=self.model_list,
                    )
                if _callback is not None:
                    litellm.callbacks.append(_callback)

    def routing_strategy_init(
        self, routing_strategy: Union[RoutingStrategy, str], routing_strategy_args: dict
    ):
        verbose_router_logger.info(f"Routing strategy: {routing_strategy}")
        if (
            routing_strategy == RoutingStrategy.LEAST_BUSY.value
            or routing_strategy == RoutingStrategy.LEAST_BUSY
        ):
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
        elif (
            routing_strategy == RoutingStrategy.USAGE_BASED_ROUTING.value
            or routing_strategy == RoutingStrategy.USAGE_BASED_ROUTING
        ):
            self.lowesttpm_logger = LowestTPMLoggingHandler(
                router_cache=self.cache,
                model_list=self.model_list,
                routing_args=routing_strategy_args,
            )
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.lowesttpm_logger)  # type: ignore
        elif (
            routing_strategy == RoutingStrategy.USAGE_BASED_ROUTING_V2.value
            or routing_strategy == RoutingStrategy.USAGE_BASED_ROUTING_V2
        ):
            self.lowesttpm_logger_v2 = LowestTPMLoggingHandler_v2(
                router_cache=self.cache,
                model_list=self.model_list,
                routing_args=routing_strategy_args,
            )
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.lowesttpm_logger_v2)  # type: ignore
        elif (
            routing_strategy == RoutingStrategy.LATENCY_BASED.value
            or routing_strategy == RoutingStrategy.LATENCY_BASED
        ):
            self.lowestlatency_logger = LowestLatencyLoggingHandler(
                router_cache=self.cache,
                model_list=self.model_list,
                routing_args=routing_strategy_args,
            )
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.lowestlatency_logger)  # type: ignore
        elif (
            routing_strategy == RoutingStrategy.COST_BASED.value
            or routing_strategy == RoutingStrategy.COST_BASED
        ):
            self.lowestcost_logger = LowestCostLoggingHandler(
                router_cache=self.cache,
                model_list=self.model_list,
                routing_args={},
            )
            if isinstance(litellm.callbacks, list):
                litellm.callbacks.append(self.lowestcost_logger)  # type: ignore
        else:
            pass

    def print_deployment(self, deployment: dict):
        """
        returns a copy of the deployment with the api key masked

        Only returns 2 characters of the api key and masks the rest with * (10 *).
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
            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)

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
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

            data = deployment["litellm_params"].copy()
            model_name = data["model"]
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
            ## only run if model group given, not model id
            if model not in self.get_model_ids():
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
            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)

            request_priority = kwargs.get("priority") or self.default_priority

            start_time = time.time()
            if request_priority is not None and isinstance(request_priority, int):
                response = await self.schedule_acompletion(**kwargs)
            else:
                response = await self.async_function_with_fallbacks(**kwargs)
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.ROUTER,
                    duration=_duration,
                    call_type="acompletion",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                )
            )

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
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            start_time = time.time()
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=messages,
                specific_deployment=kwargs.pop("specific_deployment", None),
                request_kwargs=kwargs,
            )
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.ROUTER,
                    duration=_duration,
                    call_type="async_get_available_deployment",
                    start_time=start_time,
                    end_time=end_time,
                    parent_otel_span=_get_parent_otel_span_from_kwargs(kwargs),
                )
            )

            # debug how often this deployment picked

            self._track_deployment_metrics(
                deployment=deployment, parent_otel_span=parent_otel_span
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

            data = deployment["litellm_params"].copy()
            model_name = data["model"]

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            _response = litellm.acompletion(
                **{
                    **data,
                    "messages": messages,
                    "caching": self.cache_responses,
                    "client": model_client,
                    **kwargs,
                }
            )

            logging_obj: Optional[LiteLLMLogging] = kwargs.get(
                "litellm_logging_obj", None
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
                        deployment=deployment,
                        logging_obj=logging_obj,
                        parent_otel_span=parent_otel_span,
                    )
                    response = await _response
            else:
                await self.async_routing_strategy_pre_call_checks(
                    deployment=deployment,
                    logging_obj=logging_obj,
                    parent_otel_span=parent_otel_span,
                )

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
            self._track_deployment_metrics(
                deployment=deployment,
                response=response,
                parent_otel_span=parent_otel_span,
            )

            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.acompletion(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    def _update_kwargs_before_fallbacks(self, model: str, kwargs: dict) -> None:
        """
        Adds/updates to kwargs:
        - num_retries
        - litellm_trace_id
        - metadata
        """
        kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
        kwargs.setdefault("litellm_trace_id", str(uuid.uuid4()))
        kwargs.setdefault("metadata", {}).update({"model_group": model})

    def _update_kwargs_with_default_litellm_params(self, kwargs: dict) -> None:
        """
        Adds default litellm params to kwargs, if set.
        """
        for k, v in self.default_litellm_params.items():
            if (
                k not in kwargs and v is not None
            ):  # prioritize model-specific params > default router params
                kwargs[k] = v
            elif k == "metadata":
                kwargs[k].update(v)

    def _update_kwargs_with_deployment(self, deployment: dict, kwargs: dict) -> None:
        """
        2 jobs:
        - Adds selected deployment, model_info and api_base to kwargs["metadata"] (used for logging)
        - Adds default litellm params to kwargs, if set.
        """
        kwargs.setdefault("metadata", {}).update(
            {
                "deployment": deployment["litellm_params"]["model"],
                "model_info": deployment.get("model_info", {}),
                "api_base": deployment.get("litellm_params", {}).get("api_base"),
            }
        )
        kwargs["model_info"] = deployment.get("model_info", {})
        kwargs["timeout"] = self._get_timeout(
            kwargs=kwargs, data=deployment["litellm_params"]
        )

        self._update_kwargs_with_default_litellm_params(kwargs=kwargs)

    def _get_async_openai_model_client(self, deployment: dict, kwargs: dict):
        """
        Helper to get AsyncOpenAI or AsyncAzureOpenAI client that was created for the deployment

        The same OpenAI client is re-used to optimize latency / performance in production

        If dynamic api key is provided:
            Do not re-use the client. Pass model_client=None. The OpenAI/ AzureOpenAI client will be recreated in the handler for the llm provider
        """
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

        return model_client

    def _get_timeout(self, kwargs: dict, data: dict) -> Optional[Union[float, int]]:
        """Helper to get timeout from kwargs or deployment params"""
        timeout = (
            kwargs.get("timeout", None)  # the params dynamically set by user
            or kwargs.get("request_timeout", None)  # the params dynamically set by user
            or data.get(
                "timeout", None
            )  # timeout set on litellm_params for this deployment
            or data.get(
                "request_timeout", None
            )  # timeout set on litellm_params for this deployment
            or self.timeout  # timeout set on router
            or self.default_litellm_params.get("timeout", None)
        )

        return timeout

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
        parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
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
            _healthy_deployments, _ = await self._async_get_healthy_deployments(
                model=model, parent_otel_span=parent_otel_span
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
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = self.function_with_fallbacks(**kwargs)

            return response
        except Exception as e:
            raise e

    def _image_generation(self, prompt: str, model: str, **kwargs):
        model_name = ""
        try:
            verbose_router_logger.debug(
                f"Inside _image_generation()- model: {model}; kwargs: {kwargs}"
            )
            deployment = self.get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "prompt"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)
            data = deployment["litellm_params"].copy()

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )

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
        model_name = model
        try:
            verbose_router_logger.debug(
                f"Inside _image_generation()- model: {model}; kwargs: {kwargs}"
            )
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "prompt"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

            data = deployment["litellm_params"].copy()
            model_name = data["model"]

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )

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
                        deployment=deployment, parent_otel_span=parent_otel_span
                    )
                    response = await response
            else:
                await self.async_routing_strategy_pre_call_checks(
                    deployment=deployment, parent_otel_span=parent_otel_span
                )
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

    async def atranscription(self, file: FileTypes, model: str, **kwargs):
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
            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)
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

    async def _atranscription(self, file: FileTypes, model: str, **kwargs):
        model_name = model
        try:
            verbose_router_logger.debug(
                f"Inside _atranscription()- model: {model}; kwargs: {kwargs}"
            )
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "prompt"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )

            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)
            data = deployment["litellm_params"].copy()
            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )

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
                        deployment=deployment, parent_otel_span=parent_otel_span
                    )
                    response = await response
            else:
                await self.async_routing_strategy_pre_call_checks(
                    deployment=deployment, parent_otel_span=parent_otel_span
                )
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
            data["model"]
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

            response = await litellm.aspeech(
                **{
                    **data,
                    "client": model_client,
                    **kwargs,
                }
            )
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

    async def arerank(self, model: str, **kwargs):
        try:
            kwargs["model"] = model
            kwargs["input"] = input
            kwargs["original_function"] = self._arerank
            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)

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

    async def _arerank(self, model: str, **kwargs):
        model_name = None
        try:
            verbose_router_logger.debug(
                f"Inside _rerank()- model: {model}; kwargs: {kwargs}"
            )
            deployment = await self.async_get_available_deployment(
                model=model,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)
            data = deployment["litellm_params"].copy()
            model_name = data["model"]

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            response = await litellm.arerank(
                **{
                    **data,
                    "caching": self.cache_responses,
                    "client": model_client,
                    **kwargs,
                }
            )

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.arerank(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.arerank(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    async def _arealtime(self, model: str, **kwargs):
        messages = [{"role": "user", "content": "dummy-text"}]
        try:
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            kwargs.setdefault("metadata", {}).update({"model_group": model})

            # pick the one that is available (lowest TPM/RPM)
            deployment = await self.async_get_available_deployment(
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

            return await litellm._arealtime(**{**data, "caching": self.cache_responses, **kwargs})  # type: ignore
        except Exception as e:
            if self.num_retries > 0:
                kwargs["model"] = model
                kwargs["messages"] = messages
                kwargs["original_function"] = self._arealtime
                return await self.async_function_with_retries(**kwargs)
            else:
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
        messages = [{"role": "user", "content": prompt}]
        try:
            kwargs["model"] = model
            kwargs["prompt"] = prompt
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            kwargs.setdefault("metadata", {}).update({"model_group": model})

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
            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)
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
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

            data = deployment["litellm_params"].copy()
            model_name = data["model"]

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            response = litellm.atext_completion(
                **{
                    **data,
                    "prompt": prompt,
                    "caching": self.cache_responses,
                    "client": model_client,
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
                        deployment=deployment, parent_otel_span=parent_otel_span
                    )
                    response = await response
            else:
                await self.async_routing_strategy_pre_call_checks(
                    deployment=deployment, parent_otel_span=parent_otel_span
                )
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
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "default text"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

            data = deployment["litellm_params"].copy()
            model_name = data["model"]

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            response = litellm.aadapter_completion(
                **{
                    **data,
                    "adapter_id": adapter_id,
                    "caching": self.cache_responses,
                    "client": model_client,
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
                        deployment=deployment, parent_otel_span=parent_otel_span
                    )
                    response = await response  # type: ignore
            else:
                await self.async_routing_strategy_pre_call_checks(
                    deployment=deployment, parent_otel_span=parent_otel_span
                )
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
    ) -> EmbeddingResponse:
        try:
            kwargs["model"] = model
            kwargs["input"] = input
            kwargs["original_function"] = self._embedding
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            kwargs.setdefault("metadata", {}).update({"model_group": model})
            response = self.function_with_fallbacks(**kwargs)
            return response
        except Exception as e:
            raise e

    def _embedding(self, input: Union[str, List], model: str, **kwargs):
        model_name = None
        try:
            verbose_router_logger.debug(
                f"Inside embedding()- model: {model}; kwargs: {kwargs}"
            )
            deployment = self.get_available_deployment(
                model=model,
                input=input,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)
            data = deployment["litellm_params"].copy()
            model_name = data["model"]

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
    ) -> EmbeddingResponse:
        try:
            kwargs["model"] = model
            kwargs["input"] = input
            kwargs["original_function"] = self._aembedding
            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)
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
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                input=input,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )

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
                        deployment=deployment, parent_otel_span=parent_otel_span
                    )
                    response = await response
            else:
                await self.async_routing_strategy_pre_call_checks(
                    deployment=deployment, parent_otel_span=parent_otel_span
                )
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

    #### FILES API ####
    async def acreate_file(
        self,
        model: str,
        **kwargs,
    ) -> FileObject:
        try:
            kwargs["model"] = model
            kwargs["original_function"] = self._acreate_file
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
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

    async def _acreate_file(
        self,
        model: str,
        **kwargs,
    ) -> FileObject:
        try:
            verbose_router_logger.debug(
                f"Inside _atext_completion()- model: {model}; kwargs: {kwargs}"
            )
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "files-api-fake-text"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

            data = deployment["litellm_params"].copy()
            model_name = data["model"]

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            ## REPLACE MODEL IN FILE WITH SELECTED DEPLOYMENT ##
            stripped_model, custom_llm_provider, _, _ = get_llm_provider(
                model=data["model"]
            )
            kwargs["file"] = replace_model_in_jsonl(
                file_content=kwargs["file"], new_model_name=stripped_model
            )

            response = litellm.acreate_file(
                **{
                    **data,
                    "custom_llm_provider": custom_llm_provider,
                    "caching": self.cache_responses,
                    "client": model_client,
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
                        deployment=deployment, parent_otel_span=parent_otel_span
                    )
                    response = await response  # type: ignore
            else:
                await self.async_routing_strategy_pre_call_checks(
                    deployment=deployment, parent_otel_span=parent_otel_span
                )
                response = await response  # type: ignore

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.acreate_file(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response  # type: ignore
        except Exception as e:
            verbose_router_logger.exception(
                f"litellm.acreate_file(model={model}, {kwargs})\033[31m Exception {str(e)}\033[0m"
            )
            if model is not None:
                self.fail_calls[model] += 1
            raise e

    async def acreate_batch(
        self,
        model: str,
        **kwargs,
    ) -> Batch:
        try:
            kwargs["model"] = model
            kwargs["original_function"] = self._acreate_batch
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
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

    async def _acreate_batch(
        self,
        model: str,
        **kwargs,
    ) -> Batch:
        try:
            verbose_router_logger.debug(
                f"Inside _acreate_batch()- model: {model}; kwargs: {kwargs}"
            )
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "files-api-fake-text"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            metadata_variable_name = _get_router_metadata_variable_name(
                function_name="_acreate_batch"
            )

            kwargs.setdefault(metadata_variable_name, {}).update(
                {
                    "deployment": deployment["litellm_params"]["model"],
                    "model_info": deployment.get("model_info", {}),
                    "api_base": deployment.get("litellm_params", {}).get("api_base"),
                }
            )
            kwargs["model_info"] = deployment.get("model_info", {})
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            ## SET CUSTOM PROVIDER TO SELECTED DEPLOYMENT ##
            _, custom_llm_provider, _, _ = get_llm_provider(model=data["model"])

            response = litellm.acreate_batch(
                **{
                    **data,
                    "custom_llm_provider": custom_llm_provider,
                    "caching": self.cache_responses,
                    "client": model_client,
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
                        deployment=deployment, parent_otel_span=parent_otel_span
                    )
                    response = await response  # type: ignore
            else:
                await self.async_routing_strategy_pre_call_checks(
                    deployment=deployment, parent_otel_span=parent_otel_span
                )
                response = await response  # type: ignore

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"litellm.acreate_file(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response  # type: ignore
        except Exception as e:
            verbose_router_logger.exception(
                f"litellm._acreate_batch(model={model}, {kwargs})\033[31m Exception {str(e)}\033[0m"
            )
            if model is not None:
                self.fail_calls[model] += 1
            raise e

    async def aretrieve_batch(
        self,
        **kwargs,
    ) -> Batch:
        """
        Iterate through all models in a model group to check for batch

        Future Improvement - cache the result.
        """
        try:

            filtered_model_list = self.get_model_list()
            if filtered_model_list is None:
                raise Exception("Router not yet initialized.")

            receieved_exceptions = []

            async def try_retrieve_batch(model_name):
                try:
                    # Update kwargs with the current model name or any other model-specific adjustments
                    ## SET CUSTOM PROVIDER TO SELECTED DEPLOYMENT ##
                    _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
                        model=model_name["litellm_params"]["model"]
                    )
                    new_kwargs = copy.deepcopy(kwargs)
                    new_kwargs.pop("custom_llm_provider", None)
                    return await litellm.aretrieve_batch(
                        custom_llm_provider=custom_llm_provider, **new_kwargs  # type: ignore
                    )
                except Exception as e:
                    receieved_exceptions.append(e)
                    return None

            # Check all models in parallel
            results = await asyncio.gather(
                *[try_retrieve_batch(model) for model in filtered_model_list],
                return_exceptions=True,
            )

            # Check for successful responses and handle exceptions
            for result in results:
                if isinstance(result, Batch):
                    return result

            # If no valid Batch response was found, raise the first encountered exception
            if receieved_exceptions:
                raise receieved_exceptions[0]  # Raising the first exception encountered

            # If no exceptions were encountered, raise a generic exception
            raise Exception(
                "Unable to find batch in any model. Received errors - {}".format(
                    receieved_exceptions
                )
            )
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

    async def alist_batches(
        self,
        model: str,
        **kwargs,
    ):
        """
        Return all the batches across all deployments of a model group.
        """

        filtered_model_list = self.get_model_list(model_name=model)
        if filtered_model_list is None:
            raise Exception("Router not yet initialized.")

        async def try_retrieve_batch(model: DeploymentTypedDict):
            try:
                # Update kwargs with the current model name or any other model-specific adjustments
                return await litellm.alist_batches(
                    **{**model["litellm_params"], **kwargs}
                )
            except Exception:
                return None

        # Check all models in parallel
        results = await asyncio.gather(
            *[try_retrieve_batch(model) for model in filtered_model_list]
        )

        final_results = {
            "object": "list",
            "data": [],
            "first_id": None,
            "last_id": None,
            "has_more": False,
        }

        for result in results:
            if result is not None:
                ## check batch id
                if final_results["first_id"] is None and hasattr(result, "first_id"):
                    final_results["first_id"] = getattr(result, "first_id")
                final_results["last_id"] = getattr(result, "last_id")
                final_results["data"].extend(result.data)  # type: ignore

                ## check 'has_more'
                if getattr(result, "has_more", False) is True:
                    final_results["has_more"] = True

        return final_results

    #### PASSTHROUGH API ####

    async def _pass_through_moderation_endpoint_factory(
        self,
        original_function: Callable,
        **kwargs,
    ):
        if kwargs.get("model") and self.get_model_list(model_name=kwargs["model"]):
            deployment = await self.async_get_available_deployment(
                model=kwargs["model"]
            )
            kwargs["model"] = deployment["litellm_params"]["model"]
        return await original_function(**kwargs)

    def factory_function(
        self,
        original_function: Callable,
        call_type: Literal["assistants", "moderation"] = "assistants",
    ):
        async def new_function(
            custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
            client: Optional["AsyncOpenAI"] = None,
            **kwargs,
        ):
            if call_type == "assistants":
                return await self._pass_through_assistants_endpoint_factory(
                    original_function=original_function,
                    custom_llm_provider=custom_llm_provider,
                    client=client,
                    **kwargs,
                )
            elif call_type == "moderation":

                return await self._pass_through_moderation_endpoint_factory(  # type: ignore
                    original_function=original_function,
                    **kwargs,
                )

        return new_function

    async def _pass_through_assistants_endpoint_factory(
        self,
        original_function: Callable,
        custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
        client: Optional[AsyncOpenAI] = None,
        **kwargs,
    ):
        """Internal helper function to pass through the assistants endpoint"""
        if custom_llm_provider is None:
            if self.assistants_config is not None:
                custom_llm_provider = self.assistants_config["custom_llm_provider"]
                kwargs.update(self.assistants_config["litellm_params"])
            else:
                raise Exception(
                    "'custom_llm_provider' must be set. Either via:\n `Router(assistants_config={'custom_llm_provider': ..})` \nor\n `router.arun_thread(custom_llm_provider=..)`"
                )
        return await original_function(  # type: ignore
            custom_llm_provider=custom_llm_provider, client=client, **kwargs
        )

    #### [END] ASSISTANTS API ####

    async def async_function_with_fallbacks(self, *args, **kwargs):  # noqa: PLR0915
        """
        Try calling the function_with_retries
        If it fails after num_retries, fall back to another model group
        """
        model_group: Optional[str] = kwargs.get("model")
        disable_fallbacks: Optional[bool] = kwargs.pop("disable_fallbacks", False)
        fallbacks: Optional[List] = kwargs.get("fallbacks", self.fallbacks)
        context_window_fallbacks: Optional[List] = kwargs.get(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        content_policy_fallbacks: Optional[List] = kwargs.get(
            "content_policy_fallbacks", self.content_policy_fallbacks
        )

        mock_timeout = kwargs.pop("mock_timeout", None)

        try:
            self._handle_mock_testing_fallbacks(
                kwargs=kwargs,
                model_group=model_group,
                fallbacks=fallbacks,
                context_window_fallbacks=context_window_fallbacks,
                content_policy_fallbacks=content_policy_fallbacks,
            )

            response = await self.async_function_with_retries(
                *args, **kwargs, mock_timeout=mock_timeout
            )
            verbose_router_logger.debug(f"Async Response: {response}")
            return response
        except Exception as e:
            verbose_router_logger.debug(f"Traceback{traceback.format_exc()}")
            original_exception = e
            fallback_model_group = None
            original_model_group: Optional[str] = kwargs.get("model")  # type: ignore
            fallback_failure_exception_str = ""

            if disable_fallbacks is True or original_model_group is None:
                raise e

            input_kwargs = {
                "litellm_router": self,
                "original_exception": original_exception,
                **kwargs,
            }

            if "max_fallbacks" not in input_kwargs:
                input_kwargs["max_fallbacks"] = self.max_fallbacks
            if "fallback_depth" not in input_kwargs:
                input_kwargs["fallback_depth"] = 0

            try:
                verbose_router_logger.info("Trying to fallback b/w models")

                # check if client-side fallbacks are used (e.g. fallbacks = ["gpt-3.5-turbo", "claude-3-haiku"] or fallbacks=[{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hey, how's it going?"}]}]
                is_non_standard_fallback_format = _check_non_standard_fallback_format(
                    fallbacks=fallbacks
                )

                if is_non_standard_fallback_format:
                    input_kwargs.update(
                        {
                            "fallback_model_group": fallbacks,
                            "original_model_group": original_model_group,
                        }
                    )

                    response = await run_async_fallback(
                        *args,
                        **input_kwargs,
                    )

                    return response

                if isinstance(e, litellm.ContextWindowExceededError):
                    if context_window_fallbacks is not None:
                        fallback_model_group: Optional[List[str]] = (
                            self._get_fallback_model_group_from_fallbacks(
                                fallbacks=context_window_fallbacks,
                                model_group=model_group,
                            )
                        )
                        if fallback_model_group is None:
                            raise original_exception

                        input_kwargs.update(
                            {
                                "fallback_model_group": fallback_model_group,
                                "original_model_group": original_model_group,
                            }
                        )

                        response = await run_async_fallback(
                            *args,
                            **input_kwargs,
                        )
                        return response

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
                        fallback_model_group: Optional[List[str]] = (
                            self._get_fallback_model_group_from_fallbacks(
                                fallbacks=content_policy_fallbacks,
                                model_group=model_group,
                            )
                        )
                        if fallback_model_group is None:
                            raise original_exception

                        input_kwargs.update(
                            {
                                "fallback_model_group": fallback_model_group,
                                "original_model_group": original_model_group,
                            }
                        )

                        response = await run_async_fallback(
                            *args,
                            **input_kwargs,
                        )
                        return response
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
                if fallbacks is not None and model_group is not None:
                    verbose_router_logger.debug(f"inside model fallbacks: {fallbacks}")
                    fallback_model_group, generic_fallback_idx = (
                        get_fallback_model_group(
                            fallbacks=fallbacks,  # if fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}]
                            model_group=cast(str, model_group),
                        )
                    )
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
                            original_exception.message += f"No fallback model group found for original model_group={model_group}. Fallbacks={fallbacks}"  # type: ignore
                        raise original_exception

                    input_kwargs.update(
                        {
                            "fallback_model_group": fallback_model_group,
                            "original_model_group": original_model_group,
                        }
                    )

                    response = await run_async_fallback(
                        *args,
                        **input_kwargs,
                    )

                    return response
            except Exception as new_exception:
                traceback.print_exc()
                parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
                verbose_router_logger.error(
                    "litellm.router.py::async_function_with_fallbacks() - Error occurred while trying to do fallbacks - {}\n{}\n\nDebug Information:\nCooldown Deployments={}".format(
                        str(new_exception),
                        traceback.format_exc(),
                        await _async_get_cooldown_deployments_with_debug_info(
                            litellm_router_instance=self,
                            parent_otel_span=parent_otel_span,
                        ),
                    )
                )
                fallback_failure_exception_str = str(new_exception)

            if hasattr(original_exception, "message"):
                # add the available fallbacks to the exception
                original_exception.message += "\nReceived Model Group={}\nAvailable Model Group Fallbacks={}".format(  # type: ignore
                    model_group,
                    fallback_model_group,
                )
                if len(fallback_failure_exception_str) > 0:
                    original_exception.message += (  # type: ignore
                        "\nError doing the fallback: {}".format(
                            fallback_failure_exception_str
                        )
                    )

            raise original_exception

    def _handle_mock_testing_fallbacks(
        self,
        kwargs: dict,
        model_group: Optional[str] = None,
        fallbacks: Optional[List] = None,
        context_window_fallbacks: Optional[List] = None,
        content_policy_fallbacks: Optional[List] = None,
    ):
        """
        Helper function to raise a litellm Error for mock testing purposes.

        Raises:
            litellm.InternalServerError: when `mock_testing_fallbacks=True` passed in request params
            litellm.ContextWindowExceededError: when `mock_testing_context_fallbacks=True` passed in request params
            litellm.ContentPolicyViolationError: when `mock_testing_content_policy_fallbacks=True` passed in request params
        """
        mock_testing_fallbacks = kwargs.pop("mock_testing_fallbacks", None)
        mock_testing_context_fallbacks = kwargs.pop(
            "mock_testing_context_fallbacks", None
        )
        mock_testing_content_policy_fallbacks = kwargs.pop(
            "mock_testing_content_policy_fallbacks", None
        )

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

    async def async_function_with_retries(self, *args, **kwargs):  # noqa: PLR0915
        verbose_router_logger.debug(
            f"Inside async function with retries: args - {args}; kwargs - {kwargs}"
        )
        original_function = kwargs.pop("original_function")
        fallbacks = kwargs.pop("fallbacks", self.fallbacks)
        parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
        context_window_fallbacks = kwargs.pop(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        content_policy_fallbacks = kwargs.pop(
            "content_policy_fallbacks", self.content_policy_fallbacks
        )
        model_group: Optional[str] = kwargs.get("model")
        num_retries = kwargs.pop("num_retries")

        ## ADD MODEL GROUP SIZE TO METADATA - used for model_group_rate_limit_error tracking
        _metadata: dict = kwargs.get("metadata") or {}
        if "model_group" in _metadata and isinstance(_metadata["model_group"], str):
            model_list = self.get_model_list(model_name=_metadata["model_group"])
            if model_list is not None:
                _metadata.update({"model_group_size": len(model_list)})

        verbose_router_logger.debug(
            f"async function w/ retries: original_function - {original_function}, num_retries - {num_retries}"
        )
        try:
            self._handle_mock_testing_rate_limit_error(
                model_group=model_group, kwargs=kwargs
            )
            # if the function call is successful, no exception will be raised and we'll break out of the loop
            response = await self.make_call(original_function, *args, **kwargs)

            return response
        except Exception as e:
            current_attempt = None
            original_exception = e
            deployment_num_retries = getattr(e, "num_retries", None)
            if deployment_num_retries is not None and isinstance(
                deployment_num_retries, int
            ):
                num_retries = deployment_num_retries
            """
            Retry Logic
            """
            _healthy_deployments, _all_deployments = (
                await self._async_get_healthy_deployments(
                    model=kwargs.get("model") or "",
                    parent_otel_span=parent_otel_span,
                )
            )

            # raises an exception if this error should not be retries
            self.should_retry_this_error(
                error=e,
                healthy_deployments=_healthy_deployments,
                all_deployments=_all_deployments,
                context_window_fallbacks=context_window_fallbacks,
                regular_fallbacks=fallbacks,
                content_policy_fallbacks=content_policy_fallbacks,
            )

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
            else:
                raise

            # decides how long to sleep before retry
            retry_after = self._time_to_sleep_before_retry(
                e=original_exception,
                remaining_retries=num_retries,
                num_retries=num_retries,
                healthy_deployments=_healthy_deployments,
                all_deployments=_all_deployments,
            )

            await asyncio.sleep(retry_after)
            for current_attempt in range(num_retries):
                try:
                    # if the function call is successful, no exception will be raised and we'll break out of the loop
                    response = await self.make_call(original_function, *args, **kwargs)
                    if inspect.iscoroutinefunction(
                        response
                    ):  # async errors are often returned as coroutines
                        response = await response
                    return response

                except Exception as e:
                    ## LOGGING
                    kwargs = self.log_retry(kwargs=kwargs, e=e)
                    remaining_retries = num_retries - current_attempt
                    _model: Optional[str] = kwargs.get("model")  # type: ignore
                    if _model is not None:
                        _healthy_deployments, _ = (
                            await self._async_get_healthy_deployments(
                                model=_model,
                                parent_otel_span=parent_otel_span,
                            )
                        )
                    else:
                        _healthy_deployments = []
                    _timeout = self._time_to_sleep_before_retry(
                        e=original_exception,
                        remaining_retries=remaining_retries,
                        num_retries=num_retries,
                        healthy_deployments=_healthy_deployments,
                        all_deployments=_all_deployments,
                    )
                    await asyncio.sleep(_timeout)

            if type(original_exception) in litellm.LITELLM_EXCEPTION_TYPES:
                setattr(original_exception, "max_retries", num_retries)
                setattr(original_exception, "num_retries", current_attempt)

            raise original_exception

    async def make_call(self, original_function: Any, *args, **kwargs):
        """
        Handler for making a call to the .completion()/.embeddings()/etc. functions.
        """
        model_group = kwargs.get("model")
        response = original_function(*args, **kwargs)
        if inspect.iscoroutinefunction(response) or inspect.isawaitable(response):
            response = await response
        ## PROCESS RESPONSE HEADERS
        response = await self.set_response_headers(
            response=response, model_group=model_group
        )

        return response

    def _handle_mock_testing_rate_limit_error(
        self, kwargs: dict, model_group: Optional[str] = None
    ):
        """
        Helper function to raise a mock litellm.RateLimitError error for testing purposes.

        Raises:
            litellm.RateLimitError error when `mock_testing_rate_limit_error=True` passed in request params
        """
        mock_testing_rate_limit_error: Optional[bool] = kwargs.pop(
            "mock_testing_rate_limit_error", None
        )
        if (
            mock_testing_rate_limit_error is not None
            and mock_testing_rate_limit_error is True
        ):
            verbose_router_logger.info(
                f"litellm.router.py::_mock_rate_limit_error() - Raising mock RateLimitError for model={model_group}"
            )
            raise litellm.RateLimitError(
                model=model_group,
                llm_provider="",
                message=f"This is a mock exception for model={model_group}, to trigger a rate limit error.",
            )

    def should_retry_this_error(
        self,
        error: Exception,
        healthy_deployments: Optional[List] = None,
        all_deployments: Optional[List] = None,
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
        _num_all_deployments = 0
        if all_deployments is not None and isinstance(all_deployments, list):
            _num_all_deployments = len(all_deployments)

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

        if isinstance(error, litellm.NotFoundError):
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
            if (
                _num_all_deployments <= 1
            ):  # if there is only 1 deployment for this model group then don't retry
                raise error  # then raise error

        # Do not retry if there are no healthy deployments
        # just raise the error
        if _num_healthy_deployments <= 0:  # if no healthy deployments
            raise error

        return True

    def function_with_fallbacks(self, *args, **kwargs):
        """
        Sync wrapper for async_function_with_fallbacks

        Wrapped to reduce code duplication and prevent bugs.
        """
        from concurrent.futures import ThreadPoolExecutor

        def run_in_new_loop():
            """Run the coroutine in a new event loop within this thread."""
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(
                    self.async_function_with_fallbacks(*args, **kwargs)
                )
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        try:
            # First, try to get the current event loop
            _ = asyncio.get_running_loop()
            # If we're already in an event loop, run in a separate thread
            # to avoid nested event loop issues
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()

        except RuntimeError:
            # No running event loop, we can safely run in this thread
            return run_in_new_loop()

    def _get_fallback_model_group_from_fallbacks(
        self,
        fallbacks: List[Dict[str, List[str]]],
        model_group: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        Returns the list of fallback models to use for a given model group

        If no fallback model group is found, returns None

        Example:
            fallbacks = [{"gpt-3.5-turbo": ["gpt-4"]}, {"gpt-4o": ["gpt-3.5-turbo"]}]
            model_group = "gpt-3.5-turbo"
            returns: ["gpt-4"]
        """
        if model_group is None:
            return None

        fallback_model_group: Optional[List[str]] = None
        for item in fallbacks:  # [{"gpt-3.5-turbo": ["gpt-4"]}]
            if list(item.keys())[0] == model_group:
                fallback_model_group = item[model_group]
                break
        return fallback_model_group

    def _time_to_sleep_before_retry(
        self,
        e: Exception,
        remaining_retries: int,
        num_retries: int,
        healthy_deployments: Optional[List] = None,
        all_deployments: Optional[List] = None,
    ) -> Union[int, float]:
        """
        Calculate back-off, then retry

        It should instantly retry only when:
            1. there are healthy deployments in the same model group
            2. there are fallbacks for the completion call
        """

        ## base case - single deployment
        if all_deployments is not None and len(all_deployments) == 1:
            pass
        elif (
            healthy_deployments is not None
            and isinstance(healthy_deployments, list)
            and len(healthy_deployments) > 0
        ):
            return 0

        response_headers: Optional[httpx.Headers] = None
        if hasattr(e, "response") and hasattr(e.response, "headers"):  # type: ignore
            response_headers = e.response.headers  # type: ignore
        if hasattr(e, "litellm_response_headers"):
            response_headers = e.litellm_response_headers  # type: ignore

        if response_headers is not None:
            timeout = litellm._calculate_retry_after(
                remaining_retries=remaining_retries,
                max_retries=num_retries,
                response_headers=response_headers,
                min_timeout=self.retry_after,
            )

        else:
            timeout = litellm._calculate_retry_after(
                remaining_retries=remaining_retries,
                max_retries=num_retries,
                min_timeout=self.retry_after,
            )

        return timeout

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
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )
            if standard_logging_object is None:
                raise ValueError("standard_logging_object is None")
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                deployment_name = kwargs["litellm_params"]["metadata"].get(
                    "deployment", None
                )  # stable name - works for wildcard routes as well
                model_group = standard_logging_object.get("model_group", None)
                id = standard_logging_object.get("model_id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
                total_tokens: float = standard_logging_object.get("total_tokens", 0)

                # ------------
                # Setup values
                # ------------
                dt = get_utc_datetime()
                current_minute = dt.strftime(
                    "%H-%M"
                )  # use the same timezone regardless of system clock

                tpm_key = RouterCacheEnum.TPM.value.format(
                    id=id, current_minute=current_minute, model=deployment_name
                )
                # ------------
                # Update usage
                # ------------
                # update cache

                ## TPM
                await self.cache.async_increment_cache(
                    key=tpm_key,
                    value=total_tokens,
                    parent_otel_span=parent_otel_span,
                    ttl=RoutingArgs.ttl.value,
                )

                ## RPM
                rpm_key = RouterCacheEnum.RPM.value.format(
                    id=id, current_minute=current_minute, model=deployment_name
                )
                await self.cache.async_increment_cache(
                    key=rpm_key,
                    value=1,
                    parent_otel_span=parent_otel_span,
                    ttl=RoutingArgs.ttl.value,
                )

                increment_deployment_successes_for_current_minute(
                    litellm_router_instance=self,
                    deployment_id=id,
                )

                return tpm_key

        except Exception as e:
            verbose_router_logger.exception(
                "litellm.router.Router::deployment_callback_on_success(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    def sync_deployment_callback_on_success(
        self,
        kwargs,  # kwargs to completion
        completion_response,  # response from completion
        start_time,
        end_time,  # start/end time
    ) -> Optional[str]:
        """
        Tracks the number of successes for a deployment in the current minute (using in-memory cache)

        Returns:
        - key: str - The key used to increment the cache
        - None: if no key is found
        """
        id = None
        if kwargs["litellm_params"].get("metadata") is None:
            pass
        else:
            model_group = kwargs["litellm_params"]["metadata"].get("model_group", None)
            model_info = kwargs["litellm_params"].get("model_info", {}) or {}
            id = model_info.get("id", None)
            if model_group is None or id is None:
                return None
            elif isinstance(id, int):
                id = str(id)

        if id is not None:
            key = increment_deployment_successes_for_current_minute(
                litellm_router_instance=self,
                deployment_id=id,
            )
            return key

        return None

    def deployment_callback_on_failure(
        self,
        kwargs,  # kwargs to completion
        completion_response,  # response from completion
        start_time,
        end_time,  # start/end time
    ) -> bool:
        """
        2 jobs:
        - Tracks the number of failures for a deployment in the current minute (using in-memory cache)
        - Puts the deployment in cooldown if it exceeds the allowed fails / minute

        Returns:
        - True if the deployment should be put in cooldown
        - False if the deployment should not be put in cooldown
        """
        try:
            exception = kwargs.get("exception", None)
            exception_status = getattr(exception, "status_code", "")
            _model_info = kwargs.get("litellm_params", {}).get("model_info", {})

            exception_headers = litellm.litellm_core_utils.exception_mapping_utils._get_response_headers(
                original_exception=exception
            )

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
                increment_deployment_failures_for_current_minute(
                    litellm_router_instance=self,
                    deployment_id=deployment_id,
                )
                result = _set_cooldown_deployments(
                    litellm_router_instance=self,
                    exception_status=exception_status,
                    original_exception=exception,
                    deployment=deployment_id,
                    time_to_cooldown=_time_to_cooldown,
                )  # setting deployment_id in cooldown deployments

                return result
            else:
                return False

        except Exception as e:
            raise e

    async def async_deployment_callback_on_failure(
        self, kwargs, completion_response: Optional[Any], start_time, end_time
    ):
        """
        Update RPM usage for a deployment
        """
        deployment_name = kwargs["litellm_params"]["metadata"].get(
            "deployment", None
        )  # handles wildcard routes - by giving the original name sent to `litellm.completion`
        model_group = kwargs["litellm_params"]["metadata"].get("model_group", None)
        model_info = kwargs["litellm_params"].get("model_info", {}) or {}
        id = model_info.get("id", None)
        if model_group is None or id is None:
            return
        elif isinstance(id, int):
            id = str(id)
        parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)

        dt = get_utc_datetime()
        current_minute = dt.strftime(
            "%H-%M"
        )  # use the same timezone regardless of system clock

        ## RPM
        rpm_key = RouterCacheEnum.RPM.value.format(
            id=id, current_minute=current_minute, model=deployment_name
        )
        await self.cache.async_increment_cache(
            key=rpm_key,
            value=1,
            parent_otel_span=parent_otel_span,
            ttl=RoutingArgs.ttl.value,
        )

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

    def _update_usage(
        self, deployment_id: str, parent_otel_span: Optional[Span]
    ) -> int:
        """
        Update deployment rpm for that minute

        Returns:
        - int: request count
        """
        rpm_key = deployment_id

        request_count = self.cache.get_cache(
            key=rpm_key, parent_otel_span=parent_otel_span, local_only=True
        )
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

        return request_count

    def _is_cooldown_required(
        self,
        model_id: str,
        exception_status: Union[str, int],
        exception_str: Optional[str] = None,
    ) -> bool:
        """
        A function to determine if a cooldown is required based on the exception status.

        Parameters:
            model_id (str) The id of the model in the model list
            exception_status (Union[str, int]): The status of the exception.

        Returns:
            bool: True if a cooldown is required, False otherwise.
        """
        ## BASE CASE - single deployment
        model_group = self.get_model_group(id=model_id)
        if model_group is not None and len(model_group) == 1:
            return False

        try:
            ignored_strings = ["APIConnectionError"]
            if (
                exception_str is not None
            ):  # don't cooldown on litellm api connection errors errors
                for ignored_string in ignored_strings:
                    if ignored_string in exception_str:
                        return False

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

        except Exception:
            # Catch all - if any exceptions default to cooling down
            return True

    def _has_default_fallbacks(self) -> bool:
        if self.fallbacks is None:
            return False
        for fallback in self.fallbacks:
            if isinstance(fallback, dict):
                if "*" in fallback:
                    return True
        return False

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
        elif self._has_default_fallbacks():  # default fallbacks set
            return True

        verbose_router_logger.info(
            "Content Policy Error occurred. No available fallbacks. Returning original response. model={}, content_policy_fallbacks={}".format(
                model, content_policy_fallbacks
            )
        )
        return False

    def _get_healthy_deployments(self, model: str, parent_otel_span: Optional[Span]):
        _all_deployments: list = []
        try:
            _, _all_deployments = self._common_checks_available_deployment(  # type: ignore
                model=model,
            )
            if isinstance(_all_deployments, dict):
                return []
        except Exception:
            pass

        unhealthy_deployments = _get_cooldown_deployments(
            litellm_router_instance=self, parent_otel_span=parent_otel_span
        )
        healthy_deployments: list = []
        for deployment in _all_deployments:
            if deployment["model_info"]["id"] in unhealthy_deployments:
                continue
            else:
                healthy_deployments.append(deployment)

        return healthy_deployments, _all_deployments

    async def _async_get_healthy_deployments(
        self, model: str, parent_otel_span: Optional[Span]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Returns Tuple of:
        - Tuple[List[Dict], List[Dict]]:
            1. healthy_deployments: list of healthy deployments
            2. all_deployments: list of all deployments
        """
        _all_deployments: list = []
        try:
            _, _all_deployments = self._common_checks_available_deployment(  # type: ignore
                model=model,
            )
            if isinstance(_all_deployments, dict):
                return [], _all_deployments
        except Exception:
            pass

        unhealthy_deployments = await _async_get_cooldown_deployments(
            litellm_router_instance=self, parent_otel_span=parent_otel_span
        )
        healthy_deployments: list = []
        for deployment in _all_deployments:
            if deployment["model_info"]["id"] in unhealthy_deployments:
                continue
            else:
                healthy_deployments.append(deployment)
        return healthy_deployments, _all_deployments

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
                _callback.pre_call_check(deployment)

    async def async_routing_strategy_pre_call_checks(
        self,
        deployment: dict,
        parent_otel_span: Optional[Span],
        logging_obj: Optional[LiteLLMLogging] = None,
    ):
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
                    await _callback.async_pre_call_check(deployment, parent_otel_span)
                except litellm.RateLimitError as e:
                    ## LOG FAILURE EVENT
                    if logging_obj is not None:
                        asyncio.create_task(
                            logging_obj.async_failure_handler(
                                exception=e,
                                traceback_exception=traceback.format_exc(),
                                end_time=time.time(),
                            )
                        )
                        ## LOGGING
                        threading.Thread(
                            target=logging_obj.failure_handler,
                            args=(e, traceback.format_exc()),
                        ).start()  # log response
                    _set_cooldown_deployments(
                        litellm_router_instance=self,
                        exception_status=e.status_code,
                        original_exception=e,
                        deployment=deployment["model_info"]["id"],
                        time_to_cooldown=self.cooldown_time,
                    )
                    raise e
                except Exception as e:
                    ## LOG FAILURE EVENT
                    if logging_obj is not None:
                        asyncio.create_task(
                            logging_obj.async_failure_handler(
                                exception=e,
                                traceback_exception=traceback.format_exc(),
                                end_time=time.time(),
                            )
                        )
                        ## LOGGING
                        threading.Thread(
                            target=logging_obj.failure_handler,
                            args=(e, traceback.format_exc()),
                        ).start()  # log response
                    raise e

    async def async_callback_filter_deployments(
        self,
        model: str,
        healthy_deployments: List[dict],
        messages: Optional[List[AllMessageValues]],
        parent_otel_span: Optional[Span],
        request_kwargs: Optional[dict] = None,
        logging_obj: Optional[LiteLLMLogging] = None,
    ):
        """
        For usage-based-routing-v2, enables running rpm checks before the call is made, inside the semaphore.

        -> makes the calls concurrency-safe, when rpm limits are set for a deployment

        Returns:
        - None

        Raises:
        - Rate Limit Exception - If the deployment is over it's tpm/rpm limits
        """
        returned_healthy_deployments = healthy_deployments
        for _callback in litellm.callbacks:
            if isinstance(_callback, CustomLogger):
                try:
                    returned_healthy_deployments = (
                        await _callback.async_filter_deployments(
                            model=model,
                            healthy_deployments=returned_healthy_deployments,
                            messages=messages,
                            request_kwargs=request_kwargs,
                            parent_otel_span=parent_otel_span,
                        )
                    )
                except Exception as e:
                    ## LOG FAILURE EVENT
                    if logging_obj is not None:
                        asyncio.create_task(
                            logging_obj.async_failure_handler(
                                exception=e,
                                traceback_exception=traceback.format_exc(),
                                end_time=time.time(),
                            )
                        )
                        ## LOGGING
                        threading.Thread(
                            target=logging_obj.failure_handler,
                            args=(e, traceback.format_exc()),
                        ).start()  # log response
                    raise e
        return returned_healthy_deployments

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
        self,
        deployment_info: dict,
        _model_name: str,
        _litellm_params: dict,
        _model_info: dict,
    ) -> Optional[Deployment]:
        """
        Create a deployment object and add it to the model list

        If the deployment is not active for the current environment, it is ignored

        Returns:
        - Deployment: The deployment object
        - None: If the deployment is not active for the current environment (if 'supported_environments' is set in litellm_params)
        """
        deployment = Deployment(
            **deployment_info,
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

        ## Check if LLM Deployment is allowed for this deployment
        if self.deployment_is_active_for_environment(deployment=deployment) is not True:
            verbose_router_logger.warning(
                f"Ignoring deployment {deployment.model_name} as it is not active for environment {deployment.model_info['supported_environments']}"
            )
            return None

        deployment = self._add_deployment(deployment=deployment)

        model = deployment.to_json(exclude_none=True)

        self.model_list.append(model)
        return deployment

    def deployment_is_active_for_environment(self, deployment: Deployment) -> bool:
        """
        Function to check if a llm deployment is active for a given environment. Allows using the same config.yaml across multople environments

        Requires `LITELLM_ENVIRONMENT` to be set in .env. Valid values for environment:
            - development
            - staging
            - production

        Raises:
        - ValueError: If LITELLM_ENVIRONMENT is not set in .env or not one of the valid values
        - ValueError: If supported_environments is not set in model_info or not one of the valid values
        """
        if (
            deployment.model_info is None
            or "supported_environments" not in deployment.model_info
            or deployment.model_info["supported_environments"] is None
        ):
            return True
        litellm_environment = get_secret_str(secret_name="LITELLM_ENVIRONMENT")
        if litellm_environment is None:
            raise ValueError(
                "Set 'supported_environments' for model but not 'LITELLM_ENVIRONMENT' set in .env"
            )

        if litellm_environment not in VALID_LITELLM_ENVIRONMENTS:
            raise ValueError(
                f"LITELLM_ENVIRONMENT must be one of {VALID_LITELLM_ENVIRONMENTS}. but set as: {litellm_environment}"
            )

        for _env in deployment.model_info["supported_environments"]:
            if _env not in VALID_LITELLM_ENVIRONMENTS:
                raise ValueError(
                    f"supported_environments must be one of {VALID_LITELLM_ENVIRONMENTS}. but set as: {_env} for deployment: {deployment}"
                )

        if litellm_environment in deployment.model_info["supported_environments"]:
            return True
        return False

    def set_model_list(self, model_list: list):
        original_model_list = copy.deepcopy(model_list)
        self.model_list = []
        # we add api_base/api_key each model so load balancing between azure/gpt on api_base1 and api_base2 works

        for model in original_model_list:
            _model_name = model.pop("model_name")
            _litellm_params = model.pop("litellm_params")
            ## check if litellm params in os.environ
            if isinstance(_litellm_params, dict):
                for k, v in _litellm_params.items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        _litellm_params[k] = get_secret(v)

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
                        deployment_info=model,
                        _model_name=_model_name,
                        _litellm_params=_litellm_params,
                        _model_info=_model_info,
                    )
            else:
                self._create_deployment(
                    deployment_info=model,
                    _model_name=_model_name,
                    _litellm_params=_litellm_params,
                    _model_info=_model_info,
                )

        verbose_router_logger.debug(
            f"\nInitialized Model List {self.get_model_names()}"
        )
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
        # if deployment.model_name == "*":
        #     if deployment.litellm_params.model == "*":
        #         # user wants to pass through all requests to litellm.acompletion for unknown deployments
        #         self.router_general_settings.pass_through_all_models = True
        #     else:
        #         self.default_deployment = deployment.to_json(exclude_none=True)
        # Check if user is using provider specific wildcard routing
        # example model_name = "databricks/*" or model_name = "anthropic/*"
        if "*" in deployment.model_name:
            # store this as a regex pattern - all deployments matching this pattern will be sent to this deployment
            # Store deployment.model_name as a regex pattern
            self.pattern_router.add_pattern(
                deployment.model_name, deployment.to_json(exclude_none=True)
            )
            if deployment.model_info.id:
                self.provider_default_deployment_ids.append(deployment.model_info.id)

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
        InitalizeOpenAISDKClient.set_client(
            litellm_router_instance=self, model=deployment.to_json(exclude_none=True)
        )

        # set region (if azure model) ## PREVIEW FEATURE ##
        if litellm.enable_preview_features is True:
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
        except Exception:
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

    @overload
    def get_router_model_info(
        self, deployment: dict, received_model_name: str, id: None = None
    ) -> ModelMapInfo:
        pass

    @overload
    def get_router_model_info(
        self, deployment: None, received_model_name: str, id: str
    ) -> ModelMapInfo:
        pass

    def get_router_model_info(
        self,
        deployment: Optional[dict],
        received_model_name: str,
        id: Optional[str] = None,
    ) -> ModelMapInfo:
        """
        For a given model id, return the model info (max tokens, input cost, output cost, etc.).

        Augment litellm info with additional params set in `model_info`.

        For azure models, ignore the `model:`. Only set max tokens, cost values if base_model is set.

        Returns
        - ModelInfo - If found -> typed dict with max tokens, input cost, etc.

        Raises:
        - ValueError -> If model is not mapped yet
        """
        if id is not None:
            _deployment = self.get_deployment(model_id=id)
            if _deployment is not None:
                deployment = _deployment.model_dump(exclude_none=True)

        if deployment is None:
            raise ValueError("Deployment not found")

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

            potential_models = self.pattern_router.route(received_model_name)
            if "*" in model and potential_models is not None:  # if wildcard route
                for potential_model in potential_models:
                    try:
                        if potential_model.get("model_info", {}).get(
                            "id"
                        ) == deployment.get("model_info", {}).get("id"):
                            model = potential_model.get("litellm_params", {}).get(
                                "model"
                            )
                            break
                    except Exception:
                        pass

        ## GET LITELLM MODEL INFO - raises exception, if model is not mapped
        if not model.startswith(custom_llm_provider):
            model_info_name = "{}/{}".format(custom_llm_provider, model)
        else:
            model_info_name = model

        model_info = litellm.get_model_info(model=model_info_name)

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

    def get_model_group(self, id: str) -> Optional[List]:
        """
        Return list of all models in the same model group as that model id
        """

        model_info = self.get_model_info(id=id)
        if model_info is None:
            return None

        model_name = model_info["model_name"]
        return self.get_model_list(model_name=model_name)

    def _set_model_group_info(  # noqa: PLR0915
        self, model_group: str, user_facing_model_group_name: str
    ) -> Optional[ModelGroupInfo]:
        """
        For a given model group name, return the combined model info

        Returns:
        - ModelGroupInfo if able to construct a model group
        - None if error constructing model group info
        """
        model_group_info: Optional[ModelGroupInfo] = None

        total_tpm: Optional[int] = None
        total_rpm: Optional[int] = None
        configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS = None
        model_list = self.get_model_list(model_name=model_group)
        if model_list is None:
            return None
        for model in model_list:
            is_match = False
            if (
                "model_name" in model and model["model_name"] == model_group
            ):  # exact match
                is_match = True
            elif (
                "model_name" in model
                and self.pattern_router.route(model_group) is not None
            ):  # wildcard model
                is_match = True

            if not is_match:
                continue
            # model in model group found #
            litellm_params = LiteLLM_Params(**model["litellm_params"])  # type: ignore
            # get configurable clientside auth params
            configurable_clientside_auth_params = (
                litellm_params.configurable_clientside_auth_params
            )
            # get model tpm
            _deployment_tpm: Optional[int] = None
            if _deployment_tpm is None:
                _deployment_tpm = model.get("tpm", None)  # type: ignore
            if _deployment_tpm is None:
                _deployment_tpm = model.get("litellm_params", {}).get("tpm", None)  # type: ignore
            if _deployment_tpm is None:
                _deployment_tpm = model.get("model_info", {}).get("tpm", None)  # type: ignore

            # get model rpm
            _deployment_rpm: Optional[int] = None
            if _deployment_rpm is None:
                _deployment_rpm = model.get("rpm", None)  # type: ignore
            if _deployment_rpm is None:
                _deployment_rpm = model.get("litellm_params", {}).get("rpm", None)  # type: ignore
            if _deployment_rpm is None:
                _deployment_rpm = model.get("model_info", {}).get("rpm", None)  # type: ignore

            # get model info
            try:
                model_info = litellm.get_model_info(model=litellm_params.model)
            except Exception:
                model_info = None
            # get llm provider
            litellm_model, llm_provider = "", ""
            try:
                litellm_model, llm_provider, _, _ = litellm.get_llm_provider(
                    model=litellm_params.model,
                    custom_llm_provider=litellm_params.custom_llm_provider,
                )
            except litellm.exceptions.BadRequestError as e:
                verbose_router_logger.error(
                    "litellm.router.py::get_model_group_info() - {}".format(str(e))
                )

            if model_info is None:
                supported_openai_params = litellm.get_supported_openai_params(
                    model=litellm_model, custom_llm_provider=llm_provider
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
                    model_group=user_facing_model_group_name, providers=[llm_provider], **model_info  # type: ignore
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
                    model_group_info.max_input_tokens = model_info["max_input_tokens"]
                if (
                    model_info.get("max_output_tokens", None) is not None
                    and model_info["max_output_tokens"] is not None
                    and (
                        model_group_info.max_output_tokens is None
                        or model_info["max_output_tokens"]
                        > model_group_info.max_output_tokens
                    )
                ):
                    model_group_info.max_output_tokens = model_info["max_output_tokens"]
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
                if model_info.get("tpm", None) is not None and _deployment_tpm is None:
                    _deployment_tpm = model_info.get("tpm")
                if model_info.get("rpm", None) is not None and _deployment_rpm is None:
                    _deployment_rpm = model_info.get("rpm")

            if _deployment_tpm is not None:
                if total_tpm is None:
                    total_tpm = 0
                total_tpm += _deployment_tpm  # type: ignore

            if _deployment_rpm is not None:
                if total_rpm is None:
                    total_rpm = 0
                total_rpm += _deployment_rpm  # type: ignore
        if model_group_info is not None:
            ## UPDATE WITH TOTAL TPM/RPM FOR MODEL GROUP
            if total_tpm is not None:
                model_group_info.tpm = total_tpm

            if total_rpm is not None:
                model_group_info.rpm = total_rpm

            ## UPDATE WITH CONFIGURABLE CLIENTSIDE AUTH PARAMS FOR MODEL GROUP
            if configurable_clientside_auth_params is not None:
                model_group_info.configurable_clientside_auth_params = (
                    configurable_clientside_auth_params
                )

        return model_group_info

    def get_model_group_info(self, model_group: str) -> Optional[ModelGroupInfo]:
        """
        For a given model group name, return the combined model info

        Returns:
        - ModelGroupInfo if able to construct a model group
        - None if error constructing model group info or hidden model group
        """
        ## Check if model group alias
        if model_group in self.model_group_alias:
            item = self.model_group_alias[model_group]
            if isinstance(item, str):
                _router_model_group = item
            elif isinstance(item, dict):
                if item["hidden"] is True:
                    return None
                else:
                    _router_model_group = item["model"]
            else:
                return None

            return self._set_model_group_info(
                model_group=_router_model_group,
                user_facing_model_group_name=model_group,
            )

        ## Check if actual model
        return self._set_model_group_info(
            model_group=model_group, user_facing_model_group_name=model_group
        )

    async def get_model_group_usage(
        self, model_group: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns current tpm/rpm usage for model group

        Parameters:
        - model_group: str - the received model name from the user (can be a wildcard route).

        Returns:
        - usage: Tuple[tpm, rpm]
        """
        dt = get_utc_datetime()
        current_minute = dt.strftime(
            "%H-%M"
        )  # use the same timezone regardless of system clock
        tpm_keys: List[str] = []
        rpm_keys: List[str] = []

        model_list = self.get_model_list(model_name=model_group)
        if model_list is None:  # no matching deployments
            return None, None

        for model in model_list:
            id: Optional[str] = model.get("model_info", {}).get("id")  # type: ignore
            litellm_model: Optional[str] = model["litellm_params"].get(
                "model"
            )  # USE THE MODEL SENT TO litellm.completion() - consistent with how global_router cache is written.
            if id is None or litellm_model is None:
                continue
            tpm_keys.append(
                RouterCacheEnum.TPM.value.format(
                    id=id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
            )
            rpm_keys.append(
                RouterCacheEnum.RPM.value.format(
                    id=id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
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

    async def get_remaining_model_group_usage(self, model_group: str) -> Dict[str, int]:

        current_tpm, current_rpm = await self.get_model_group_usage(model_group)

        model_group_info = self.get_model_group_info(model_group)

        if model_group_info is not None and model_group_info.tpm is not None:
            tpm_limit = model_group_info.tpm
        else:
            tpm_limit = None

        if model_group_info is not None and model_group_info.rpm is not None:
            rpm_limit = model_group_info.rpm
        else:
            rpm_limit = None

        returned_dict = {}
        if tpm_limit is not None:
            returned_dict["x-ratelimit-remaining-tokens"] = tpm_limit - (
                current_tpm or 0
            )
            returned_dict["x-ratelimit-limit-tokens"] = tpm_limit
        if rpm_limit is not None:
            returned_dict["x-ratelimit-remaining-requests"] = rpm_limit - (
                current_rpm or 0
            )
            returned_dict["x-ratelimit-limit-requests"] = rpm_limit

        return returned_dict

    async def set_response_headers(
        self, response: Any, model_group: Optional[str] = None
    ) -> Any:
        """
        Add the most accurate rate limit headers for a given model response.

        ## TODO: add model group rate limit headers
        # - if healthy_deployments > 1, return model group rate limit headers
        # - else return the model's rate limit headers
        """
        if (
            isinstance(response, BaseModel)
            and hasattr(response, "_hidden_params")
            and isinstance(response._hidden_params, dict)  # type: ignore
        ):
            response._hidden_params.setdefault("additional_headers", {})  # type: ignore
            response._hidden_params["additional_headers"][  # type: ignore
                "x-litellm-model-group"
            ] = model_group

            additional_headers = response._hidden_params["additional_headers"]  # type: ignore

            if (
                "x-ratelimit-remaining-tokens" not in additional_headers
                and "x-ratelimit-remaining-requests" not in additional_headers
                and model_group is not None
            ):
                remaining_usage = await self.get_remaining_model_group_usage(
                    model_group
                )

                for header, value in remaining_usage.items():
                    if value is not None:
                        additional_headers[header] = value
        return response

    def get_model_ids(self, model_name: Optional[str] = None) -> List[str]:
        """
        if 'model_name' is none, returns all.

        Returns list of model id's.
        """
        ids = []
        for model in self.model_list:
            if "model_info" in model and "id" in model["model_info"]:
                id = model["model_info"]["id"]
                if model_name is not None and model["model_name"] == model_name:
                    ids.append(id)
                elif model_name is None:
                    ids.append(id)
        return ids

    def _get_all_deployments(
        self, model_name: str, model_alias: Optional[str] = None
    ) -> List[DeploymentTypedDict]:
        """
        Return all deployments of a model name

        Used for accurate 'get_model_list'.
        """
        returned_models: List[DeploymentTypedDict] = []
        for model in self.model_list:
            if model_name is not None and model["model_name"] == model_name:
                if model_alias is not None:
                    alias_model = copy.deepcopy(model)
                    alias_model["model_name"] = model_alias
                    returned_models.append(alias_model)
                else:
                    returned_models.append(model)

        return returned_models

    def get_model_names(self) -> List[str]:
        """
        Returns all possible model names for router.

        Includes model_group_alias models too.
        """
        model_list = self.get_model_list()
        if model_list is None:
            return []

        model_names = []
        for m in model_list:
            model_names.append(m["model_name"])
        return model_names

    def get_model_list(
        self, model_name: Optional[str] = None
    ) -> Optional[List[DeploymentTypedDict]]:
        """
        Includes router model_group_alias'es as well
        """
        if hasattr(self, "model_list"):
            returned_models: List[DeploymentTypedDict] = []

            if model_name is not None:
                returned_models.extend(self._get_all_deployments(model_name=model_name))

            if hasattr(self, "model_group_alias"):
                for model_alias, model_value in self.model_group_alias.items():

                    if isinstance(model_value, str):
                        _router_model_name: str = model_value
                    elif isinstance(model_value, dict):
                        _model_value = RouterModelGroupAliasItem(**model_value)  # type: ignore
                        if _model_value["hidden"] is True:
                            continue
                        else:
                            _router_model_name = _model_value["model"]
                    else:
                        continue

                    returned_models.extend(
                        self._get_all_deployments(
                            model_name=_router_model_name, model_alias=model_alias
                        )
                    )

            if len(returned_models) == 0:  # check if wildcard route
                potential_wildcard_models = self.pattern_router.route(model_name)
                if potential_wildcard_models is not None:
                    returned_models.extend(
                        [DeploymentTypedDict(**m) for m in potential_wildcard_models]  # type: ignore
                    )

            if model_name is None:
                returned_models += self.model_list

                return returned_models

            return returned_models
        return None

    def get_model_access_groups(
        self, model_name: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        If model_name is provided, only return access groups for that model.
        """
        from collections import defaultdict

        access_groups = defaultdict(list)

        model_list = self.get_model_list(model_name=model_name)
        if model_list:
            for m in model_list:
                for group in m.get("model_info", {}).get("access_groups", []):
                    model_name = m["model_name"]
                    access_groups[group].append(model_name)

        return access_groups

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
        """
        Update the router settings.
        """
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
        parent_otel_span: Optional[Span] = _get_parent_otel_span_from_kwargs(kwargs)
        if client_type == "max_parallel_requests":
            cache_key = "{}_max_parallel_requests_client".format(model_id)
            client = self.cache.get_cache(
                key=cache_key, local_only=True, parent_otel_span=parent_otel_span
            )
            return client
        elif client_type == "async":
            if kwargs.get("stream") is True:
                cache_key = f"{model_id}_stream_async_client"
                client = self.cache.get_cache(
                    key=cache_key, local_only=True, parent_otel_span=parent_otel_span
                )
                if client is None:
                    """
                    Re-initialize the client
                    """
                    InitalizeOpenAISDKClient.set_client(
                        litellm_router_instance=self, model=deployment
                    )
                    client = self.cache.get_cache(
                        key=cache_key,
                        local_only=True,
                        parent_otel_span=parent_otel_span,
                    )
                return client
            else:
                cache_key = f"{model_id}_async_client"
                client = self.cache.get_cache(
                    key=cache_key, local_only=True, parent_otel_span=parent_otel_span
                )
                if client is None:
                    """
                    Re-initialize the client
                    """
                    InitalizeOpenAISDKClient.set_client(
                        litellm_router_instance=self, model=deployment
                    )
                    client = self.cache.get_cache(
                        key=cache_key,
                        local_only=True,
                        parent_otel_span=parent_otel_span,
                    )
                return client
        else:
            if kwargs.get("stream") is True:
                cache_key = f"{model_id}_stream_client"
                client = self.cache.get_cache(
                    key=cache_key, parent_otel_span=parent_otel_span
                )
                if client is None:
                    """
                    Re-initialize the client
                    """
                    InitalizeOpenAISDKClient.set_client(
                        litellm_router_instance=self, model=deployment
                    )
                    client = self.cache.get_cache(
                        key=cache_key, parent_otel_span=parent_otel_span
                    )
                return client
            else:
                cache_key = f"{model_id}_client"
                client = self.cache.get_cache(
                    key=cache_key, parent_otel_span=parent_otel_span
                )
                if client is None:
                    """
                    Re-initialize the client
                    """
                    InitalizeOpenAISDKClient.set_client(
                        litellm_router_instance=self, model=deployment
                    )
                    client = self.cache.get_cache(
                        key=cache_key, parent_otel_span=parent_otel_span
                    )
                return client

    def _pre_call_checks(  # noqa: PLR0915
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
        parent_otel_span = _get_parent_otel_span_from_kwargs(request_kwargs)

        ## get model group RPM ##
        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        rpm_key = f"{model}:rpm:{current_minute}"
        model_group_cache = (
            self.cache.get_cache(
                key=rpm_key, local_only=True, parent_otel_span=parent_otel_span
            )
            or {}
        )  # check the in-memory cache used by lowest_latency and usage-based routing. Only check the local cache.
        for idx, deployment in enumerate(_returned_deployments):
            # see if we have the info for this model
            try:
                base_model = deployment.get("model_info", {}).get("base_model", None)
                if base_model is None:
                    base_model = deployment.get("litellm_params", {}).get(
                        "base_model", None
                    )
                model_info = self.get_router_model_info(
                    deployment=deployment, received_model_name=model
                )
                model = base_model or deployment.get("litellm_params", {}).get(
                    "model", None
                )

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
                verbose_router_logger.exception("An error occurs - {}".format(str(e)))

            _litellm_params = deployment.get("litellm_params", {})
            model_id = deployment.get("model_info", {}).get("id", "")
            ## RPM CHECK ##
            ### get local router cache ###
            current_request_cache_local = (
                self.cache.get_cache(
                    key=model_id, local_only=True, parent_otel_span=parent_otel_span
                )
                or 0
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
            ):
                allowed_model_region = request_kwargs.get("allowed_model_region")

                if allowed_model_region is not None:
                    if not is_region_allowed(
                        litellm_params=LiteLLM_Params(**_litellm_params),
                        allowed_model_region=allowed_model_region,
                    ):
                        invalid_model_indices.append(idx)
                        continue

            ## INVALID PARAMS ## -> catch 'gpt-3.5-turbo-16k' not supporting 'response_format' param
            if request_kwargs is not None and litellm.drop_params is False:
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

            if _rate_limit_error is True:  # allow generic fallback logic to take place
                raise RouterRateLimitErrorBasic(
                    model=model,
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

    def _get_model_from_alias(self, model: str) -> Optional[str]:
        """
        Get the model from the alias.

        Returns:
        - str, the litellm model name
        - None, if model is not in model group alias
        """
        if model not in self.model_group_alias:
            return None

        _item = self.model_group_alias[model]
        if isinstance(_item, str):
            model = _item
        else:
            model = _item["model"]

        return model

    def _get_deployment_by_litellm_model(self, model: str) -> List:
        """
        Get the deployment by litellm model.
        """
        return [m for m in self.model_list if m["litellm_params"]["model"] == model]

    def _common_checks_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Tuple[str, Union[List, Dict]]:
        """
        Common checks for 'get_available_deployment' across sync + async call.

        If 'healthy_deployments' returned is None, this means the user chose a specific deployment

        Returns
        - str, the litellm model name
        - List, if multiple models chosen
        - Dict, if specific model chosen
        """
        # check if aliases set on litellm model alias map
        if specific_deployment is True:
            return model, self._get_deployment_by_litellm_model(model=model)
        elif model in self.get_model_ids():
            deployment = self.get_deployment(model_id=model)
            if deployment is not None:
                deployment_model = deployment.litellm_params.model
                return deployment_model, deployment.model_dump(exclude_none=True)
            raise ValueError(
                f"LiteLLM Router: Trying to call specific deployment, but Model ID :{model} does not exist in \
                    Model ID List: {self.get_model_ids}"
            )

        _model_from_alias = self._get_model_from_alias(model=model)
        if _model_from_alias is not None:
            model = _model_from_alias

        if model not in self.model_names:
            # check if provider/ specific wildcard routing use pattern matching
            pattern_deployments = self.pattern_router.get_deployments_by_pattern(
                model=model,
            )
            if pattern_deployments:
                return model, pattern_deployments

            # check if default deployment is set
            if self.default_deployment is not None:
                updated_deployment = copy.deepcopy(
                    self.default_deployment
                )  # self.default_deployment
                updated_deployment["litellm_params"]["model"] = model
                return model, updated_deployment

        ## get healthy deployments
        ### get all deployments
        healthy_deployments = self._get_all_deployments(model_name=model)

        if len(healthy_deployments) == 0:
            # check if the user sent in a deployment name instead
            healthy_deployments = self._get_deployment_by_litellm_model(model=model)

        verbose_router_logger.debug(
            f"initial list of deployments: {healthy_deployments}"
        )

        if len(healthy_deployments) == 0:
            raise litellm.BadRequestError(
                message="You passed in model={}. There is no 'model_name' with this string ".format(
                    model
                ),
                model=model,
                llm_provider="",
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
            and self.routing_strategy != "latency-based-routing"
            and self.routing_strategy != "least-busy"
        ):  # prevent regressions for other routing strategies, that don't have async get available deployments implemented.
            return self.get_available_deployment(
                model=model,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
                request_kwargs=request_kwargs,
            )
        try:
            parent_otel_span = _get_parent_otel_span_from_kwargs(request_kwargs)
            model, healthy_deployments = self._common_checks_available_deployment(
                model=model,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
            )  # type: ignore
            if isinstance(healthy_deployments, dict):
                return healthy_deployments

            cooldown_deployments = await _async_get_cooldown_deployments(
                litellm_router_instance=self, parent_otel_span=parent_otel_span
            )
            verbose_router_logger.debug(
                f"async cooldown deployments: {cooldown_deployments}"
            )
            verbose_router_logger.debug(f"cooldown_deployments: {cooldown_deployments}")
            healthy_deployments = self._filter_cooldown_deployments(
                healthy_deployments=healthy_deployments,
                cooldown_deployments=cooldown_deployments,
            )

            healthy_deployments = await self.async_callback_filter_deployments(
                model=model,
                healthy_deployments=healthy_deployments,
                messages=(
                    cast(List[AllMessageValues], messages)
                    if messages is not None
                    else None
                ),
                request_kwargs=request_kwargs,
                parent_otel_span=parent_otel_span,
            )

            if self.enable_pre_call_checks and messages is not None:
                healthy_deployments = self._pre_call_checks(
                    model=model,
                    healthy_deployments=cast(List[Dict], healthy_deployments),
                    messages=messages,
                    request_kwargs=request_kwargs,
                )
            # check if user wants to do tag based routing
            healthy_deployments = await get_deployments_for_tag(  # type: ignore
                llm_router_instance=self,
                model=model,
                request_kwargs=request_kwargs,
                healthy_deployments=healthy_deployments,
            )

            if len(healthy_deployments) == 0:
                exception = await async_raise_no_deployment_exception(
                    litellm_router_instance=self,
                    model=model,
                    parent_otel_span=parent_otel_span,
                )
                raise exception
            start_time = time.time()
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
            elif (
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
            elif (
                self.routing_strategy == "latency-based-routing"
                and self.lowestlatency_logger is not None
            ):
                deployment = (
                    await self.lowestlatency_logger.async_get_available_deployments(
                        model_group=model,
                        healthy_deployments=healthy_deployments,  # type: ignore
                        messages=messages,
                        input=input,
                        request_kwargs=request_kwargs,
                    )
                )
            elif self.routing_strategy == "simple-shuffle":
                return simple_shuffle(
                    llm_router_instance=self,
                    healthy_deployments=healthy_deployments,
                    model=model,
                )
            elif (
                self.routing_strategy == "least-busy"
                and self.leastbusy_logger is not None
            ):
                deployment = (
                    await self.leastbusy_logger.async_get_available_deployments(
                        model_group=model,
                        healthy_deployments=healthy_deployments,  # type: ignore
                    )
                )
            else:
                deployment = None
            if deployment is None:
                exception = await async_raise_no_deployment_exception(
                    litellm_router_instance=self,
                    model=model,
                    parent_otel_span=parent_otel_span,
                )
                raise exception
            verbose_router_logger.info(
                f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment)} for model: {model}"
            )

            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.service_logger_obj.async_service_success_hook(
                    service=ServiceTypes.ROUTER,
                    duration=_duration,
                    call_type="<routing_strategy>.async_get_available_deployments",
                    parent_otel_span=parent_otel_span,
                    start_time=start_time,
                    end_time=end_time,
                )
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

        parent_otel_span: Optional[Span] = _get_parent_otel_span_from_kwargs(
            request_kwargs
        )
        cooldown_deployments = _get_cooldown_deployments(
            litellm_router_instance=self, parent_otel_span=parent_otel_span
        )
        healthy_deployments = self._filter_cooldown_deployments(
            healthy_deployments=healthy_deployments,
            cooldown_deployments=cooldown_deployments,
        )

        # filter pre-call checks
        if self.enable_pre_call_checks and messages is not None:
            healthy_deployments = self._pre_call_checks(
                model=model,
                healthy_deployments=healthy_deployments,
                messages=messages,
                request_kwargs=request_kwargs,
            )

        if len(healthy_deployments) == 0:
            model_ids = self.get_model_ids(model_name=model)
            _cooldown_time = self.cooldown_cache.get_min_cooldown(
                model_ids=model_ids, parent_otel_span=parent_otel_span
            )
            _cooldown_list = _get_cooldown_deployments(
                litellm_router_instance=self, parent_otel_span=parent_otel_span
            )
            raise RouterRateLimitError(
                model=model,
                cooldown_time=_cooldown_time,
                enable_pre_call_checks=self.enable_pre_call_checks,
                cooldown_list=_cooldown_list,
            )

        if self.routing_strategy == "least-busy" and self.leastbusy_logger is not None:
            deployment = self.leastbusy_logger.get_available_deployments(
                model_group=model, healthy_deployments=healthy_deployments  # type: ignore
            )
        elif self.routing_strategy == "simple-shuffle":
            # if users pass rpm or tpm, we do a random weighted pick - based on rpm/tpm
            ############## Check 'weight' param set for weighted pick #################
            return simple_shuffle(
                llm_router_instance=self,
                healthy_deployments=healthy_deployments,
                model=model,
            )
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
        else:
            deployment = None

        if deployment is None:
            verbose_router_logger.info(
                f"get_available_deployment for model: {model}, No deployment available"
            )
            model_ids = self.get_model_ids(model_name=model)
            _cooldown_time = self.cooldown_cache.get_min_cooldown(
                model_ids=model_ids, parent_otel_span=parent_otel_span
            )
            _cooldown_list = _get_cooldown_deployments(
                litellm_router_instance=self, parent_otel_span=parent_otel_span
            )
            raise RouterRateLimitError(
                model=model,
                cooldown_time=_cooldown_time,
                enable_pre_call_checks=self.enable_pre_call_checks,
                cooldown_list=_cooldown_list,
            )
        verbose_router_logger.info(
            f"get_available_deployment for model: {model}, Selected deployment: {self.print_deployment(deployment)} for model: {model}"
        )
        return deployment

    def _filter_cooldown_deployments(
        self, healthy_deployments: List[Dict], cooldown_deployments: List[str]
    ) -> List[Dict]:
        """
        Filters out the deployments currently cooling down from the list of healthy deployments

        Args:
            healthy_deployments: List of healthy deployments
            cooldown_deployments: List of model_ids cooling down. cooldown_deployments is a list of model_id's cooling down, cooldown_deployments = ["16700539-b3cd-42f4-b426-6a12a1bb706a", "16700539-b3cd-42f4-b426-7899"]

        Returns:
            List of healthy deployments
        """
        # filter out the deployments currently cooling down
        deployments_to_remove = []
        verbose_router_logger.debug(f"cooldown deployments: {cooldown_deployments}")
        # Find deployments in model_list whose model_id is cooling down
        for deployment in healthy_deployments:
            deployment_id = deployment["model_info"]["id"]
            if deployment_id in cooldown_deployments:
                deployments_to_remove.append(deployment)

        # remove unhealthy deployments from healthy deployments
        for deployment in deployments_to_remove:
            healthy_deployments.remove(deployment)
        return healthy_deployments

    def _track_deployment_metrics(
        self, deployment, parent_otel_span: Optional[Span], response=None
    ):
        """
        Tracks successful requests rpm usage.
        """
        try:
            model_id = deployment.get("model_info", {}).get("id", None)
            if response is None:

                # update self.deployment_stats
                if model_id is not None:
                    self._update_usage(
                        model_id, parent_otel_span
                    )  # update in-memory cache for tracking
        except Exception as e:
            verbose_router_logger.error(f"Error in _track_deployment_metrics: {str(e)}")

    def get_num_retries_from_retry_policy(
        self, exception: Exception, model_group: Optional[str] = None
    ):
        return _get_num_retries_from_retry_policy(
            exception=exception,
            model_group=model_group,
            model_group_retry_policy=self.model_group_retry_policy,
            retry_policy=self.retry_policy,
        )

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
        from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting

        if self.alerting_config is None:
            return

        router_alerting_config: AlertingConfig = self.alerting_config

        _slack_alerting_logger = SlackAlerting(
            alerting_threshold=router_alerting_config.alerting_threshold,
            alerting=["slack"],
            default_webhook_url=router_alerting_config.webhook_url,
        )

        self.slack_alerting_logger = _slack_alerting_logger

        litellm.callbacks.append(_slack_alerting_logger)  # type: ignore
        litellm.success_callback.append(
            _slack_alerting_logger.response_taking_too_long_callback
        )
        verbose_router_logger.info(
            "\033[94m\nInitialized Alerting for litellm.Router\033[0m\n"
        )

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

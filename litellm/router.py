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
from collections import defaultdict
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
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
from litellm._uuid import uuid
from litellm.caching.caching import (
    DualCache,
    InMemoryCache,
    RedisCache,
    RedisClusterCache,
)
from litellm.constants import DEFAULT_MAX_LRU_CACHE_SIZE
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,
    get_metadata_variable_name_from_kwargs,
)
from litellm.litellm_core_utils.coroutine_checker import coroutine_checker
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler
from litellm.router_strategy.lowest_cost import LowestCostLoggingHandler
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.router_strategy.simple_shuffle import simple_shuffle
from litellm.router_strategy.tag_based_routing import get_deployments_for_tag
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
)
from litellm.router_utils.batch_utils import (
    _get_router_metadata_variable_name,
    replace_model_in_jsonl,
    should_replace_model_in_jsonl,
)
from litellm.router_utils.client_initalization_utils import InitalizeCachedClient
from litellm.router_utils.clientside_credential_handler import (
    get_dynamic_litellm_params,
    is_clientside_credential,
)
from litellm.router_utils.common_utils import (
    filter_team_based_models,
    filter_web_search_deployments,
)
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
from litellm.router_utils.pre_call_checks.responses_api_deployment_check import (
    ResponsesApiDeploymentCheck,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    increment_deployment_failures_for_current_minute,
    increment_deployment_successes_for_current_minute,
)
from litellm.scheduler import FlowItem, Scheduler
from litellm.types.llms.openai import (
    AllMessageValues,
    FileTypes,
    OpenAIFileObject,
    OpenAIFilesPurpose,
)
from litellm.types.router import (
    CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS,
    VALID_LITELLM_ENVIRONMENTS,
    AlertingConfig,
    AllowedFailsPolicy,
    AssistantsTypedDict,
    CredentialLiteLLMParams,
    CustomRoutingStrategyBase,
    Deployment,
    DeploymentTypedDict,
    LiteLLM_Params,
    MockRouterTestingParams,
    ModelGroupInfo,
    OptionalPreCallChecks,
    RetryPolicy,
    RouterCacheEnum,
    RouterGeneralSettings,
    RouterModelGroupAliasItem,
    RouterRateLimitError,
    RouterRateLimitErrorBasic,
    RoutingStrategy,
    SearchToolTypedDict,
)
from litellm.types.services import ServiceTypes
from litellm.types.utils import (
    CustomPricingLiteLLMParams,
    GenericBudgetConfigType,
    LiteLLMBatch,
)
from litellm.types.utils import ModelInfo
from litellm.types.utils import ModelInfo as ModelMapInfo
from litellm.types.utils import (
    ModelResponseStream,
    StandardLoggingPayload,
    Usage,
)
from litellm.utils import (
    CustomStreamWrapper,
    EmbeddingResponse,
    ModelResponse,
    Rules,
    function_setup,
    get_llm_provider,
    get_non_default_completion_params,
    get_secret,
    get_utc_datetime,
    is_region_allowed,
)

from .router_utils.pattern_match_deployments import PatternMatchRouter

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.router_strategy.auto_router.auto_router import (
        AutoRouter,
        PreRoutingHookResponse,
    )

    Span = Union[_Span, Any]
else:
    Span = Any
    AutoRouter = Any
    PreRoutingHookResponse = Any


class RoutingArgs(enum.Enum):
    ttl = 60  # 1min (RPM/TPM expire key)


class Router:
    model_names: set = set()
    cache_responses: Optional[bool] = False
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour
    tenacity = None
    leastbusy_logger: Optional[LeastBusyLoggingHandler] = None
    lowesttpm_logger: Optional[LowestTPMLoggingHandler] = None
    optional_callbacks: Optional[List[Union[CustomLogger, Callable, str]]] = None

    def __init__(  # noqa: PLR0915
        self,
        model_list: Optional[
            Union[List[DeploymentTypedDict], List[Dict[str, Any]]]
        ] = None,
        ## ASSISTANTS API ##
        assistants_config: Optional[AssistantsTypedDict] = None,
        ## SEARCH API ##
        search_tools: Optional[List[SearchToolTypedDict]] = None,
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
        stream_timeout: Optional[float] = None,
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
        ignore_invalid_deployments: bool = False,
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
            ignore_invalid_deployments (bool): Ignores invalid deployments, and continues with other deployments. Default is to raise an error.
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

        self.set_verbose = set_verbose
        self.ignore_invalid_deployments = ignore_invalid_deployments
        self.debug_level = debug_level
        self.enable_pre_call_checks = enable_pre_call_checks
        self.enable_tag_filtering = enable_tag_filtering
        from litellm._service_logger import ServiceLogging

        self.service_logger_obj: ServiceLogging = ServiceLogging()
        litellm.suppress_debug_info = True  # prevents 'Give Feedback/Get help' message from being emitted on Router - Relevant Issue: https://github.com/BerriAI/litellm/issues/5942
        if self.set_verbose is True:
            if debug_level == "INFO":
                verbose_router_logger.setLevel(logging.INFO)
            elif debug_level == "DEBUG":
                verbose_router_logger.setLevel(logging.DEBUG)
        self.router_general_settings: RouterGeneralSettings = (
            router_general_settings or RouterGeneralSettings()
        )

        self.assistants_config = assistants_config
        self.search_tools = search_tools or []
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
            redis_cache = self._create_redis_cache(cache_config)

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
        self.team_pattern_routers: Dict[str, PatternMatchRouter] = (
            {}
        )  # {"TEAM_ID": PatternMatchRouter}
        self.auto_routers: Dict[str, "AutoRouter"] = {}

        # Initialize model_group_alias early since it's used in set_model_list
        self.model_group_alias: Dict[str, Union[str, RouterModelGroupAliasItem]] = (
            model_group_alias or {}
        )  # dict to store aliases for router, ex. {"gpt-4": "gpt-3.5-turbo"}, all requests with gpt-4 -> get routed to gpt-3.5-turbo group

        # Initialize model ID to deployment index mapping for O(1) lookups
        self.model_id_to_deployment_index_map: Dict[str, int] = {}
        # Initialize model name to deployment indices mapping for O(1) lookups
        # Maps model_name -> list of indices in model_list
        self.model_name_to_deployment_indices: Dict[str, List[int]] = {}

        if model_list is not None:
            # set_model_list will build indices automatically
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
        self.stream_timeout = stream_timeout

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
            litellm.logging_callback_manager.add_litellm_async_success_callback(
                self.deployment_callback_on_success
            )
        else:
            litellm.logging_callback_manager.add_litellm_async_success_callback(
                self.deployment_callback_on_success
            )
        if isinstance(litellm.success_callback, list):
            litellm.logging_callback_manager.add_litellm_success_callback(
                self.sync_deployment_callback_on_success
            )
        else:
            litellm.success_callback = [self.sync_deployment_callback_on_success]
        if isinstance(litellm._async_failure_callback, list):
            litellm.logging_callback_manager.add_litellm_async_failure_callback(
                self.async_deployment_callback_on_failure
            )
        else:
            litellm._async_failure_callback = [
                self.async_deployment_callback_on_failure
            ]
        ## COOLDOWNS ##
        if isinstance(litellm.failure_callback, list):
            litellm.logging_callback_manager.add_litellm_failure_callback(
                self.deployment_callback_on_failure
            )
        else:
            litellm.failure_callback = [self.deployment_callback_on_failure]
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
        self.initialize_router_endpoints()
        self.apply_default_settings()

    def apply_default_settings(self):
        """
        Apply the default settings to the router.
        """

        default_pre_call_checks: OptionalPreCallChecks = []
        self.add_optional_pre_call_checks(default_pre_call_checks)
        return None

    def discard(self):
        """
        Pseudo-destructor to be invoked to clean up global data structures when router is no longer used.
        For now, unhook router's callbacks from all lists
        """
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm._async_success_callback, self
        )
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm.success_callback, self
        )
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm._async_failure_callback, self
        )
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm.failure_callback, self
        )
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm.input_callback, self
        )
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm.service_callback, self
        )
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm.callbacks, self
        )

        # Remove ForwardClientSideHeadersByModelGroup if it exists
        if self.optional_callbacks is not None:
            for callback in self.optional_callbacks:
                litellm.logging_callback_manager.remove_callback_from_list_by_object(
                    litellm.callbacks, callback, require_self=False
                )

    @staticmethod
    def _create_redis_cache(
        cache_config: Dict[str, Any],
    ) -> Union[RedisCache, RedisClusterCache]:
        """
        Initializes either a RedisCache or RedisClusterCache based on the cache_config.
        """
        if cache_config.get("startup_nodes"):
            return RedisClusterCache(**cache_config)
        else:
            return RedisCache(**cache_config)

    def _update_redis_cache(self, cache: RedisCache):
        """
        Update the redis cache for the router, if none set.

        Allows proxy user to just do
        ```yaml
        litellm_settings:
            cache: true
        ```
        and caching to just work.
        """
        if self.cache.redis_cache is None:
            self.cache.redis_cache = cache

    def routing_strategy_init(
        self, routing_strategy: Union[RoutingStrategy, str], routing_strategy_args: dict
    ):
        verbose_router_logger.info(f"Routing strategy: {routing_strategy}")
        if (
            routing_strategy == RoutingStrategy.LEAST_BUSY.value
            or routing_strategy == RoutingStrategy.LEAST_BUSY
        ):
            self.leastbusy_logger = LeastBusyLoggingHandler(router_cache=self.cache)
            ## add callback
            if isinstance(litellm.input_callback, list):
                litellm.input_callback.append(self.leastbusy_logger)  # type: ignore
            else:
                litellm.input_callback = [self.leastbusy_logger]  # type: ignore
            if isinstance(litellm.callbacks, list):
                litellm.logging_callback_manager.add_litellm_callback(self.leastbusy_logger)  # type: ignore
        elif (
            routing_strategy == RoutingStrategy.USAGE_BASED_ROUTING.value
            or routing_strategy == RoutingStrategy.USAGE_BASED_ROUTING
        ):
            self.lowesttpm_logger = LowestTPMLoggingHandler(
                router_cache=self.cache,
                routing_args=routing_strategy_args,
            )
            if isinstance(litellm.callbacks, list):
                litellm.logging_callback_manager.add_litellm_callback(self.lowesttpm_logger)  # type: ignore
        elif (
            routing_strategy == RoutingStrategy.USAGE_BASED_ROUTING_V2.value
            or routing_strategy == RoutingStrategy.USAGE_BASED_ROUTING_V2
        ):
            self.lowesttpm_logger_v2 = LowestTPMLoggingHandler_v2(
                router_cache=self.cache,
                routing_args=routing_strategy_args,
            )
            if isinstance(litellm.callbacks, list):
                litellm.logging_callback_manager.add_litellm_callback(self.lowesttpm_logger_v2)  # type: ignore
        elif (
            routing_strategy == RoutingStrategy.LATENCY_BASED.value
            or routing_strategy == RoutingStrategy.LATENCY_BASED
        ):
            self.lowestlatency_logger = LowestLatencyLoggingHandler(
                router_cache=self.cache,
                routing_args=routing_strategy_args,
            )
            if isinstance(litellm.callbacks, list):
                litellm.logging_callback_manager.add_litellm_callback(self.lowestlatency_logger)  # type: ignore
        elif (
            routing_strategy == RoutingStrategy.COST_BASED.value
            or routing_strategy == RoutingStrategy.COST_BASED
        ):
            self.lowestcost_logger = LowestCostLoggingHandler(
                router_cache=self.cache,
                routing_args={},
            )
            if isinstance(litellm.callbacks, list):
                litellm.logging_callback_manager.add_litellm_callback(self.lowestcost_logger)  # type: ignore
        else:
            pass

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

    def _initialize_core_endpoints(self):
        """Helper to initialize core router endpoints."""
        self.amoderation = self.factory_function(
            litellm.amoderation, call_type="moderation"
        )
        self.aanthropic_messages = self.factory_function(
            litellm.anthropic_messages, call_type="anthropic_messages"
        )
        self.anthropic_messages = self.factory_function(
            litellm.anthropic_messages, call_type="anthropic_messages"
        )
        self.agenerate_content = self.factory_function(
            litellm.agenerate_content, call_type="agenerate_content"
        )
        self.aadapter_generate_content = self.factory_function(
            litellm.aadapter_generate_content, call_type="aadapter_generate_content"
        )
        self.aresponses = self.factory_function(
            litellm.aresponses, call_type="aresponses"
        )
        self.afile_delete = self.factory_function(
            litellm.afile_delete, call_type="afile_delete"
        )
        self.afile_content = self.factory_function(
            litellm.afile_content, call_type="afile_content"
        )
        self.responses = self.factory_function(litellm.responses, call_type="responses")
        self.aget_responses = self.factory_function(
            litellm.aget_responses, call_type="aget_responses"
        )
        self.acancel_responses = self.factory_function(
            litellm.acancel_responses, call_type="acancel_responses"
        )
        self.adelete_responses = self.factory_function(
            litellm.adelete_responses, call_type="adelete_responses"
        )
        self.alist_input_items = self.factory_function(
            litellm.alist_input_items, call_type="alist_input_items"
        )
        self._arealtime = self.factory_function(
            litellm._arealtime, call_type="_arealtime"
        )
        self.acreate_fine_tuning_job = self.factory_function(
            litellm.acreate_fine_tuning_job, call_type="acreate_fine_tuning_job"
        )
        self.acancel_fine_tuning_job = self.factory_function(
            litellm.acancel_fine_tuning_job, call_type="acancel_fine_tuning_job"
        )
        self.alist_fine_tuning_jobs = self.factory_function(
            litellm.alist_fine_tuning_jobs, call_type="alist_fine_tuning_jobs"
        )
        self.aretrieve_fine_tuning_job = self.factory_function(
            litellm.aretrieve_fine_tuning_job, call_type="aretrieve_fine_tuning_job"
        )
        self.afile_list = self.factory_function(
            litellm.afile_list, call_type="alist_files"
        )
        self.aimage_edit = self.factory_function(
            litellm.aimage_edit, call_type="aimage_edit"
        )
        self.allm_passthrough_route = self.factory_function(
            litellm.allm_passthrough_route, call_type="allm_passthrough_route"
        )
        self.acancel_batch = self.factory_function(
            litellm.acancel_batch, call_type="acancel_batch"
        )

    def _initialize_vector_store_endpoints(self):
        """Initialize vector store endpoints."""
        from litellm.vector_stores.main import acreate, asearch, create, search

        self.avector_store_search = self.factory_function(
            asearch, call_type="avector_store_search"
        )
        self.avector_store_create = self.factory_function(
            acreate, call_type="avector_store_create"
        )
        self.vector_store_search = self.factory_function(
            search, call_type="vector_store_search"
        )
        self.vector_store_create = self.factory_function(
            create, call_type="vector_store_create"
        )

    def _initialize_vector_store_file_endpoints(self):
        """Initialize vector store file endpoints."""
        from litellm.vector_store_files.main import (
            acreate as avector_store_file_create_fn,
        )
        from litellm.vector_store_files.main import (
            adelete as avector_store_file_delete_fn,
        )
        from litellm.vector_store_files.main import alist as avector_store_file_list_fn
        from litellm.vector_store_files.main import (
            aretrieve as avector_store_file_retrieve_fn,
        )
        from litellm.vector_store_files.main import (
            aretrieve_content as avector_store_file_content_fn,
        )
        from litellm.vector_store_files.main import (
            aupdate as avector_store_file_update_fn,
        )
        from litellm.vector_store_files.main import (
            create as vector_store_file_create_fn,
        )
        from litellm.vector_store_files.main import (
            delete as vector_store_file_delete_fn,
        )
        from litellm.vector_store_files.main import list as vector_store_file_list_fn
        from litellm.vector_store_files.main import (
            retrieve as vector_store_file_retrieve_fn,
        )
        from litellm.vector_store_files.main import (
            retrieve_content as vector_store_file_content_fn,
        )
        from litellm.vector_store_files.main import (
            update as vector_store_file_update_fn,
        )

        self.avector_store_file_create = self.factory_function(
            avector_store_file_create_fn, call_type="avector_store_file_create"
        )
        self.vector_store_file_create = self.factory_function(
            vector_store_file_create_fn, call_type="vector_store_file_create"
        )
        self.avector_store_file_list = self.factory_function(
            avector_store_file_list_fn, call_type="avector_store_file_list"
        )
        self.vector_store_file_list = self.factory_function(
            vector_store_file_list_fn, call_type="vector_store_file_list"
        )
        self.avector_store_file_retrieve = self.factory_function(
            avector_store_file_retrieve_fn, call_type="avector_store_file_retrieve"
        )
        self.vector_store_file_retrieve = self.factory_function(
            vector_store_file_retrieve_fn, call_type="vector_store_file_retrieve"
        )
        self.avector_store_file_content = self.factory_function(
            avector_store_file_content_fn, call_type="avector_store_file_content"
        )
        self.vector_store_file_content = self.factory_function(
            vector_store_file_content_fn, call_type="vector_store_file_content"
        )
        self.avector_store_file_update = self.factory_function(
            avector_store_file_update_fn, call_type="avector_store_file_update"
        )
        self.vector_store_file_update = self.factory_function(
            vector_store_file_update_fn, call_type="vector_store_file_update"
        )
        self.avector_store_file_delete = self.factory_function(
            avector_store_file_delete_fn, call_type="avector_store_file_delete"
        )
        self.vector_store_file_delete = self.factory_function(
            vector_store_file_delete_fn, call_type="vector_store_file_delete"
        )

    def _initialize_google_genai_endpoints(self):
        """Initialize Google GenAI endpoints."""
        from litellm.google_genai import (
            agenerate_content,
            agenerate_content_stream,
            generate_content,
            generate_content_stream,
        )

        self.agenerate_content = self.factory_function(
            agenerate_content, call_type="agenerate_content"
        )
        self.generate_content = self.factory_function(
            generate_content, call_type="generate_content"
        )
        self.agenerate_content_stream = self.factory_function(
            agenerate_content_stream, call_type="agenerate_content_stream"
        )
        self.generate_content_stream = self.factory_function(
            generate_content_stream, call_type="generate_content_stream"
        )

    def _initialize_ocr_search_endpoints(self):
        """Initialize OCR and search endpoints."""
        from litellm.ocr import aocr, ocr

        self.aocr = self.factory_function(aocr, call_type="aocr")
        self.ocr = self.factory_function(ocr, call_type="ocr")

        from litellm.search import asearch, search

        self.asearch = self.factory_function(asearch, call_type="asearch")
        self.search = self.factory_function(search, call_type="search")

    def _initialize_video_endpoints(self):
        """Initialize video endpoints."""
        from litellm.videos import (
            avideo_content,
            avideo_generation,
            avideo_list,
            avideo_remix,
            avideo_status,
            video_content,
            video_generation,
            video_list,
            video_remix,
            video_status,
        )

        self.avideo_generation = self.factory_function(
            avideo_generation, call_type="avideo_generation"
        )
        self.video_generation = self.factory_function(
            video_generation, call_type="video_generation"
        )
        self.avideo_list = self.factory_function(avideo_list, call_type="avideo_list")
        self.video_list = self.factory_function(video_list, call_type="video_list")
        self.avideo_status = self.factory_function(
            avideo_status, call_type="avideo_status"
        )
        self.video_status = self.factory_function(
            video_status, call_type="video_status"
        )
        self.avideo_content = self.factory_function(
            avideo_content, call_type="avideo_content"
        )
        self.video_content = self.factory_function(
            video_content, call_type="video_content"
        )
        self.avideo_remix = self.factory_function(
            avideo_remix, call_type="avideo_remix"
        )
        self.video_remix = self.factory_function(video_remix, call_type="video_remix")

    def _initialize_container_endpoints(self):
        """Initialize container endpoints."""
        from litellm.containers import (
            acreate_container,
            adelete_container,
            alist_containers,
            aretrieve_container,
            create_container,
            delete_container,
            list_containers,
            retrieve_container,
        )
        from litellm.containers.endpoint_factory import (
            _generated_endpoints as container_file_endpoints,
        )

        self.acreate_container = self.factory_function(
            acreate_container, call_type="acreate_container"
        )
        self.create_container = self.factory_function(
            create_container, call_type="create_container"
        )
        self.alist_containers = self.factory_function(
            alist_containers, call_type="alist_containers"
        )
        self.list_containers = self.factory_function(
            list_containers, call_type="list_containers"
        )
        self.aretrieve_container = self.factory_function(
            aretrieve_container, call_type="aretrieve_container"
        )
        self.retrieve_container = self.factory_function(
            retrieve_container, call_type="retrieve_container"
        )
        self.adelete_container = self.factory_function(
            adelete_container, call_type="adelete_container"
        )
        self.delete_container = self.factory_function(
            delete_container, call_type="delete_container"
        )
        
        # Auto-register JSON-generated container file endpoints
        for name, func in container_file_endpoints.items():
            setattr(self, name, self.factory_function(func, call_type=name))  # type: ignore[arg-type]

    def _initialize_skills_endpoints(self):
        """Initialize Anthropic Skills API endpoints."""
        self.acreate_skill = self.factory_function(
            litellm.acreate_skill, call_type="acreate_skill"
        )
        self.alist_skills = self.factory_function(
            litellm.alist_skills, call_type="alist_skills"
        )
        self.aget_skill = self.factory_function(
            litellm.aget_skill, call_type="aget_skill"
        )
        self.adelete_skill = self.factory_function(
            litellm.adelete_skill, call_type="adelete_skill"
        )

    def _initialize_interactions_endpoints(self):
        """Initialize Google Interactions API endpoints."""
        from litellm.interactions import acancel as acancel_interaction
        from litellm.interactions import acreate as acreate_interaction
        from litellm.interactions import adelete as adelete_interaction
        from litellm.interactions import aget as aget_interaction
        from litellm.interactions import cancel as cancel_interaction
        from litellm.interactions import create as create_interaction
        from litellm.interactions import delete as delete_interaction
        from litellm.interactions import get as get_interaction

        self.acreate_interaction = self.factory_function(
            acreate_interaction, call_type="acreate_interaction"
        )
        self.create_interaction = self.factory_function(
            create_interaction, call_type="create_interaction"
        )
        self.aget_interaction = self.factory_function(
            aget_interaction, call_type="aget_interaction"
        )
        self.get_interaction = self.factory_function(
            get_interaction, call_type="get_interaction"
        )
        self.adelete_interaction = self.factory_function(
            adelete_interaction, call_type="adelete_interaction"
        )
        self.delete_interaction = self.factory_function(
            delete_interaction, call_type="delete_interaction"
        )
        self.acancel_interaction = self.factory_function(
            acancel_interaction, call_type="acancel_interaction"
        )
        self.cancel_interaction = self.factory_function(
            cancel_interaction, call_type="cancel_interaction"
        )

    def _initialize_specialized_endpoints(self):
        """Helper to initialize specialized router endpoints (vector store, OCR, search, video, container, skills, interactions)."""
        self._initialize_vector_store_endpoints()
        self._initialize_vector_store_file_endpoints()
        self._initialize_google_genai_endpoints()
        self._initialize_ocr_search_endpoints()
        self._initialize_video_endpoints()
        self._initialize_container_endpoints()
        self._initialize_skills_endpoints()
        self._initialize_interactions_endpoints()

    def initialize_router_endpoints(self):
        self._initialize_core_endpoints()
        self._initialize_specialized_endpoints()

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
                elif pre_call_check == "responses_api_deployment_check":
                    _callback = ResponsesApiDeploymentCheck()
                if _callback is not None:
                    if self.optional_callbacks is None:
                        self.optional_callbacks = []
                    self.optional_callbacks.append(_callback)
                    litellm.logging_callback_manager.add_litellm_callback(_callback)

    def print_deployment(self, deployment: dict):
        """
        returns a copy of the deployment with the api key masked

        Only returns 2 characters of the api key and masks the rest with * (10 *).
        """
        try:
            _deployment_copy = copy.deepcopy(deployment)
            litellm_params: dict = _deployment_copy["litellm_params"]

            if litellm.redact_user_api_key_info:
                masker = SensitiveDataMasker(visible_prefix=2, visible_suffix=0)
                _deployment_copy["litellm_params"] = masker.mask_dict(litellm_params)
            elif "api_key" in litellm_params:
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
                request_kwargs=kwargs,
            )
            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

            # No copy needed - data is only read and spread into new dict below
            data = deployment["litellm_params"]
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
            if not self.has_model_id(model):
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
        self, model: str, messages: List[AllMessageValues], stream: Literal[True], **kwargs
    ) -> CustomStreamWrapper: 
        ...

    @overload
    async def acompletion(
        self, model: str, messages: List[AllMessageValues], stream: Literal[False] = False, **kwargs
    ) -> ModelResponse: 
        ...

    @overload
    async def acompletion(
        self, model: str, messages: List[AllMessageValues], stream: Union[Literal[True], Literal[False]] = False, **kwargs
    ) -> Union[CustomStreamWrapper, ModelResponse]: 
        ...

    # fmt: on

    # The actual implementation of the function
    async def acompletion(
        self,
        model: str,
        messages: List[AllMessageValues],
        stream: bool = False,
        **kwargs,
    ):
        try:
            kwargs["model"] = model
            kwargs["messages"] = messages
            kwargs["stream"] = stream
            kwargs["original_function"] = self._acompletion

            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)
            request_priority = kwargs.get("priority") or self.default_priority
            start_time = time.time()
            _is_prompt_management_model = self._is_prompt_management_model(model)

            if _is_prompt_management_model:
                return await self._prompt_management_factory(
                    model=model,
                    messages=messages,
                    kwargs=kwargs,
                )
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

    async def _acompletion_streaming_iterator(
        self,
        model_response: CustomStreamWrapper,
        messages: List[Dict[str, str]],
        initial_kwargs: dict,
    ) -> CustomStreamWrapper:
        """
        Helper to iterate over a streaming response.

        Catches errors for fallbacks using the router's fallback system
        """
        from litellm.exceptions import MidStreamFallbackError

        class FallbackStreamWrapper(CustomStreamWrapper):
            def __init__(self, async_generator: AsyncGenerator):
                # Copy attributes from the original model_response
                super().__init__(
                    completion_stream=async_generator,
                    model=model_response.model,
                    custom_llm_provider=model_response.custom_llm_provider,
                    logging_obj=model_response.logging_obj,
                )
                self._async_generator = async_generator

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await self._async_generator.__anext__()

        async def stream_with_fallbacks():
            try:
                async for item in model_response:
                    yield item
            except MidStreamFallbackError as e:
                from litellm.main import stream_chunk_builder

                complete_response_object = stream_chunk_builder(
                    chunks=model_response.chunks
                )
                complete_response_object_usage = cast(
                    Optional[Usage],
                    getattr(complete_response_object, "usage", None),
                )
                try:
                    # Use the router's fallback system
                    model_group = cast(str, initial_kwargs.get("model"))
                    fallbacks: Optional[List] = initial_kwargs.get(
                        "fallbacks", self.fallbacks
                    )
                    context_window_fallbacks: Optional[List] = initial_kwargs.get(
                        "context_window_fallbacks", self.context_window_fallbacks
                    )
                    content_policy_fallbacks: Optional[List] = initial_kwargs.get(
                        "content_policy_fallbacks", self.content_policy_fallbacks
                    )
                    initial_kwargs["original_function"] = self._acompletion
                    initial_kwargs["messages"] = messages + [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. You are given a message and you need to respond to it. You are also given a generated content. You need to respond to the message in continuation of the generated content. Do not repeat the same content. Your response should be in continuation of this text: ",
                        },
                        {
                            "role": "assistant",
                            "content": e.generated_content,
                            "prefix": True,
                        },
                    ]
                    self._update_kwargs_before_fallbacks(
                        model=model_group, kwargs=initial_kwargs
                    )
                    fallback_response = (
                        await self.async_function_with_fallbacks_common_utils(
                            e=e,
                            disable_fallbacks=False,
                            fallbacks=fallbacks,
                            context_window_fallbacks=context_window_fallbacks,
                            content_policy_fallbacks=content_policy_fallbacks,
                            model_group=model_group,
                            args=(),
                            kwargs=initial_kwargs,
                        )
                    )

                    # If fallback returns a streaming response, iterate over it
                    if hasattr(fallback_response, "__aiter__"):
                        async for fallback_item in fallback_response:  # type: ignore
                            if (
                                fallback_item
                                and isinstance(fallback_item, ModelResponseStream)
                                and hasattr(fallback_item, "usage")
                            ):
                                from litellm.cost_calculator import (
                                    BaseTokenUsageProcessor,
                                )

                                usage = cast(
                                    Optional[Usage],
                                    getattr(fallback_item, "usage", None),
                                )
                                if usage is not None:
                                    usage_objects = [usage]
                                else:
                                    usage_objects = []

                                if (
                                    complete_response_object_usage is not None
                                    and hasattr(complete_response_object_usage, "usage")
                                    and complete_response_object_usage.usage is not None  # type: ignore
                                ):
                                    usage_objects.append(complete_response_object_usage)

                                combined_usage = (
                                    BaseTokenUsageProcessor.combine_usage_objects(
                                        usage_objects=usage_objects
                                    )
                                )
                                setattr(fallback_item, "usage", combined_usage)
                            yield fallback_item
                    else:
                        # If fallback returns a non-streaming response, yield None
                        yield None

                except Exception as fallback_error:
                    # If fallback also fails, log and re-raise original error
                    verbose_router_logger.error(
                        f"Fallback also failed: {fallback_error}"
                    )
                    raise fallback_error

        return FallbackStreamWrapper(stream_with_fallbacks())

    async def _acompletion(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Union[
        ModelResponse,
        CustomStreamWrapper,
    ]:
        """
        - Get an available deployment
        - call it with a semaphore over the call
        - semaphore specific to it's rpm
        - in the semaphore,  make a check against it's local rpm before running
        """
        model_name = None
        _timeout_debug_deployment_dict = (
            {}
        )  # this is a temporary dict to debug timeout issues
        try:
            input_kwargs_for_streaming_fallback = kwargs.copy()
            input_kwargs_for_streaming_fallback["model"] = model

            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            start_time = time.time()
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=messages,
                specific_deployment=kwargs.pop("specific_deployment", None),
                request_kwargs=kwargs,
            )

            _timeout_debug_deployment_dict = deployment
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
            # No copy needed - data is only read and spread into new dict below
            data = deployment["litellm_params"]

            model_name = data["model"]

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            input_kwargs = {
                **data,
                "messages": messages,
                "caching": self.cache_responses,
                "client": model_client,
                **kwargs,
            }

            _response = litellm.acompletion(**input_kwargs)

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

            if isinstance(response, CustomStreamWrapper):
                return await self._acompletion_streaming_iterator(
                    model_response=response,
                    messages=messages,
                    initial_kwargs=input_kwargs_for_streaming_fallback,
                )

            return response
        except litellm.Timeout as e:
            deployment_request_timeout_param = _timeout_debug_deployment_dict.get(
                "litellm_params", {}
            ).get("request_timeout", None)
            deployment_timeout_param = _timeout_debug_deployment_dict.get(
                "litellm_params", {}
            ).get("timeout", None)
            e.message += f"\n\nDeployment Info: request_timeout: {deployment_request_timeout_param}\ntimeout: {deployment_timeout_param}"
            raise e
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.acompletion(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    def _update_kwargs_before_fallbacks(
        self,
        model: str,
        kwargs: dict,
        metadata_variable_name: Optional[str] = "metadata",
    ) -> None:
        """
        Adds/updates to kwargs:
        - num_retries
        - litellm_trace_id
        - metadata
        """
        kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
        kwargs.setdefault("litellm_trace_id", str(uuid.uuid4()))
        model_group_alias: Optional[str] = None
        if self._get_model_from_alias(model=model):
            model_group_alias = model
        kwargs.setdefault(metadata_variable_name, {}).update(
            {"model_group": model, "model_group_alias": model_group_alias}
        )

    def _update_kwargs_with_default_litellm_params(
        self, kwargs: dict, metadata_variable_name: Optional[str] = "metadata"
    ) -> None:
        """
        Adds default litellm params to kwargs, if set.

        Handles inserting this as either "metadata" or "litellm_metadata" depending on the metadata_variable_name
        """
        # 1) copy your defaults and pull out metadata
        defaults = self.default_litellm_params.copy()
        metadata_defaults = defaults.pop("metadata", {}) or {}

        # 2) add any non-metadata defaults that aren't already in kwargs
        for key, value in defaults.items():
            if value is None:
                continue
            kwargs.setdefault(key, value)

        # 3) merge in metadata, this handles inserting this as either "metadata" or "litellm_metadata"
        kwargs.setdefault(metadata_variable_name, {}).update(metadata_defaults)

    def _handle_clientside_credential(
        self, deployment: dict, kwargs: dict, function_name: Optional[str] = None
    ) -> Deployment:
        """
        Handle clientside credential
        """
        model_info = deployment.get("model_info", {}).copy()
        litellm_params = deployment["litellm_params"].copy()
        dynamic_litellm_params = get_dynamic_litellm_params(
            litellm_params=litellm_params, request_kwargs=kwargs
        )
        # Use deployment model_name as model_group for generating model_id
        metadata_variable_name = _get_router_metadata_variable_name(
            function_name=function_name,
        )
        model_group = kwargs.get(metadata_variable_name, {}).get("model_group")
        _model_id = self._generate_model_id(
            model_group=model_group, litellm_params=dynamic_litellm_params
        )
        original_model_id = model_info.get("id")
        model_info["id"] = _model_id
        model_info["original_model_id"] = original_model_id
        deployment_pydantic_obj = Deployment(
            model_name=model_group,
            litellm_params=LiteLLM_Params(**dynamic_litellm_params),
            model_info=model_info,
        )
        self.upsert_deployment(
            deployment=deployment_pydantic_obj
        )  # add new deployment to router
        return deployment_pydantic_obj

    def _update_kwargs_with_deployment(
        self,
        deployment: dict,
        kwargs: dict,
        function_name: Optional[str] = None,
    ) -> None:
        """
        2 jobs:
        - Adds selected deployment, model_info and api_base to kwargs["metadata"] (used for logging)
        - Adds default litellm params to kwargs, if set.
        """
        model_info = deployment.get("model_info", {}).copy()
        deployment_litellm_model_name = deployment["litellm_params"]["model"]
        deployment_api_base = deployment["litellm_params"].get("api_base")
        deployment_model_name = deployment["model_name"]
        if is_clientside_credential(request_kwargs=kwargs):
            deployment_pydantic_obj = self._handle_clientside_credential(
                deployment=deployment, kwargs=kwargs, function_name=function_name
            )
            model_info = deployment_pydantic_obj.model_info.model_dump()
            deployment_litellm_model_name = deployment_pydantic_obj.litellm_params.model
            deployment_api_base = deployment_pydantic_obj.litellm_params.api_base

        metadata_variable_name = _get_router_metadata_variable_name(
            function_name=function_name,
        )

        kwargs.setdefault(metadata_variable_name, {}).update(
            {
                "deployment": deployment_litellm_model_name,
                "model_info": model_info,
                "api_base": deployment_api_base,
                "deployment_model_name": deployment_model_name,
            }
        )
        kwargs["model_info"] = model_info

        kwargs["timeout"] = self._get_timeout(
            kwargs=kwargs, data=deployment["litellm_params"]
        )

        self._update_kwargs_with_default_litellm_params(
            kwargs=kwargs, metadata_variable_name=metadata_variable_name
        )

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

    def _get_stream_timeout(
        self, kwargs: dict, data: dict
    ) -> Optional[Union[float, int]]:
        """Helper to get stream timeout from kwargs or deployment params"""
        return (
            kwargs.get("stream_timeout", None)  # the params dynamically set by user
            or data.get(
                "stream_timeout", None
            )  # timeout set on litellm_params for this deployment
            or self.stream_timeout  # timeout set on router
            or self.default_litellm_params.get("stream_timeout", None)
        )

    def _get_non_stream_timeout(
        self, kwargs: dict, data: dict
    ) -> Optional[Union[float, int]]:
        """Helper to get non-stream timeout from kwargs or deployment params"""
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

    def _get_timeout(self, kwargs: dict, data: dict) -> Optional[Union[float, int]]:
        """Helper to get timeout from kwargs or deployment params"""
        timeout: Optional[Union[float, int]] = None
        if kwargs.get("stream", False):
            timeout = self._get_stream_timeout(kwargs=kwargs, data=data)
        if timeout is None:
            timeout = self._get_non_stream_timeout(
                kwargs=kwargs, data=data
            )  # default to this if no stream specific timeout set
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
            model: str, messages: List[AllMessageValues], **kwargs
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
            messages: List[AllMessageValues],
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
        self, model: str, messages: List[List[AllMessageValues]], **kwargs
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
            model: str, messages: List[AllMessageValues], **kwargs
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
                result = await self.acompletion(model=model, messages=messages, stream=stream, **kwargs)  # type: ignore
                return result
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
        self, model: str, messages: List[AllMessageValues], priority: int, stream: Literal[False] = False, **kwargs
    ) -> ModelResponse: 
        ...
    
    @overload
    async def schedule_acompletion(
        self, model: str, messages: List[AllMessageValues], priority: int, stream: Literal[True], **kwargs
    ) -> CustomStreamWrapper: 
        ...

    # fmt: on

    async def schedule_acompletion(
        self,
        model: str,
        messages: List[AllMessageValues],
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
        end_time = time.monotonic() + self.timeout
        curr_time = time.monotonic()
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
                curr_time = time.monotonic()

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

    async def _schedule_factory(
        self,
        model: str,
        priority: int,
        original_function: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
        ### FLOW ITEM ###
        _request_id = str(uuid.uuid4())
        item = FlowItem(
            priority=priority,  # 👈 SET PRIORITY FOR REQUEST
            request_id=_request_id,  # 👈 SET REQUEST ID
            model_name=model,  # 👈 SAME as 'Router'
        )
        ### [fin] ###

        ## ADDS REQUEST TO QUEUE ##
        await self.scheduler.add_request(request=item)

        ## POLL QUEUE
        end_time = time.monotonic() + self.timeout
        curr_time = time.monotonic()
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
                curr_time = time.monotonic()

        if make_request:
            try:
                _response = await original_function(*args, **kwargs)
                if isinstance(_response._hidden_params, dict):
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

    def _is_prompt_management_model(self, model: str) -> bool:
        model_list = self.get_model_list(model_name=model)
        if model_list is None or len(model_list) != 1:
            return False

        litellm_model = model_list[0]["litellm_params"].get("model", None)
        if litellm_model is None or "/" not in litellm_model:
            return False

        split_litellm_model = litellm_model.split("/")[0]
        return split_litellm_model in litellm._known_custom_logger_compatible_callbacks

    async def _prompt_management_factory(
        self,
        model: str,
        messages: List[AllMessageValues],
        kwargs: Dict[str, Any],
    ):
        litellm_logging_object = kwargs.get("litellm_logging_obj", None)
        if litellm_logging_object is None:
            litellm_logging_object, kwargs = function_setup(
                **{
                    "original_function": "acompletion",
                    "rules_obj": Rules(),
                    "start_time": get_utc_datetime(),
                    **kwargs,
                }
            )
        litellm_logging_object = cast(LiteLLMLogging, litellm_logging_object)
        prompt_management_deployment = self.get_available_deployment(
            model=model,
            messages=[{"role": "user", "content": "prompt"}],
            specific_deployment=kwargs.pop("specific_deployment", None),
        )

        self._update_kwargs_with_deployment(
            deployment=prompt_management_deployment, kwargs=kwargs
        )
        data = prompt_management_deployment["litellm_params"].copy()

        litellm_model = data.get("model", None)

        prompt_id = kwargs.get("prompt_id") or prompt_management_deployment[
            "litellm_params"
        ].get("prompt_id", None)
        prompt_variables = kwargs.get(
            "prompt_variables"
        ) or prompt_management_deployment["litellm_params"].get(
            "prompt_variables", None
        )
        prompt_label = kwargs.get("prompt_label", None) or prompt_management_deployment[
            "litellm_params"
        ].get("prompt_label", None)

        if prompt_id is None or not isinstance(prompt_id, str):
            raise ValueError(
                f"Prompt ID is not set or not a string. Got={prompt_id}, type={type(prompt_id)}"
            )
        if prompt_variables is not None and not isinstance(prompt_variables, dict):
            raise ValueError(
                f"Prompt variables is set but not a dictionary. Got={prompt_variables}, type={type(prompt_variables)}"
            )

        (
            model,
            messages,
            optional_params,
        ) = litellm_logging_object.get_chat_completion_prompt(
            model=litellm_model,
            messages=messages,
            non_default_params=get_non_default_completion_params(kwargs=kwargs),
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            prompt_label=prompt_label,
        )

        # Filter out prompt management specific parameters from data before merging
        prompt_management_params = {
            "bitbucket_config",
            "dotprompt_config",
            "prompt_id",
            "prompt_variables",
            "prompt_label",
            "prompt_version",
        }
        filtered_data = {
            k: v for k, v in data.items() if k not in prompt_management_params
        }

        kwargs = {**filtered_data, **kwargs, **optional_params}
        kwargs["model"] = model
        kwargs["messages"] = messages
        kwargs["litellm_logging_obj"] = litellm_logging_object
        kwargs["prompt_id"] = prompt_id
        kwargs["prompt_variables"] = prompt_variables
        kwargs["prompt_label"] = prompt_label

        _model_list = self.get_model_list(model_name=model)
        if _model_list is None or len(_model_list) == 0:  # if direct call to model
            kwargs.pop("original_function")
            return await litellm.acompletion(**kwargs)

        return await self.async_function_with_fallbacks(**kwargs)

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
                request_kwargs=kwargs,
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
                request_kwargs=kwargs,
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
                request_kwargs=kwargs,
            )
            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)
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
                request_kwargs=kwargs,
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
        if kwargs.get("priority", None) is not None:
            return await self._schedule_factory(
                model=model,
                priority=kwargs.pop("priority"),
                original_function=self.atext_completion,
                args=(model, prompt),
                kwargs=kwargs,
            )
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
                request_kwargs=kwargs,
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
                request_kwargs=kwargs,
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

    async def _asearch_with_fallbacks(self, original_function: Callable, **kwargs):
        """
        Helper function to make a search API call through the router with load balancing and fallbacks.
        Reuses the router's retry/fallback infrastructure.
        """
        from litellm.router_utils.search_api_router import SearchAPIRouter

        return await SearchAPIRouter.async_search_with_fallbacks(
            router_instance=self,
            original_function=original_function,
            **kwargs,
        )

    async def _asearch_with_fallbacks_helper(
        self, model: str, original_generic_function: Callable, **kwargs
    ):
        """
        Helper function for search API calls - selects a search tool and calls the original function.
        Called by async_function_with_fallbacks for each retry attempt.
        """
        from litellm.router_utils.search_api_router import SearchAPIRouter

        return await SearchAPIRouter.async_search_with_fallbacks_helper(
            router_instance=self,
            model=model,
            original_generic_function=original_generic_function,
            **kwargs,
        )

    async def _ageneric_api_call_with_fallbacks(
        self, model: str, original_function: Callable, **kwargs
    ):
        """
        Helper function to make a generic LLM API call through the router, this allows you to use retries/fallbacks with litellm router
        """
        try:
            kwargs["model"] = model
            kwargs["original_generic_function"] = original_function
            kwargs["original_function"] = self._ageneric_api_call_with_fallbacks_helper
            self._update_kwargs_before_fallbacks(
                model=model, kwargs=kwargs, metadata_variable_name="litellm_metadata"
            )
            verbose_router_logger.debug(
                f"Inside ageneric_api_call_with_fallbacks() - model: {model}; kwargs: {kwargs}"
            )
            response = await self.async_function_with_fallbacks(**kwargs)
            return response

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

    def _add_deployment_model_to_endpoint_for_llm_passthrough_route(
        self, kwargs: Dict[str, Any], model: str, model_name: str
    ) -> Dict[str, Any]:
        """
        Add the deployment model to the endpoint for LLM passthrough route.

        e.g for bedrock invoke users can pass endpoint as /model/special-bedrock-model/invoke
          it should be actually sent as /model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke
        """
        if "endpoint" in kwargs and kwargs["endpoint"]:
            # For provider-specific endpoints, strip the provider prefix from model_name
            # e.g., "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0" -> "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
            from litellm import get_llm_provider

            try:
                # get_llm_provider returns (model_without_prefix, provider, api_key, api_base)
                stripped_model_name, _, _, _ = get_llm_provider(
                    model=model_name,
                    custom_llm_provider=kwargs.get("custom_llm_provider"),
                    api_base=kwargs.get("api_base"),
                )
                replacement_model_name = stripped_model_name
            except Exception:
                # If get_llm_provider fails, fall back to using model_name as-is
                replacement_model_name = model_name

            kwargs["endpoint"] = kwargs["endpoint"].replace(
                model, replacement_model_name
            )
        return kwargs

    async def _ageneric_api_call_with_fallbacks_helper(
        self, model: str, original_generic_function: Callable, **kwargs
    ):
        """
        Helper function to make a generic LLM API call through the router, this allows you to use retries/fallbacks with litellm router
        """

        passthrough_on_no_deployment = kwargs.pop("passthrough_on_no_deployment", False)
        function_name = "_ageneric_api_call_with_fallbacks"
        try:
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            try:
                deployment = await self.async_get_available_deployment(
                    model=model,
                    request_kwargs=kwargs,
                    messages=kwargs.get("messages", None),
                    specific_deployment=kwargs.pop("specific_deployment", None),
                )
            except Exception as e:
                if passthrough_on_no_deployment:
                    return await original_generic_function(model=model, **kwargs)
                raise e

            self._update_kwargs_with_deployment(
                deployment=deployment, kwargs=kwargs, function_name=function_name
            )

            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            self.total_calls[model_name] += 1

            self._add_deployment_model_to_endpoint_for_llm_passthrough_route(
                kwargs=kwargs, model=model, model_name=model_name
            )
            ### get custom
            response = original_generic_function(
                **{
                    **data,
                    "caching": self.cache_responses,
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
                f"ageneric_api_call_with_fallbacks(model={model_name})\033[32m 200 OK\033[0m"
            )

            return response
        except Exception as e:
            verbose_router_logger.info(
                f"ageneric_api_call_with_fallbacks(model={model})\033[31m Exception {str(e)}\033[0m"
            )
            if model is not None:
                self.fail_calls[model] += 1
            raise e

    def _generic_api_call_with_fallbacks(
        self, model: str, original_function: Callable, **kwargs
    ):
        """
        Make a generic LLM API call through the router, this allows you to use retries/fallbacks with litellm router
        Args:
            model: The model to use
            original_function: The handler function to call (e.g., litellm.completion)
            **kwargs: Additional arguments to pass to the handler function
        Returns:
            The response from the handler function
        """
        handler_name = original_function.__name__
        try:
            verbose_router_logger.debug(
                f"Inside _generic_api_call() - handler: {handler_name}, model: {model}; kwargs: {kwargs}"
            )
            deployment = self.get_available_deployment(
                model=model,
                messages=kwargs.get("messages", None),
                specific_deployment=kwargs.pop("specific_deployment", None),
            )
            self._update_kwargs_with_deployment(
                deployment=deployment, kwargs=kwargs, function_name="generic_api_call"
            )

            data = deployment["litellm_params"].copy()
            model_name = data["model"]

            self.total_calls[model_name] += 1

            # For passthrough routes, use the actual model from deployment
            # and swap model name in endpoint if present
            if "endpoint" in kwargs and kwargs["endpoint"]:
                kwargs["endpoint"] = kwargs["endpoint"].replace(model, model_name)
            kwargs["model"] = model_name

            # Perform pre-call checks for routing strategy
            self.routing_strategy_pre_call_checks(deployment=deployment)

            try:
                _, custom_llm_provider, _, _ = get_llm_provider(model=data["model"])
            except Exception:
                custom_llm_provider = None

            response = original_function(
                **{
                    **data,
                    "custom_llm_provider": custom_llm_provider,
                    "caching": self.cache_responses,
                    **kwargs,
                }
            )

            self.success_calls[model_name] += 1
            verbose_router_logger.info(
                f"{handler_name}(model={model_name})\033[32m 200 OK\033[0m"
            )
            return response
        except Exception as e:
            verbose_router_logger.info(
                f"{handler_name}(model={model})\033[31m Exception {str(e)}\033[0m"
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
                request_kwargs=kwargs,
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
    ) -> OpenAIFileObject:
        try:
            kwargs["model"] = model
            kwargs["original_function"] = self._acreate_file
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
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

    async def _acreate_file(
        self,
        model: str,
        **kwargs,
    ) -> OpenAIFileObject:
        try:
            from litellm.router_utils.common_utils import add_model_file_id_mappings

            verbose_router_logger.debug(
                f"Inside _atext_completion()- model: {model}; kwargs: {kwargs}"
            )
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            healthy_deployments = await self.async_get_healthy_deployments(
                model=model,
                messages=[{"role": "user", "content": "files-api-fake-text"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
                request_kwargs=kwargs,
                parent_otel_span=parent_otel_span,
            )

            async def create_file_for_deployment(deployment: dict) -> OpenAIFileObject:
                from litellm.litellm_core_utils.core_helpers import safe_deep_copy

                kwargs_copy = safe_deep_copy(kwargs)
                self._update_kwargs_with_deployment(
                    deployment=deployment,
                    kwargs=kwargs_copy,
                    function_name="acreate_file",
                )
                data = deployment["litellm_params"].copy()
                model_name = data["model"]

                model_client = self._get_async_openai_model_client(
                    deployment=deployment,
                    kwargs=kwargs_copy,
                )
                self.total_calls[model_name] += 1

                ## REPLACE MODEL IN FILE WITH SELECTED DEPLOYMENT ##
                stripped_model, custom_llm_provider, _, _ = get_llm_provider(
                    model=data["model"]
                )

                ## REPLACE MODEL IN FILE WITH SELECTED DEPLOYMENT ##
                purpose = cast(Optional[OpenAIFilesPurpose], kwargs.get("purpose"))
                file = cast(Optional[FileTypes], kwargs.get("file"))
                if not file or not purpose:
                    raise Exception(
                        "file and file_purpose are required for create_file"
                    )

                replace_model_in_jsonl_bool = should_replace_model_in_jsonl(
                    purpose=purpose,
                )
                if replace_model_in_jsonl_bool:
                    file = replace_model_in_jsonl(
                        file_content=file,
                        new_model_name=stripped_model,
                    )

                    kwargs_copy["file"] = file

                response = litellm.acreate_file(
                    **{
                        **data,
                        "custom_llm_provider": custom_llm_provider,
                        "caching": self.cache_responses,
                        "client": model_client,
                        **kwargs_copy,
                    }
                )

                rpm_semaphore = self._get_client(
                    deployment=deployment,
                    kwargs=kwargs_copy,
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

                return response

            tasks = []

            if isinstance(healthy_deployments, dict):
                tasks.append(create_file_for_deployment(healthy_deployments))
            else:
                for deployment in healthy_deployments:
                    tasks.append(create_file_for_deployment(deployment))

            responses = await asyncio.gather(*tasks)

            if len(responses) == 0:
                raise Exception("No healthy deployments found.")

            model_file_id_mapping = add_model_file_id_mappings(
                healthy_deployments=healthy_deployments, responses=responses
            )
            returned_response = cast(OpenAIFileObject, responses[0])
            returned_response._hidden_params["model_file_id_mapping"] = (
                model_file_id_mapping
            )
            return returned_response
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
    ) -> LiteLLMBatch:
        try:
            kwargs["model"] = model
            kwargs["original_function"] = self._acreate_batch
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            metadata_variable_name = _get_router_metadata_variable_name(
                function_name="_acreate_batch"
            )
            self._update_kwargs_before_fallbacks(
                model=model,
                kwargs=kwargs,
                metadata_variable_name=metadata_variable_name,
            )
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
    ) -> LiteLLMBatch:
        try:
            verbose_router_logger.debug(
                f"Inside _acreate_batch()- model: {model}; kwargs: {kwargs}"
            )
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "files-api-fake-text"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
                request_kwargs=kwargs,
            )

            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            self._update_kwargs_with_deployment(
                deployment=deployment, kwargs=kwargs, function_name="_acreate_batch"
            )

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
                f"litellm.acreate_batch(model={model_name})\033[32m 200 OK\033[0m"
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
        model: Optional[str] = None,
        **kwargs,
    ) -> LiteLLMBatch:
        """
        Iterate through all models in a model group to check for batch

        Future Improvement - cache the result.
        """
        try:
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            if model is not None:
                filtered_model_list: Optional[
                    Union[List[DeploymentTypedDict], List[Dict], Dict]
                ] = await self.async_get_healthy_deployments(
                    model=model,
                    messages=[{"role": "user", "content": "retrieve-api-fake-text"}],
                    specific_deployment=kwargs.pop("specific_deployment", None),
                    request_kwargs=kwargs,
                    parent_otel_span=parent_otel_span,
                )
            else:
                filtered_model_list = self.get_model_list()
            if filtered_model_list is None:
                raise Exception("Router not yet initialized.")

            receieved_exceptions = []

            async def try_retrieve_batch(model_name: DeploymentTypedDict):
                try:
                    from litellm.litellm_core_utils.core_helpers import safe_deep_copy

                    model = model_name["litellm_params"].get("model")
                    data = model_name["litellm_params"].copy()
                    custom_llm_provider = data.get("custom_llm_provider")
                    if model is None:
                        raise Exception(
                            f"Model not found in litellm_params for deployment: {model_name}"
                        )
                    # Update kwargs with the current model name or any other model-specific adjustments
                    ## SET CUSTOM PROVIDER TO SELECTED DEPLOYMENT ##
                    if not custom_llm_provider:
                        _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
                            model=model
                        )
                    new_kwargs = safe_deep_copy(kwargs)
                    self._update_kwargs_with_deployment(
                        deployment=cast(dict, model_name),
                        kwargs=new_kwargs,
                        function_name="aretrieve_batch",
                    )
                    new_kwargs.pop("custom_llm_provider", None)
                    data.pop("custom_llm_provider", None)
                    return await litellm.aretrieve_batch(
                        **{
                            **data,
                            "custom_llm_provider": custom_llm_provider,
                            **new_kwargs,  # type: ignore
                        },
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    receieved_exceptions.append(e)
                    return None

            # Check all models in parallel
            if (
                filtered_model_list is not None
                and isinstance(filtered_model_list, list)
                and len(filtered_model_list) > 0
            ):
                results = await asyncio.gather(
                    *[
                        try_retrieve_batch(cast(DeploymentTypedDict, model))
                        for model in filtered_model_list
                    ],
                    return_exceptions=True,
                )
            elif filtered_model_list is not None and isinstance(
                filtered_model_list, dict
            ):
                results = await try_retrieve_batch(
                    cast(DeploymentTypedDict, filtered_model_list)
                )
            else:
                raise Exception("No healthy deployments found.")

            # Check for successful responses and handle exceptions
            if results is not None:
                if isinstance(results, LiteLLMBatch):
                    return results
                elif isinstance(results, list):
                    for result in results:
                        if isinstance(result, LiteLLMBatch):
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

        final_results: Dict = {
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
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        # update kwargs with model_group
        self._update_kwargs_before_fallbacks(
            model=kwargs.get("model", ""),
            kwargs=kwargs,
        )
        if kwargs.get("model") and self.get_model_list(model_name=kwargs["model"]):
            deployment = await self.async_get_available_deployment(
                model=kwargs["model"],
                request_kwargs=kwargs,
            )
            kwargs["model"] = deployment["litellm_params"]["model"]
            data = deployment["litellm_params"].copy()
            self._update_kwargs_with_deployment(
                deployment=deployment,
                kwargs=kwargs,
            )
            kwargs.update(data)

        return await original_function(**kwargs)

    def factory_function(
        self,
        original_function: Callable,
        call_type: Literal[
            "assistants",
            "moderation",
            "anthropic_messages",
            "aresponses",
            "acancel_responses",
            "responses",
            "aget_responses",
            "adelete_responses",
            "afile_delete",
            "afile_content",
            "_arealtime",
            "acancel_batch",
            "acreate_fine_tuning_job",
            "acancel_fine_tuning_job",
            "alist_fine_tuning_jobs",
            "aretrieve_fine_tuning_job",
            "alist_files",
            "aimage_edit",
            "allm_passthrough_route",
            "alist_input_items",
            "agenerate_content",
            "generate_content",
            "agenerate_content_stream",
            "generate_content_stream",
            "avector_store_search",
            "avector_store_create",
            "avector_store_file_create",
            "avector_store_file_list",
            "avector_store_file_retrieve",
            "avector_store_file_content",
            "avector_store_file_update",
            "avector_store_file_delete",
            "vector_store_search",
            "vector_store_create",
            "vector_store_file_create",
            "vector_store_file_list",
            "vector_store_file_retrieve",
            "vector_store_file_content",
            "vector_store_file_update",
            "vector_store_file_delete",
            "aocr",
            "ocr",
            "asearch",
            "search",
            "aadapter_generate_content",
            "avideo_generation",
            "video_generation",
            "avideo_list",
            "video_list",
            "avideo_status",
            "video_status",
            "avideo_content",
            "video_content",
            "avideo_remix",
            "video_remix",
            "acreate_container",
            "create_container",
            "alist_containers",
            "list_containers",
            "aretrieve_container",
            "retrieve_container",
            "adelete_container",
            "delete_container",
            "alist_container_files",
            "list_container_files",
            "aretrieve_container_file",
            "retrieve_container_file",
            "adelete_container_file",
            "delete_container_file",
            "acreate_skill",
            "alist_skills",
            "aget_skill",
            "adelete_skill",
            "acreate_interaction",
            "create_interaction",
            "aget_interaction",
            "get_interaction",
            "adelete_interaction",
            "delete_interaction",
            "acancel_interaction",
            "cancel_interaction",
        ] = "assistants",
    ):
        """
        Creates appropriate wrapper functions for different API call types.

        Returns:
            - A synchronous function for synchronous call types
            - An asynchronous function for asynchronous call types
        """
        # Handle synchronous call types
        if call_type in (
            "responses",
            "generate_content",
            "generate_content_stream",
            "vector_store_search",
            "vector_store_create",
            "ocr",
            "search",
            "video_generation",
            "video_list",
            "video_status",
            "video_content",
            "video_remix",
            "create_container",
            "list_containers",
            "retrieve_container",
            "delete_container",
        ):

            def sync_wrapper(
                custom_llm_provider: Optional[str] = None,
                client: Optional[Any] = None,
                **kwargs,
            ):
                return self._generic_api_call_with_fallbacks(
                    original_function=original_function, **kwargs
                )

            return sync_wrapper

        if call_type in (
            "vector_store_file_create",
            "vector_store_file_list",
            "vector_store_file_retrieve",
            "vector_store_file_content",
            "vector_store_file_update",
            "vector_store_file_delete",
        ):

            def vector_store_file_sync_wrapper(
                custom_llm_provider: Optional[str] = None,
                client: Optional[Any] = None,
                **kwargs,
            ):
                return original_function(
                    custom_llm_provider=custom_llm_provider,
                    client=client,
                    **kwargs,
                )

            return vector_store_file_sync_wrapper

        # Handle asynchronous call types
        async def async_wrapper(
            custom_llm_provider: Optional[str] = None,
            client: Optional[Any] = None,
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
                return await self._pass_through_moderation_endpoint_factory(
                    original_function=original_function, **kwargs
                )
            elif call_type in ("asearch", "search"):
                return await self._asearch_with_fallbacks(
                    original_function=original_function,
                    **kwargs,
                )
            elif call_type in (
                "avector_store_file_create",
                "avector_store_file_list",
                "avector_store_file_retrieve",
                "avector_store_file_content",
                "avector_store_file_update",
                "avector_store_file_delete",
            ):
                return await self._init_vector_store_api_endpoints(
                    original_function=original_function,
                    custom_llm_provider=custom_llm_provider,
                    **kwargs,
                )
            elif call_type in (
                "anthropic_messages",
                "aresponses",
                "_arealtime",
                "acreate_fine_tuning_job",
                "acancel_fine_tuning_job",
                "alist_fine_tuning_jobs",
                "aretrieve_fine_tuning_job",
                "alist_files",
                "aimage_edit",
                "agenerate_content",
                "agenerate_content_stream",
                "aocr",
                "ocr",
                "avideo_generation",
                "avideo_list",
                "avideo_status",
                "avideo_content",
                "avideo_remix",
                "acancel_batch",
                "acreate_skill",
                "alist_skills",
                "aget_skill",
                "adelete_skill",
                "acreate_interaction",
                "create_interaction",
            ):
                return await self._ageneric_api_call_with_fallbacks(
                    original_function=original_function,
                    **kwargs,
                )
            elif call_type in (
                "acreate_container",
                "alist_containers",
                "aretrieve_container",
                "adelete_container",
                "alist_container_files",
                "aretrieve_container_file",
                "adelete_container_file",
                "aretrieve_container_file_content",
            ):
                return await self._init_containers_api_endpoints(
                    original_function=original_function,
                    custom_llm_provider=custom_llm_provider,
                    **kwargs,
                )
            elif call_type == "allm_passthrough_route":
                return await self._ageneric_api_call_with_fallbacks(
                    original_function=original_function,
                    passthrough_on_no_deployment=True,
                    **kwargs,
                )
            elif call_type in (
                "aget_responses",
                "acancel_responses",
                "adelete_responses",
                "alist_input_items",
            ):
                return await self._init_responses_api_endpoints(
                    original_function=original_function,
                    **kwargs,
                )
            elif call_type in (
                "avector_store_search",
                "avector_store_create",
            ):
                return await self._init_vector_store_api_endpoints(
                    original_function=original_function,
                    custom_llm_provider=custom_llm_provider,
                    **kwargs,
                )
            elif call_type in ("afile_delete", "afile_content"):
                return await self._ageneric_api_call_with_fallbacks(
                    original_function=original_function,
                    custom_llm_provider=custom_llm_provider,
                    client=client,
                    **kwargs,
                )
            elif call_type in (
                "aget_interaction",
                "adelete_interaction",
                "acancel_interaction",
            ):
                return await self._init_interactions_api_endpoints(
                    original_function=original_function,
                    custom_llm_provider=custom_llm_provider,
                    **kwargs,
                )

        return async_wrapper

    async def _init_vector_store_api_endpoints(
        self,
        original_function: Callable,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Vector Store API endpoints on the router.
        """
        if custom_llm_provider and "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = custom_llm_provider
        return await original_function(**kwargs)

    async def _init_containers_api_endpoints(
        self,
        original_function: Callable,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Containers API endpoints on the router.

        Container operations don't need model-based routing, so we call the
        original function directly with the custom_llm_provider.
        """
        if custom_llm_provider and "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = custom_llm_provider
        return await original_function(**kwargs)

    async def _init_responses_api_endpoints(
        self,
        original_function: Callable,
        **kwargs,
    ):
        """
        Initialize the Responses API endpoints on the router.

        GET, DELETE, CANCEL Responses API Requests encode the model_id in the response_id, this function decodes the response_id and sets the model to the model_id.
        """
        from litellm.responses.utils import ResponsesAPIRequestUtils

        model_id = ResponsesAPIRequestUtils.get_model_id_from_response_id(
            kwargs.get("response_id")
        )
        if model_id is not None:
            kwargs["model"] = model_id
        return await self._ageneric_api_call_with_fallbacks(
            original_function=original_function,
            **kwargs,
        )

    async def _init_interactions_api_endpoints(
        self,
        original_function: Callable,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Interactions API endpoints on the router.

        GET, DELETE, CANCEL Interactions API Requests don't need model-based routing,
        so we call the original function directly with the custom_llm_provider.
        """
        if custom_llm_provider and "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = custom_llm_provider
        # Default to gemini for interactions API
        if "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = "gemini"
        return await original_function(**kwargs)

    async def _pass_through_assistants_endpoint_factory(
        self,
        original_function: Callable,
        custom_llm_provider: Optional[str] = None,
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

    async def async_function_with_fallbacks_common_utils(  # noqa: PLR0915
        self,
        e: Exception,
        disable_fallbacks: Optional[bool],
        fallbacks: Optional[List],
        context_window_fallbacks: Optional[List],
        content_policy_fallbacks: Optional[List],
        model_group: Optional[str],
        args: tuple,
        kwargs: dict,
    ):
        """
        Common utilities for async_function_with_fallbacks
        """
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
                    context_window_fallback_model_group: Optional[List[str]] = (
                        self._get_fallback_model_group_from_fallbacks(
                            fallbacks=context_window_fallbacks,
                            model_group=model_group,
                        )
                    )
                    if context_window_fallback_model_group is None:
                        raise original_exception

                    input_kwargs.update(
                        {
                            "fallback_model_group": context_window_fallback_model_group,
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
                    content_policy_fallback_model_group: Optional[List[str]] = (
                        self._get_fallback_model_group_from_fallbacks(
                            fallbacks=content_policy_fallbacks,
                            model_group=model_group,
                        )
                    )
                    if content_policy_fallback_model_group is None:
                        raise original_exception

                    input_kwargs.update(
                        {
                            "fallback_model_group": content_policy_fallback_model_group,
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
                (
                    fallback_model_group,
                    generic_fallback_idx,
                ) = get_fallback_model_group(
                    fallbacks=fallbacks,  # if fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}]
                    model_group=cast(str, model_group),
                )
                ## if none, check for generic fallback
                if fallback_model_group is None and generic_fallback_idx is not None:
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
            original_exception.message += ". Received Model Group={}\nAvailable Model Group Fallbacks={}".format(  # type: ignore
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

    @tracer.wrap()
    async def async_function_with_fallbacks(self, *args, **kwargs):
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

            if mock_timeout is not None:
                response = await self.async_function_with_retries(
                    *args, **kwargs, mock_timeout=mock_timeout
                )
            else:
                response = await self.async_function_with_retries(*args, **kwargs)
            verbose_router_logger.debug(f"Async Response: {response}")
            response = add_fallback_headers_to_response(
                response=response,
                attempted_fallbacks=0,
            )
            return response
        except Exception as e:
            return await self.async_function_with_fallbacks_common_utils(
                e,
                disable_fallbacks,
                fallbacks,
                context_window_fallbacks,
                content_policy_fallbacks,
                model_group,
                args,
                kwargs,
            )

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
        mock_testing_params = MockRouterTestingParams.from_kwargs(kwargs)
        if (
            mock_testing_params.mock_testing_fallbacks is not None
            and mock_testing_params.mock_testing_fallbacks is True
        ):
            raise litellm.InternalServerError(
                model=model_group,
                llm_provider="",
                message=f"This is a mock exception for model={model_group}, to trigger a fallback. Fallbacks={fallbacks}",
            )
        elif (
            mock_testing_params.mock_testing_context_fallbacks is not None
            and mock_testing_params.mock_testing_context_fallbacks is True
        ):
            raise litellm.ContextWindowExceededError(
                model=model_group,
                llm_provider="",
                message=f"This is a mock exception for model={model_group}, to trigger a fallback. \
                    Context_Window_Fallbacks={context_window_fallbacks}",
            )
        elif (
            mock_testing_params.mock_testing_content_policy_fallbacks is not None
            and mock_testing_params.mock_testing_content_policy_fallbacks is True
        ):
            raise litellm.ContentPolicyViolationError(
                model=model_group,
                llm_provider="",
                message=f"This is a mock exception for model={model_group}, to trigger a fallback. \
                    Context_Policy_Fallbacks={content_policy_fallbacks}",
            )

    @tracer.wrap()
    async def async_function_with_retries(self, *args, **kwargs):  # noqa: PLR0915
        verbose_router_logger.debug("Inside async function with retries.")
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
        _metadata: dict = kwargs.get("litellm_metadata", kwargs.get("metadata")) or {}
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
            response = add_retry_headers_to_response(
                response=response, attempted_retries=0, max_retries=None
            )
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
            (
                _healthy_deployments,
                _all_deployments,
            ) = await self._async_get_healthy_deployments(
                model=kwargs.get("model") or "",
                parent_otel_span=parent_otel_span,
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

            verbose_router_logger.debug(
                f"Retrying request with num_retries: {num_retries}"
            )
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
                    if coroutine_checker.is_async_callable(
                        response
                    ):  # async errors are often returned as coroutines
                        response = await response

                    response = add_retry_headers_to_response(
                        response=response,
                        attempted_retries=current_attempt + 1,
                        max_retries=num_retries,
                    )
                    return response

                except Exception as e:
                    ## LOGGING
                    kwargs = self.log_retry(kwargs=kwargs, e=e)
                    remaining_retries = num_retries - current_attempt
                    _model: Optional[str] = kwargs.get("model")  # type: ignore
                    if _model is not None:
                        (
                            _healthy_deployments,
                            _,
                        ) = await self._async_get_healthy_deployments(
                            model=_model,
                            parent_otel_span=parent_otel_span,
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
        if coroutine_checker.is_async_callable(response) or inspect.isawaitable(
            response
        ):
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

        available_models = self.get_model_list(model_name=model_group)
        num_retries: Optional[int] = None

        if available_models is not None and len(available_models) == 1:
            num_retries = cast(
                Optional[int], available_models[0]["litellm_params"].get("num_retries")
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
                num_retries=num_retries,
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
        return run_async_function(self.async_function_with_fallbacks, *args, **kwargs)

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

    def _get_first_default_fallback(self) -> Optional[str]:
        """
        Returns the first model from the default_fallbacks list, if it exists.
        """
        if self.fallbacks is None:
            return None
        for fallback in self.fallbacks:
            if isinstance(fallback, dict) and "*" in fallback:
                default_list = fallback["*"]
                if isinstance(default_list, list) and len(default_list) > 0:
                    return default_list[0]
        return None

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
        from litellm.types.caching import RedisPipelineIncrementOperation

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
                # Get model_group and id from kwargs like the sync version does
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )
                model_info = kwargs["litellm_params"].get("model_info", {}) or {}
                id = model_info.get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                ## get deployment info
                deployment_info = self.get_deployment(model_id=id)

                if deployment_info is None:
                    return
                else:
                    deployment_model_info = self.get_router_model_info(
                        deployment=deployment_info.model_dump(),
                        received_model_name=model_group,
                    )
                    # get tpm/rpm from deployment info
                    tpm = deployment_info.get("tpm", None)
                    rpm = deployment_info.get("rpm", None)

                    ## check tpm/rpm in litellm_params
                    tpm_litellm_params = deployment_info.litellm_params.tpm
                    rpm_litellm_params = deployment_info.litellm_params.rpm

                    ## check tpm/rpm in model_info
                    tpm_model_info = deployment_model_info.get("tpm", None)
                    rpm_model_info = deployment_model_info.get("rpm", None)

                # Always track deployment successes for cooldown logic, regardless of TPM/RPM limits
                increment_deployment_successes_for_current_minute(
                    litellm_router_instance=self,
                    deployment_id=id,
                )

                ## if all are none, return - no need to track current tpm/rpm usage for models with no tpm/rpm set
                if (
                    tpm is None
                    and rpm is None
                    and tpm_litellm_params is None
                    and rpm_litellm_params is None
                    and tpm_model_info is None
                    and rpm_model_info is None
                ):
                    return

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
                pipeline_operations: List[RedisPipelineIncrementOperation] = []

                ## TPM
                pipeline_operations.append(
                    RedisPipelineIncrementOperation(
                        key=tpm_key,
                        increment_value=total_tokens,
                        ttl=RoutingArgs.ttl.value,
                    )
                )

                ## RPM
                rpm_key = RouterCacheEnum.RPM.value.format(
                    id=id, current_minute=current_minute, model=deployment_name
                )
                pipeline_operations.append(
                    RedisPipelineIncrementOperation(
                        key=rpm_key,
                        increment_value=1,
                        ttl=RoutingArgs.ttl.value,
                    )
                )

                await self.cache.async_increment_cache_pipeline(
                    increment_list=pipeline_operations,
                    parent_otel_span=parent_otel_span,
                )

                return tpm_key

        except Exception as e:
            verbose_router_logger.debug(
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
        verbose_router_logger.debug("Router: Entering 'deployment_callback_on_failure'")
        try:
            exception = kwargs.get("exception", None)
            exception_status = getattr(exception, "status_code", "")

            # Cache litellm_params to avoid repeated dict lookups
            litellm_params = kwargs.get("litellm_params", {})
            _model_info = litellm_params.get("model_info", {})

            exception_headers = litellm.litellm_core_utils.exception_mapping_utils._get_response_headers(
                original_exception=exception
            )

            # Determine cooldown time with priority: deployment config > response header > router default
            deployment_cooldown = litellm_params.get("cooldown_time", None)

            header_cooldown = None
            if exception_headers is not None:
                header_cooldown = litellm.utils._get_retry_after_from_exception_header(
                    response_headers=exception_headers
                )
            ##############################################
            # Logic to determine cooldown time
            # 1. Check if a cooldown time is set in the deployment config
            # 2. Check if a cooldown time is set in the response header
            # 3. If no cooldown time is set, use the router default cooldown time
            ##############################################
            if deployment_cooldown is not None and deployment_cooldown >= 0:
                _time_to_cooldown = deployment_cooldown
            elif header_cooldown is not None and header_cooldown >= 0:
                _time_to_cooldown = header_cooldown
            else:
                _time_to_cooldown = self.cooldown_time

            if isinstance(_model_info, dict):
                deployment_id: Optional[str] = _model_info.get("id")
                if deployment_id is None:
                    return False
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
                verbose_router_logger.debug(
                    "Router: Exiting 'deployment_callback_on_failure' without cooldown. No model_info found."
                )
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

    def _get_metadata_variable_name_from_kwargs(
        self, kwargs: dict
    ) -> Literal["metadata", "litellm_metadata"]:
        """
        Helper to return what the "metadata" field should be called in the request data

        - New endpoints return `litellm_metadata`
        - Old endpoints return `metadata`

        Context:
        - LiteLLM used `metadata` as an internal field for storing metadata
        - OpenAI then started using this field for their metadata
        - LiteLLM is now moving to using `litellm_metadata` for our metadata
        """
        return get_metadata_variable_name_from_kwargs(kwargs)

    def log_retry(self, kwargs: dict, e: Exception) -> dict:
        """
        When a retry or fallback happens, log the details of the just failed model call - similar to Sentry breadcrumbing
        """
        try:
            _metadata_var = (
                "litellm_metadata" if "litellm_metadata" in kwargs else "metadata"
            )
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
                if k not in [_metadata_var, "messages", "original_function"]:
                    previous_model[k] = v
                elif k == _metadata_var and isinstance(v, dict):
                    previous_model[_metadata_var] = {}  # type: ignore
                    for metadata_k, metadata_v in kwargs[_metadata_var].items():
                        if metadata_k != "previous_models":
                            previous_model[k][metadata_k] = metadata_v  # type: ignore

            # check current size of self.previous_models, if it's larger than 3, remove the first element
            if len(self.previous_models) > 3:
                self.previous_models.pop(0)

            self.previous_models.append(previous_model)
            kwargs[_metadata_var]["previous_models"] = self.previous_models
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
        if response.choices and len(response.choices) > 0:
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

        verbose_router_logger.debug(
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
        # Convert to set for O(1) lookup instead of O(n)
        unhealthy_deployments_set = set(unhealthy_deployments)
        healthy_deployments: list = []
        for deployment in _all_deployments:
            if deployment["model_info"]["id"] not in unhealthy_deployments_set:
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
        # Optimized: Use list and join instead of string concatenation in loop
        # This avoids creating many temporary string objects (O(n) vs O(n²) complexity)
        parts = [model_group]
        for k, v in litellm_params.items():
            if isinstance(k, str):
                parts.append(k)
            elif isinstance(k, dict):
                parts.append(json.dumps(k))
            else:
                parts.append(str(k))

            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                parts.append(json.dumps(v))
            else:
                parts.append(str(v))

        concat_str = "".join(parts)
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
        try:
            litellm_params: LiteLLM_Params = LiteLLM_Params(**_litellm_params)
            deployment = Deployment(
                **deployment_info,
                model_name=_model_name,
                litellm_params=litellm_params,
                model_info=_model_info,
            )
            for field in CustomPricingLiteLLMParams.model_fields.keys():
                if deployment.litellm_params.get(field) is not None:
                    _model_info[field] = deployment.litellm_params[field]

            ## REGISTER MODEL INFO IN LITELLM MODEL COST MAP
            model_id = deployment.model_info.id
            if model_id is not None:
                litellm.register_model(
                    model_cost={
                        model_id: _model_info,
                    }
                )

            ## OLD MODEL REGISTRATION ## Kept to prevent breaking changes
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
            if (
                self.deployment_is_active_for_environment(deployment=deployment)
                is not True
            ):
                verbose_router_logger.warning(
                    f"Ignoring deployment {deployment.model_name} as it is not active for environment {deployment.model_info['supported_environments']}"
                )
                return None

            deployment = self._add_deployment(deployment=deployment)

            model = deployment.to_json(exclude_none=True)

            self._add_model_to_list_and_index_map(
                model=model, model_id=deployment.model_info.id
            )
            return deployment
        except Exception as e:
            if self.ignore_invalid_deployments:
                verbose_router_logger.exception(
                    f"Error creating deployment: {e}, ignoring and continuing with other deployments."
                )
                return None
            else:
                raise e

    def _is_auto_router_deployment(self, litellm_params: LiteLLM_Params) -> bool:
        """
        Check if the deployment is an auto-router deployment.

        Returns True if the litellm_params model starts with "auto_router/"
        """
        if litellm_params.model.startswith("auto_router/"):
            return True
        return False

    def init_auto_router_deployment(self, deployment: Deployment):
        """
        Initialize the auto-router deployment.

        This will initialize the auto-router and add it to the auto-routers dictionary.
        """
        from litellm.router_strategy.auto_router.auto_router import AutoRouter

        auto_router_config_path: Optional[str] = (
            deployment.litellm_params.auto_router_config_path
        )
        auto_router_config: Optional[str] = deployment.litellm_params.auto_router_config
        if auto_router_config_path is None and auto_router_config is None:
            raise ValueError(
                "auto_router_config_path or auto_router_config is required for auto-router deployments. Please set it in the litellm_params"
            )

        default_model: Optional[str] = (
            deployment.litellm_params.auto_router_default_model
        )
        if default_model is None:
            raise ValueError(
                "auto_router_default_model is required for auto-router deployments. Please set it in the litellm_params"
            )

        embedding_model: Optional[str] = (
            deployment.litellm_params.auto_router_embedding_model
        )
        if embedding_model is None:
            raise ValueError(
                "auto_router_embedding_model is required for auto-router deployments. Please set it in the litellm_params"
            )

        autor_router: AutoRouter = AutoRouter(
            model_name=deployment.model_name,
            auto_router_config_path=auto_router_config_path,
            auto_router_config=auto_router_config,
            default_model=default_model,
            embedding_model=embedding_model,
            litellm_router_instance=self,
        )
        if deployment.model_name in self.auto_routers:
            raise ValueError(
                f"Auto-router deployment {deployment.model_name} already exists. Please use a different model name."
            )
        self.auto_routers[deployment.model_name] = autor_router

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
        self.model_id_to_deployment_index_map = {}  # Reset the index
        self.model_name_to_deployment_indices = {}  # Reset the model_name index
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
        self.model_names = {m["model_name"] for m in model_list}

        # Note: model_name_to_deployment_indices is already built incrementally
        # by _create_deployment -> _add_model_to_list_and_index_map

    def _add_deployment(self, deployment: Deployment) -> Deployment:
        import os

        #### VALIDATE MODEL ########
        # Check if this is a prompt management model before validating as LLM provider
        litellm_model = deployment.litellm_params.model
        is_prompt_management_model = False

        if "/" in litellm_model:
            split_litellm_model = litellm_model.split("/")[0]
            if split_litellm_model in litellm._known_custom_logger_compatible_callbacks:
                is_prompt_management_model = True

        if is_prompt_management_model:
            # For prompt management models, skip LLM provider validation
            # The actual model will be resolved at runtime from the prompt file
            _model = litellm_model
            custom_llm_provider = None
            dynamic_api_key = None
            api_base = None
        else:
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
            # done reading model["litellm_params"]
            if custom_llm_provider not in litellm.provider_list:
                raise Exception(f"Unsupported provider - {custom_llm_provider}")

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

        _team_id = deployment.model_info.get("team_id")
        _team_public_model_name = deployment.model_info.get("team_public_model_name")
        if (
            _team_id is not None
            and _team_public_model_name is not None
            and "*" in _team_public_model_name
        ):
            if _team_id not in self.team_pattern_routers:
                self.team_pattern_routers[_team_id] = PatternMatchRouter()
            self.team_pattern_routers[_team_id].add_pattern(
                _team_public_model_name, deployment.to_json(exclude_none=True)
            )

        # Azure GPT-Vision Enhancements, users can pass os.environ/
        data_sources = deployment.litellm_params.get("dataSources", []) or []

        for data_source in data_sources:
            params = data_source.get("parameters", {})
            for param_key in ["endpoint", "key"]:
                # if endpoint or key set for Azure GPT Vision Enhancements, check if it's an env var
                if param_key in params and params[param_key].startswith("os.environ/"):
                    env_name = params[param_key].replace("os.environ/", "")
                    params[param_key] = os.environ.get(env_name, "")

        # # init OpenAI, Azure clients
        # InitalizeOpenAISDKClient.set_client(
        #     litellm_router_instance=self, model=deployment.to_json(exclude_none=True)
        # )

        if custom_llm_provider is not None:
            self._initialize_deployment_for_pass_through(
                deployment=deployment,
                custom_llm_provider=custom_llm_provider,
                model=deployment.litellm_params.model,
            )

        #########################################################
        # Check if this is an auto-router deployment
        #########################################################
        if self._is_auto_router_deployment(litellm_params=deployment.litellm_params):
            self.init_auto_router_deployment(deployment=deployment)

        return deployment

    def _initialize_deployment_for_pass_through(
        self, deployment: Deployment, custom_llm_provider: str, model: str
    ):
        """
        Optional: Initialize deployment for pass-through endpoints if `deployment.litellm_params.use_in_pass_through` is True

        Each provider uses diff .env vars for pass-through endpoints, this helper uses the deployment credentials to set the .env vars for pass-through endpoints
        """
        if deployment.litellm_params.use_in_pass_through is True:
            from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
                passthrough_endpoint_router,
            )

            if deployment.litellm_params.litellm_credential_name is not None:
                credential_values = CredentialAccessor.get_credential_values(
                    deployment.litellm_params.litellm_credential_name
                )
            else:
                credential_values = {}

            if custom_llm_provider == "vertex_ai":
                vertex_project = (
                    credential_values.get("vertex_project")
                    or deployment.litellm_params.vertex_project
                )
                vertex_location = (
                    credential_values.get("vertex_location")
                    or deployment.litellm_params.vertex_location
                )
                vertex_credentials = (
                    credential_values.get("vertex_credentials")
                    or deployment.litellm_params.vertex_credentials
                )

                if vertex_project is None or vertex_location is None:
                    raise ValueError(
                        "vertex_project, and vertex_location must be set in litellm_params for pass-through endpoints."
                    )
                passthrough_endpoint_router.add_vertex_credentials(
                    project_id=vertex_project,
                    location=vertex_location,
                    vertex_credentials=vertex_credentials,
                )
            else:
                api_base = (
                    credential_values.get("api_base")
                    or deployment.litellm_params.api_base
                )
                api_key = (
                    credential_values.get("api_key")
                    or deployment.litellm_params.api_key
                )
                passthrough_endpoint_router.set_pass_through_credentials(
                    custom_llm_provider=custom_llm_provider,
                    api_base=api_base,
                    api_key=api_key,
                )
            pass
        pass

    def add_deployment(self, deployment: Deployment) -> Optional[Deployment]:
        """
        Parameters:
        - deployment: Deployment - the deployment to be added to the Router

        Returns:
        - The added deployment
        - OR None (if deployment already exists)
        """
        # check if deployment already exists

        _deployment_model_id = deployment.model_info.id
        if _deployment_model_id and self.has_model_id(_deployment_model_id):
            return None

        # add to model list
        _deployment = deployment.to_json(exclude_none=True)
        # initialize client
        self._add_deployment(deployment=deployment)

        # add to model names
        self._add_model_to_list_and_index_map(
            model=_deployment, model_id=deployment.model_info.id
        )
        self.model_names.add(deployment.model_name)
        return deployment

    def _update_deployment_indices_after_removal(
        self, model_id: str, removal_idx: int
    ) -> None:
        """
        Helper method to update deployment indices after a deployment has been removed from model_list.

        Parameters:
        - model_id: str - the id of the deployment that was removed
        - removal_idx: int - the index where the deployment was removed from model_list
        """
        # Update indices for all models after the removed one
        for deployment_id, idx in self.model_id_to_deployment_index_map.items():
            if idx > removal_idx:
                self.model_id_to_deployment_index_map[deployment_id] = idx - 1
        # Remove the deleted model from index
        if model_id in self.model_id_to_deployment_index_map:
            del self.model_id_to_deployment_index_map[model_id]

        # Update model_name_to_deployment_indices
        for model_name, indices in list(self.model_name_to_deployment_indices.items()):
            # Remove the deleted index
            if removal_idx in indices:
                indices.remove(removal_idx)

            # Decrement all indices greater than removal_idx
            updated_indices = []
            for idx in indices:
                if idx > removal_idx:
                    updated_indices.append(idx - 1)
                else:
                    updated_indices.append(idx)

            # Update or remove the entry
            if len(updated_indices) > 0:
                self.model_name_to_deployment_indices[model_name] = updated_indices
            else:
                del self.model_name_to_deployment_indices[model_name]

    def _add_model_to_list_and_index_map(
        self, model: dict, model_id: Optional[str] = None
    ) -> None:
        """
        Helper method to add a model to the model_list and update both indices.

        Parameters:
        - model: dict - the model to add to the list
        - model_id: Optional[str] - the model ID to use for indexing. If None, will try to get from model["model_info"]["id"]
        """
        idx = len(self.model_list)
        self.model_list.append(model)

        # Update model_id index for O(1) lookup
        if model_id is not None:
            self.model_id_to_deployment_index_map[model_id] = idx
        elif model.get("model_info", {}).get("id") is not None:
            self.model_id_to_deployment_index_map[model["model_info"]["id"]] = idx

        # Update model_name index for O(1) lookup
        model_name = model.get("model_name")
        if model_name:
            if model_name not in self.model_name_to_deployment_indices:
                self.model_name_to_deployment_indices[model_name] = []
            self.model_name_to_deployment_indices[model_name].append(idx)

    def upsert_deployment(self, deployment: Deployment) -> Optional[Deployment]:
        """
        Add or update deployment
        Parameters:
        - deployment: Deployment - the deployment to be added to the Router

        Returns:
        - The added/updated deployment
        """
        try:
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
                deployment_id = deployment.model_info.id
                deployment_fast_mapping = self.model_id_to_deployment_index_map

                if deployment_id in deployment_fast_mapping:
                    removal_idx = deployment_fast_mapping[deployment_id]

                    if removal_idx is not None:
                        self.model_list.pop(removal_idx)
                        self._update_deployment_indices_after_removal(
                            model_id=deployment_id, removal_idx=removal_idx
                        )

            # if the model_id is not in router
            self.add_deployment(deployment=deployment)
            return deployment
        except Exception as e:
            if self.ignore_invalid_deployments:
                verbose_router_logger.debug(
                    f"Error upserting deployment: {e}, ignoring and continuing with other deployments."
                )
                return None
            else:
                raise e

    def delete_deployment(self, id: str) -> Optional[Deployment]:
        """
        Parameters:
        - id: str - the id of the deployment to be deleted

        Returns:
        - The deleted deployment
        - OR None (if deleted deployment not found)
        """
        deployment_idx = None
        if id in self.model_id_to_deployment_index_map:
            deployment_idx = self.model_id_to_deployment_index_map[id]

        try:
            if deployment_idx is not None:
                # Pop the item from the list first
                item = self.model_list.pop(deployment_idx)
                self._update_deployment_indices_after_removal(
                    model_id=id, removal_idx=deployment_idx
                )
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
        # Use O(1) lookup via model_id_to_deployment_index_map only
        if model_id in self.model_id_to_deployment_index_map:
            idx = self.model_id_to_deployment_index_map[model_id]
            model = self.model_list[idx]
            if isinstance(model, dict):
                return Deployment(**model)
            elif isinstance(model, Deployment):
                return model
            else:
                raise Exception("Model invalid format - {}".format(type(model)))

        return None

    def get_deployment_credentials(self, model_id: str) -> Optional[dict]:
        """
        Returns -> dict of credentials for a given model id
        """
        deployment = self.get_deployment(model_id=model_id)
        if deployment is None:
            return None
        return CredentialLiteLLMParams(
            **deployment.litellm_params.model_dump(exclude_none=True)
        ).model_dump(exclude_none=True)

    def get_deployment_by_model_group_name(
        self, model_group_name: str
    ) -> Optional[Deployment]:
        """
        Returns -> Deployment or None

        Raise Exception -> if model found in invalid format

        Optimized with O(1) index lookup instead of O(n) linear scan.
        """
        # O(1) lookup in model_name index
        if model_group_name in self.model_name_to_deployment_indices:
            indices = self.model_name_to_deployment_indices[model_group_name]
            if indices:
                # Return first deployment for this model_name
                model = self.model_list[indices[0]]
                if isinstance(model, dict):
                    return Deployment(**model)
                elif isinstance(model, Deployment):
                    return model
                else:
                    raise Exception("Model Name invalid - {}".format(type(model)))
        return None

    def get_deployment_credentials_with_provider(
        self, model_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get API credentials and provider info from a model name in model_list.
        Useful for passthrough endpoints (files, batches, etc.) that need credentials.

        This method tries to find a deployment by model_id first, and if not found,
        it tries to find by model_group_name (model_name).

        Args:
            model_id: Model ID or model name from model_list (e.g., "gpt-4o-litellm")

        Returns:
            Dictionary containing api_key, api_base, custom_llm_provider, etc.
            Returns None if model not found.

        Example:
            credentials = router.get_deployment_credentials_with_provider("gpt-4o-litellm")
            # Returns: {"api_key": "sk-...", "custom_llm_provider": "openai", ...}
        """
        # Try to get deployment by model_id first
        deployment = self.get_deployment(model_id=model_id)

        # If not found, try by model_group_name
        if deployment is None:
            deployment = self.get_deployment_by_model_group_name(
                model_group_name=model_id
            )

        if deployment is None:
            return None

        # Get basic credentials
        credentials = CredentialLiteLLMParams(
            **deployment.litellm_params.model_dump(exclude_none=True)
        ).model_dump(exclude_none=True)

        # Add custom_llm_provider
        if deployment.litellm_params.custom_llm_provider:
            credentials["custom_llm_provider"] = (
                deployment.litellm_params.custom_llm_provider
            )
        elif "/" in deployment.litellm_params.model:
            # Extract provider from "provider/model" format
            credentials["custom_llm_provider"] = deployment.litellm_params.model.split(
                "/"
            )[0]
        else:
            credentials["custom_llm_provider"] = "openai"  # default

        return credentials

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
                f"Could not identify azure model '{_model}'. Set azure 'base_model' for accurate max tokens, cost tracking, etc.- https://docs.litellm.ai/docs/proxy/cost_tracking#spend-tracking-for-azure-openai-models"
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
        if model is None:
            # Handle case where base_model is None (e.g., Azure models without base_model set)
            # Use the original model from litellm_params
            model = _model

        if not model.startswith("{}/".format(custom_llm_provider)):
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

        Optimized with O(1) index lookup instead of O(n) linear scan.
        """
        # O(1) lookup via model_id_to_deployment_index_map
        if id in self.model_id_to_deployment_index_map:
            idx = self.model_id_to_deployment_index_map[id]
            return self.model_list[idx]
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

    def get_deployment_model_info(
        self, model_id: str, model_name: str
    ) -> Optional[ModelInfo]:
        """
        For a given model id, return the model info

        1. Check if model_id is in model info
        2. If not, check if litellm model name is in model info
        3. If not, return None
        """
        from litellm.utils import _update_dictionary

        model_info: Optional[ModelInfo] = None
        custom_model_info: Optional[dict] = None
        litellm_model_name_model_info: Optional[ModelInfo] = None

        try:
            custom_model_info = litellm.model_cost.get(model_id)
        except Exception:
            pass

        try:
            litellm_model_name_model_info = litellm.get_model_info(model=model_name)
        except Exception:
            pass

        ## check for base model
        try:
            if custom_model_info is not None:
                base_model = custom_model_info.get("base_model", None)
                if base_model is not None:
                    ## update litellm model info with base model info
                    base_model_info = litellm.get_model_info(model=base_model)
                    if base_model_info is not None:
                        custom_model_info = custom_model_info or {}
                        # Base model provides defaults, custom model info overrides
                        custom_model_info = _update_dictionary(
                            cast(dict, base_model_info),
                            custom_model_info,
                        )
        except Exception:
            pass

        if custom_model_info is not None and litellm_model_name_model_info is not None:
            model_info = cast(
                ModelInfo,
                _update_dictionary(
                    cast(dict, litellm_model_name_model_info).copy(),
                    custom_model_info,
                ),
            )
        elif litellm_model_name_model_info is not None:
            model_info = litellm_model_name_model_info

        return model_info

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

            # Cache nested dict access to avoid repeated temporary dict allocations
            model_litellm_params = model.get("litellm_params", {})
            model_info_dict = model.get("model_info", {})

            # get model tpm
            _deployment_tpm: Optional[int] = None
            if _deployment_tpm is None:
                _deployment_tpm = model.get("tpm", None)  # type: ignore
            if _deployment_tpm is None:
                _deployment_tpm = model_litellm_params.get("tpm", None)  # type: ignore
            if _deployment_tpm is None:
                _deployment_tpm = model_info_dict.get("tpm", None)  # type: ignore

            # get model rpm
            _deployment_rpm: Optional[int] = None
            if _deployment_rpm is None:
                _deployment_rpm = model.get("rpm", None)  # type: ignore
            if _deployment_rpm is None:
                _deployment_rpm = model_litellm_params.get("rpm", None)  # type: ignore
            if _deployment_rpm is None:
                _deployment_rpm = model_info_dict.get("rpm", None)  # type: ignore

            # get model info
            try:
                model_id = model_info_dict.get("id", None)
                if model_id is not None:
                    model_info = self.get_deployment_model_info(
                        model_id=model_id, model_name=litellm_params.model
                    )
                else:
                    model_info = None
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

                # Get mode from database model_info if available, otherwise default to "chat"
                db_model_info = model.get("model_info", {})
                mode = db_model_info.get("mode", "chat")

                model_info = ModelMapInfo(
                    key=model_group,
                    max_tokens=None,
                    max_input_tokens=None,
                    max_output_tokens=None,
                    input_cost_per_token=0,
                    output_cost_per_token=0,
                    litellm_provider=llm_provider,
                    mode=mode,
                    supported_openai_params=supported_openai_params,
                    supports_system_messages=None,
                )

            if model_group_info is None:
                model_group_info = ModelGroupInfo(  # type: ignore
                    **{
                        "model_group": user_facing_model_group_name,
                        "providers": [llm_provider],
                        **model_info,
                    }
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
                    model_info.get("supports_web_search", None) is not None
                    and model_info["supports_web_search"] is True  # type: ignore
                ):
                    model_group_info.supports_web_search = True
                if (
                    model_info.get("supports_url_context", None) is not None
                    and model_info["supports_url_context"] is True  # type: ignore
                ):
                    model_group_info.supports_url_context = True

                if (
                    model_info.get("supports_reasoning", None) is not None
                    and model_info["supports_reasoning"] is True  # type: ignore
                ):
                    model_group_info.supports_reasoning = True
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

    @lru_cache(maxsize=DEFAULT_MAX_LRU_CACHE_SIZE)
    def _cached_get_model_group_info(
        self, model_group: str
    ) -> Optional[ModelGroupInfo]:
        """
        Cached version of get_model_group_info, uses @lru_cache wrapper

        This is a speed optimization, since set_response_headers makes a call to get_model_group_info on every request
        """
        return self.get_model_group_info(model_group)

    async def get_remaining_model_group_usage(self, model_group: str) -> Dict[str, int]:
        model_group_info = self._cached_get_model_group_info(model_group)

        if model_group_info is not None and model_group_info.tpm is not None:
            tpm_limit = model_group_info.tpm
        else:
            tpm_limit = None

        if model_group_info is not None and model_group_info.rpm is not None:
            rpm_limit = model_group_info.rpm
        else:
            rpm_limit = None

        if tpm_limit is None and rpm_limit is None:
            return {}

        current_tpm, current_rpm = await self.get_model_group_usage(model_group)

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

    def _build_model_name_index(self, model_list: list) -> None:
        """
        Build model_name -> deployment indices mapping for O(1) lookups.

        This index allows us to find all deployments for a given model_name in O(1) time
        instead of O(n) linear scan through the entire model_list.
        """
        self.model_name_to_deployment_indices.clear()

        for idx, model in enumerate(model_list):
            model_name = model.get("model_name")
            if model_name:
                if model_name not in self.model_name_to_deployment_indices:
                    self.model_name_to_deployment_indices[model_name] = []
                self.model_name_to_deployment_indices[model_name].append(idx)

    def _build_model_id_to_deployment_index_map(self, model_list: list):
        """
        Build model index from model list to enable O(1) lookups immediately.
        This is called during initialization to avoid the race condition where
        requests arrive before model_id_to_deployment_index_map is populated.
        """
        # First populate the model_list
        self.model_list = []
        for _, model in enumerate(model_list):
            # Extract model_info from the model dict
            model_info = model.get("model_info", {})
            model_id = model_info.get("id")

            # If no ID exists, generate one using the same logic as set_model_list
            if model_id is None:
                model_name = model.get("model_name", "")
                litellm_params = model.get("litellm_params", {})
                model_id = self._generate_model_id(model_name, litellm_params)
                # Update the model_info in the original list
                if "model_info" not in model:
                    model["model_info"] = {}
                model["model_info"]["id"] = model_id

            self._add_model_to_list_and_index_map(model=model, model_id=model_id)

    def get_model_ids(
        self, model_name: Optional[str] = None, exclude_team_models: bool = False
    ) -> List[str]:
        """
        if 'model_name' is none, returns all.

        Returns list of model id's.

        Optimized with O(1) or O(k) index lookup when model_name provided,
        instead of O(n) linear scan.
        """
        ids = []

        if model_name is not None:
            # O(1) lookup in model_name index, then O(k) iteration where k = deployments for this model_name
            if model_name in self.model_name_to_deployment_indices:
                indices = self.model_name_to_deployment_indices[model_name]
                for idx in indices:
                    model = self.model_list[idx]
                    if "model_info" in model and "id" in model["model_info"]:
                        if exclude_team_models and model["model_info"].get("team_id"):
                            continue
                        ids.append(model["model_info"]["id"])
        else:
            # When model_name is None, return all model IDs
            # Use the index map keys for O(n) where n = total deployments
            for model_id in self.model_id_to_deployment_index_map.keys():
                idx = self.model_id_to_deployment_index_map[model_id]
                model = self.model_list[idx]
                if "model_info" in model and "id" in model["model_info"]:
                    if exclude_team_models and model["model_info"].get("team_id"):
                        continue
                    ids.append(model_id)

        return ids

    def has_model_id(self, candidate_id: str) -> bool:
        """
        O(1) membership check for a deployment ID without allocating large lists.

        Note: Call sites may pass a variable named `model` when it actually
        contains a deployment ID. This helper expects the deployment ID string.

        Uses the existing `model_id_to_deployment_index_map` which is kept
        in sync by `_build_model_id_to_deployment_index_map` and model-list
        mutation helpers.
        """
        return candidate_id in self.model_id_to_deployment_index_map

    def resolve_model_name_from_model_id(self, model_id: Optional[str]) -> Optional[str]:
        """
        Resolve model_name from model_id.
        
        This method attempts to find the correct model_name to use with the router
        so that litellm_params can be automatically injected from the model config.
        
        Strategy:
        1. First, check if model_id directly matches a model_name or deployment ID
        2. If not, search through router's model_list to find a match by litellm_params.model
        3. Return the model_name if found, None otherwise
        
        Args:
            model_id: The model_id extracted from decoded video_id
                     (could be model_name or litellm_params.model value)
        
        Returns:
            model_name if found, None otherwise. If None, the request will fall through
            to normal flow using environment variables.
        """
        if not model_id:
            return None
        
        # Strategy 1: Check if model_id directly matches a model_name or deployment ID
        if model_id in self.model_names or self.has_model_id(model_id):
            return model_id
        
        # Strategy 2: Search through router's model_list to find by litellm_params.model
        all_models = self.get_model_list(model_name=None)
        if not all_models:
            return None
        
        for deployment in all_models:
            litellm_params = deployment.get("litellm_params", {})
            actual_model = litellm_params.get("model")
            
            # Match by exact match or by checking if actual_model ends with /model_id or :model_id
            # e.g., model_id="veo-2.0-generate-001" matches actual_model="vertex_ai/veo-2.0-generate-001"
            matches = (
                actual_model == model_id
                or (actual_model and actual_model.endswith(f"/{model_id}"))
                or (actual_model and actual_model.endswith(f":{model_id}"))
            )
            
            if matches:
                model_name = deployment.get("model_name")
                if model_name:
                    return model_name
        
        # No match found
        return None

    def map_team_model(self, team_model_name: str, team_id: str) -> Optional[str]:
        """
        Map a team model name to a team-specific model name.

        Returns:
        - deployment id: str - the deployment id of the team-specific model
        - None: if no team-specific model name is found
        """
        models = self.get_model_list(model_name=team_model_name, team_id=team_id)
        if not models:
            return None
        for model in models:
            if model.get("model_info", {}).get("team_id") == team_id:
                return model.get("model_name")

        ## wildcard models
        return None

    def should_include_deployment(
        self, model_name: str, model: dict, team_id: Optional[str] = None
    ) -> bool:
        """
        Get the team-specific model name if team_id matches the deployment.
        """
        if (
            team_id is not None
            and model["model_info"].get("team_id") == team_id
            and model_name == model["model_info"].get("team_public_model_name")
        ):
            return True
        elif model_name is not None and model["model_name"] == model_name:
            return True
        return False

    def _get_all_deployments(
        self,
        model_name: str,
        model_alias: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> List[DeploymentTypedDict]:
        """
        Return all deployments of a model name

        Used for accurate 'get_model_list'.

        if team_id specified, only return team-specific models

        Optimized with O(1) index lookup instead of O(n) linear scan.
        """
        returned_models: List[DeploymentTypedDict] = []

        # O(1) lookup in model_name index
        if model_name in self.model_name_to_deployment_indices:
            indices = self.model_name_to_deployment_indices[model_name]

            # O(k) where k = deployments for this model_name (typically 1-10)
            for idx in indices:
                model = self.model_list[idx]
                if self.should_include_deployment(
                    model_name=model_name, model=model, team_id=team_id
                ):
                    if model_alias is not None:
                        # Optimized: Use shallow copy since we only modify top-level model_name
                        # This is much faster than deepcopy for nested dict structures
                        alias_model = model.copy()
                        alias_model["model_name"] = model_alias
                        returned_models.append(alias_model)
                    else:
                        returned_models.append(model)
        elif team_id is not None:
            # Fallback: if team_id is provided and model_name not in index,
            # check if model_name matches any team_public_model_name
            # O(n) scan but only when team_id lookup fails
            for idx, model in enumerate(self.model_list):
                if self.should_include_deployment(
                    model_name=model_name, model=model, team_id=team_id
                ):
                    if model_alias is not None:
                        # Optimized: Use shallow copy since we only modify top-level model_name
                        alias_model = model.copy()
                        alias_model["model_name"] = model_alias
                        returned_models.append(alias_model)
                    else:
                        returned_models.append(model)

        return returned_models

    def get_model_names(self, team_id: Optional[str] = None) -> List[str]:
        """
        Returns all possible model names for the router, including models defined via model_group_alias.

        If a team_id is provided, only deployments configured with that team_id (i.e. team‐specific models)
        will yield their team public name.
        """
        deployments = self.get_model_list() or []
        model_names = []

        for deployment in deployments:
            model_info = deployment.get("model_info")
            if self._is_team_specific_model(model_info):
                team_model_name = self._get_team_specific_model(
                    deployment=deployment, team_id=team_id
                )
                if team_model_name:
                    model_names.append(team_model_name)
            else:
                model_names.append(deployment.get("model_name", ""))

        return model_names

    def _get_team_specific_model(
        self, deployment: DeploymentTypedDict, team_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the team-specific model name if team_id matches the deployment.

        Args:
            deployment: DeploymentTypedDict - The model deployment
            team_id: Optional[str] - If passed, will return router models set with a `team_id` matching the passed `team_id`.

        Returns:
            str: The `team_public_model_name` if team_id matches
            None: If team_id doesn't match or no team info exists
        """
        model_info: Optional[Dict] = deployment.get("model_info") or {}
        if model_info is None:
            return None
        if team_id == model_info.get("team_id"):
            return model_info.get("team_public_model_name")
        return None

    def _is_team_specific_model(self, model_info: Optional[Dict]) -> bool:
        """
        Check if model info contains team-specific configuration.

        Args:
            model_info: Model information dictionary

        Returns:
            bool: True if model has team-specific configuration
        """
        return bool(model_info and model_info.get("team_id"))

    def get_model_list_from_model_alias(
        self, model_name: Optional[str] = None
    ) -> List[DeploymentTypedDict]:
        """
        Helper function to get model list from model alias.

        Used by `.get_model_list` to get model list from model alias.
        """
        returned_models: List[DeploymentTypedDict] = []
        for model_alias, model_value in self.model_group_alias.items():
            if model_name is not None and model_alias != model_name:
                continue
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

        return returned_models

    def get_model_list(
        self, model_name: Optional[str] = None, team_id: Optional[str] = None
    ) -> Optional[List[DeploymentTypedDict]]:
        """
        Includes router model_group_alias'es as well

        if team_id specified, returns matching team-specific models
        """
        # Note: model_list and model_group_alias are always initialized in __init__
        # so hasattr checks are unnecessary
        returned_models: List[DeploymentTypedDict] = []

        if model_name is not None:
            returned_models.extend(
                self._get_all_deployments(model_name=model_name, team_id=team_id)
            )

        returned_models.extend(
            self.get_model_list_from_model_alias(model_name=model_name)
        )

        if len(returned_models) == 0:  # check if wildcard route
            potential_wildcard_models = self.pattern_router.route(model_name) or []

            ## check for team-specific wildcard models
            if team_id is not None and team_id in self.team_pattern_routers:
                potential_team_only_wildcard_models = (
                    self.team_pattern_routers[team_id].route(model_name) or []
                )
                potential_wildcard_models.extend(potential_team_only_wildcard_models)

            if model_name is not None and potential_wildcard_models is not None:
                for m in potential_wildcard_models:
                    deployment_typed_dict = DeploymentTypedDict(**m)  # type: ignore
                    deployment_typed_dict["model_name"] = model_name
                    returned_models.append(deployment_typed_dict)

        if model_name is None:
            returned_models += self.model_list

        return returned_models

    def get_model_access_groups(
        self,
        model_name: Optional[str] = None,
        model_access_group: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        If model_name is provided, only return access groups for that model.

        Parameters:
        - model_name: Optional[str] - the received model name from the user (can be a wildcard route). If set, will only return access groups for that model.
        - model_access_group: Optional[str] - the received model access group from the user. If set, will only return models for that access group.
        - team_id: Optional[str] - the team id, to resolve team-specific models
        """
        from collections import defaultdict

        access_groups = defaultdict(list)

        model_list = self.get_model_list(model_name=model_name, team_id=team_id)
        if model_list:
            for m in model_list:
                _model_info = m.get("model_info")
                if _model_info:
                    for group in _model_info.get("access_groups", []) or []:
                        if model_access_group is not None:
                            if group == model_access_group:
                                model_name = m["model_name"]
                                access_groups[group].append(model_name)
                        else:
                            model_name = m["model_name"]
                            access_groups[group].append(model_name)

        return access_groups

    def _is_model_access_group_for_wildcard_route(
        self, model_access_group: str
    ) -> bool:
        """
        Return True if model access group is a wildcard route
        """
        # GET ACCESS GROUPS
        access_groups = self.get_model_access_groups(
            model_access_group=model_access_group
        )

        if len(access_groups) == 0:
            return False

        models = access_groups.get(model_access_group, [])

        for model in models:
            # CHECK IF MODEL ACCESS GROUP IS A WILDCARD ROUTE
            if self.pattern_router.route(request=model) is not None:
                return True

        return False

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
            "retry_policy",
            "model_group_alias",
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
            "model_group_alias",
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
            if client is None:
                InitalizeCachedClient.set_max_parallel_requests_client(
                    litellm_router_instance=self, model=deployment
                )
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
                return client
            else:
                cache_key = f"{model_id}_async_client"
                client = self.cache.get_cache(
                    key=cache_key, local_only=True, parent_otel_span=parent_otel_span
                )
                return client
        else:
            if kwargs.get("stream") is True:
                cache_key = f"{model_id}_stream_client"
                client = self.cache.get_cache(
                    key=cache_key, parent_otel_span=parent_otel_span
                )
                return client
            else:
                cache_key = f"{model_id}_client"
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

        # Optimized: Use list() shallow copy instead of deepcopy
        # We only pop from the list, not modify deployment dicts - 100x+ faster on hot path (every request)
        _returned_deployments = list(healthy_deployments)

        invalid_model_indices = set()  # Use set for O(1) membership checks

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
            # Cache nested dict access to avoid repeated temporary dict allocations
            _litellm_params = deployment.get("litellm_params", {})
            _model_info = deployment.get("model_info", {})

            # see if we have the info for this model
            try:
                base_model = _model_info.get("base_model", None)
                if base_model is None:
                    base_model = _litellm_params.get("base_model", None)
                model_info = self.get_router_model_info(
                    deployment=deployment, received_model_name=model
                )
                model = base_model or _litellm_params.get("model", None)

                if (
                    isinstance(model_info, dict)
                    and model_info.get("max_input_tokens", None) is not None
                ):
                    if (
                        isinstance(model_info["max_input_tokens"], int)
                        and input_tokens > model_info["max_input_tokens"]
                    ):
                        invalid_model_indices.add(idx)
                        _context_window_error = True
                        _potential_error_str += (
                            "Model={}, Max Input Tokens={}, Got={}".format(
                                model, model_info["max_input_tokens"], input_tokens
                            )
                        )
                        continue
            except Exception as e:
                verbose_router_logger.exception("An error occurs - {}".format(str(e)))

            model_id = _model_info.get("id", "")
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
                        invalid_model_indices.add(idx)
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
                        invalid_model_indices.add(idx)
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
                            invalid_model_indices.add(idx)

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
            # Single-pass filter using set for O(1) lookups (avoids O(n^2) from repeated pops)
            _returned_deployments = [
                d
                for i, d in enumerate(_returned_deployments)
                if i not in invalid_model_indices
            ]

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
        request_kwargs: Optional[Dict] = None,
    ) -> Tuple[str, Union[List, Dict]]:
        """
        Common checks for 'get_available_deployment' across sync + async call.

        If 'healthy_deployments' returned is None, this means the user chose a specific deployment

        Returns
        - str, the litellm model name
        - List, if multiple models chosen
        - Dict, if specific model chosen
        """

        request_team_id: Optional[str] = None
        if request_kwargs is not None:
            metadata = request_kwargs.get("metadata") or {}
            litellm_metadata = request_kwargs.get("litellm_metadata") or {}
            request_team_id = metadata.get(
                "user_api_key_team_id"
            ) or litellm_metadata.get("user_api_key_team_id")
        # check if aliases set on litellm model alias map
        if specific_deployment is True:
            return model, self._get_deployment_by_litellm_model(model=model)
        elif self.has_model_id(model):
            deployment = self.get_deployment(model_id=model)
            if deployment is not None:
                deployment_model = deployment.litellm_params.model
                return deployment_model, deployment.model_dump(exclude_none=True)
            raise ValueError(
                f"LiteLLM Router: Trying to call specific deployment, but Model ID :{model} does not exist in Model ID map"
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

            if (
                request_team_id is not None
                and request_team_id in self.team_pattern_routers
            ):
                pattern_deployments = self.team_pattern_routers[
                    request_team_id
                ].get_deployments_by_pattern(
                    model=model,
                )
                if pattern_deployments:
                    return model, pattern_deployments

            # check if default deployment is set
            if self.default_deployment is not None:
                # Shallow copy with nested litellm_params copy (100x+ faster than deepcopy)
                updated_deployment = self.default_deployment.copy()
                updated_deployment["litellm_params"] = self.default_deployment[
                    "litellm_params"
                ].copy()
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
            # Check for default fallbacks if no deployments are found for the requested model
            if self._has_default_fallbacks():
                fallback_model = self._get_first_default_fallback()
                if fallback_model:
                    verbose_router_logger.info(
                        f"Model '{model}' not found. Attempting to use default fallback model '{fallback_model}'."
                    )
                    # Re-assign model to the fallback and try to get deployments again
                    model = fallback_model
                    healthy_deployments = self._get_all_deployments(model_name=model)

            # If still no deployments after checking for fallbacks, raise an error
            if len(healthy_deployments) == 0:
                if self.get_model_list(model_name=model) is None:
                    message = f"You passed in model={model}. There is no 'model_name' with this string".format(
                        model
                    )
                else:
                    message = f"You passed in model={model}. There are no healthy deployments for this model".format(
                        model
                    )

                raise litellm.BadRequestError(
                    message=message,
                    model=model,
                    llm_provider="",
                )

        if litellm.model_alias_map and model in litellm.model_alias_map:
            model = litellm.model_alias_map[
                model
            ]  # update the model to the actual value if an alias has been passed in

        return model, healthy_deployments

    async def async_get_healthy_deployments(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        parent_otel_span: Optional[Span] = None,
    ) -> Union[List[Dict], Dict]:
        """
        Get the healthy deployments for a model.

        Returns:
        - List[Dict], if multiple models chosen
        *OR*
        - Dict, if specific model chosen
        """

        model, healthy_deployments = self._common_checks_available_deployment(
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            request_kwargs=request_kwargs,
        )  # type: ignore

        # IF TEAM ID SPECIFIED ON MODEL, AND REQUEST CONTAINS USER_API_KEY_TEAM_ID, FILTER OUT MODELS THAT ARE NOT IN THE TEAM
        ## THIS PREVENTS WRITING FILES OF OTHER TEAMS TO MODELS THAT ARE TEAM-ONLY MODELS
        healthy_deployments = filter_team_based_models(
            healthy_deployments=healthy_deployments,
            request_kwargs=request_kwargs,
        )

        verbose_router_logger.debug(f"healthy_deployments after team filter: {healthy_deployments}")

        healthy_deployments = filter_web_search_deployments(
            healthy_deployments=healthy_deployments,
            request_kwargs=request_kwargs,
        )

        verbose_router_logger.debug(f"healthy_deployments after web search filter: {healthy_deployments}")

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
                cast(List[AllMessageValues], messages) if messages is not None else None
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
            metadata_variable_name=self._get_metadata_variable_name_from_kwargs(
                request_kwargs
            ),
        )

        if len(healthy_deployments) == 0:
            exception = await async_raise_no_deployment_exception(
                litellm_router_instance=self,
                model=model,
                parent_otel_span=parent_otel_span,
            )
            raise exception

        return healthy_deployments

    async def async_get_available_deployment(
        self,
        model: str,
        request_kwargs: Dict,
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

            #########################################################
            # Execute Pre-Routing Hooks
            # this hook can modify the model, messages before the routing decision is made
            #########################################################
            pre_routing_hook_response = await self.async_pre_routing_hook(
                model=model,
                request_kwargs=request_kwargs,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
            )
            if pre_routing_hook_response is not None:
                model = pre_routing_hook_response.model
                messages = pre_routing_hook_response.messages
            #########################################################

            healthy_deployments = await self.async_get_healthy_deployments(
                model=model,
                request_kwargs=request_kwargs,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
                parent_otel_span=parent_otel_span,
            )
            if isinstance(healthy_deployments, dict):
                return healthy_deployments

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

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional[PreRoutingHookResponse]:
        """
        This hook is called before the routing decision is made.

        Used for the litellm auto-router to modify the request before the routing decision is made.
        """
        #########################################################
        # Check if any auto-router should be used
        #########################################################
        if model in self.auto_routers:
            return await self.auto_routers[model].async_pre_routing_hook(
                model=model,
                request_kwargs=request_kwargs,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
            )

        return None

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
        verbose_router_logger.debug(f"cooldown deployments: {cooldown_deployments}")
        # Convert to set for O(1) lookup and use list comprehension for O(n) filtering
        cooldown_set = set(cooldown_deployments)
        return [
            deployment
            for deployment in healthy_deployments
            if deployment["model_info"]["id"] not in cooldown_set
        ]

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

        litellm.logging_callback_manager.add_litellm_callback(_slack_alerting_logger)  # type: ignore
        litellm.logging_callback_manager.add_litellm_success_callback(
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

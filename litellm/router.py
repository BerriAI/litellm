# +-----------------------------------------------+
# |                                               |
# |           反馈与求助（Give Feedback / Get Help）            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  感谢使用！我们 ❤️ 你！ - Krrish 与 Ishaan

import asyncio
import copy
import enum
import hashlib
import inspect
import json
import logging
import re
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
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import anyio
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
from litellm.constants import (
    DEFAULT_HEALTH_CHECK_INTERVAL,
    DEFAULT_HEALTH_CHECK_STALENESS_MULTIPLIER,
    DEFAULT_MAX_LRU_CACHE_SIZE,
)
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
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
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
from litellm.router_utils.health_state_cache import DeploymentHealthCache
from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
    DeploymentAffinityCheck,
)
from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
    ModelRateLimitingCheck,
)
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
    PromptCachingDeploymentCheck,
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
    GuardrailTypedDict,
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
    from litellm.router_strategy.complexity_router.complexity_router import (
        ComplexityRouter,
    )

    Span = Union[_Span, Any]
else:
    Span = Any
    AutoRouter = Any
    ComplexityRouter = Any
    PreRoutingHookResponse = Any


class RoutingArgs(enum.Enum):
    ttl = 60  # 1 分钟（RPM/TPM 缓存 key 的过期时间）


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
        ## GUARDRAIL API ##
        guardrail_list: Optional[List[GuardrailTypedDict]] = None,
        ## CACHING ##
        redis_url: Optional[str] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        redis_db: Optional[int] = None,
        cache_responses: Optional[bool] = False,
        cache_kwargs: dict = {},  # 传递给 RedisCache 的额外参数（见 caching.py）
        caching_groups: Optional[
            List[tuple]
        ] = None,  # 如果希望跨 model group 共享缓存，可以在这里配置
        client_ttl: int = 3600,  # 缓存的 client 的 TTL（单位：秒），超过后会重新初始化
        ## SCHEDULER ##
        polling_interval: Optional[float] = None,
        default_priority: Optional[int] = None,
        ## RELIABILITY ##
        num_retries: Optional[int] = None,
        max_fallbacks: Optional[
            int
        ] = None,  # 在放弃请求之前最多尝试的 fallback 次数，默认 5 次
        timeout: Optional[float] = None,
        stream_timeout: Optional[float] = None,
        default_litellm_params: Optional[
            dict
        ] = None,  # Router.chat.completion.create 的默认参数
        default_max_parallel_requests: Optional[int] = None,
        set_verbose: bool = False,
        debug_level: Literal["DEBUG", "INFO"] = "INFO",
        default_fallbacks: Optional[
            List[str]
        ] = None,  # 通用 fallback，对所有 deployment 生效
        fallbacks: List = [],
        context_window_fallbacks: List = [],
        content_policy_fallbacks: List = [],
        model_group_alias: Optional[
            Dict[str, Union[str, RouterModelGroupAliasItem]]
        ] = {},
        enable_pre_call_checks: bool = False,
        enable_tag_filtering: bool = False,
        tag_filtering_match_any: bool = True,
        retry_after: int = 0,  # 重试失败请求前的最短等待时间（单位：秒）
        retry_policy: Optional[
            Union[RetryPolicy, dict]
        ] = None,  # 针对不同异常类型的自定义重试策略
        model_group_retry_policy: Dict[
            str, RetryPolicy
        ] = {},  # 按 model group 维度的自定义重试策略
        allowed_fails: Optional[
            int
        ] = None,  # 在被加入冷却列表前，一个 deployment 允许失败的次数
        allowed_fails_policy: Optional[
            AllowedFailsPolicy
        ] = None,  # 自定义的 allowed_fails 策略
        cooldown_time: Optional[
            float
        ] = None,  # deployment 失败后进入冷却的时长（单位：秒）
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
        routing_strategy_args: dict = {},  # 仅用于 latency-based 策略
        provider_budget_config: Optional[GenericBudgetConfigType] = None,
        alerting_config: Optional[AlertingConfig] = None,
        router_general_settings: Optional[
            RouterGeneralSettings
        ] = RouterGeneralSettings(),
        deployment_affinity_ttl_seconds: int = 3600,
        model_group_affinity_config: Optional[Dict[str, List[str]]] = None,
        ignore_invalid_deployments: bool = False,
        enable_health_check_routing: bool = False,
        health_check_staleness_threshold: Optional[int] = None,
        health_check_ignore_transient_errors: bool = False,
    ) -> None:
        """
        使用给定的缓存、可靠性和路由策略参数来初始化 Router 类。

        参数：
            model_list (Optional[list])：要使用的模型列表。默认为 None。
            redis_url (Optional[str])：Redis 服务器的 URL。默认为 None。
            redis_host (Optional[str])：Redis 服务器的主机名。默认为 None。
            redis_port (Optional[int])：Redis 服务器的端口。默认为 None。
            redis_password (Optional[str])：Redis 服务器的密码。默认为 None。
            cache_responses (Optional[bool])：是否启用响应缓存。默认 False。
            cache_kwargs (dict)：传递给 RedisCache 的额外参数。默认 {}。
            caching_groups (Optional[List[tuple]])：用于跨 model group 共享缓存的分组列表。默认 None。
            client_ttl (int)：缓存的 client 存活时间（秒）。默认 3600。
            polling_interval (Optional[float])：队列轮询频率，仅用于 '.scheduler_acompletion()'。默认 3ms。
            default_priority (Optional[int])：请求的默认优先级，仅用于 '.scheduler_acompletion()'。默认 None。
            num_retries (Optional[int])：失败请求的重试次数。默认 2。
            timeout (Optional[float])：请求超时时间。默认 None。
            default_litellm_params (dict)：Router.chat.completion.create 的默认参数。默认 {}。
            set_verbose (bool)：是否开启 verbose 模式。默认 False。
            debug_level (Literal["DEBUG", "INFO"])：日志级别。默认 "INFO"。
            fallbacks (List)：fallback 列表。默认 []。
            context_window_fallbacks (List)：上下文窗口 fallback 列表。默认 []。
            enable_pre_call_checks (boolean)：对给定 prompt，过滤掉超出上下文窗口限制的 deployment。
            model_group_alias (Optional[dict])：model group 别名。默认 {}。
            retry_after (int)：重试失败请求的最短等待时间。默认 0。
            allowed_fails (Optional[int])：被加入冷却前允许的失败次数。默认 None。
            cooldown_time (float)：deployment 失败后冷却的秒数。默认 1。
            routing_strategy (Literal["simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing", "cost-based-routing"])：路由策略。默认 "simple-shuffle"。
            routing_strategy_args (dict)：基于延迟路由策略的额外参数。默认 {}。
            alerting_config (AlertingConfig)：Slack 告警配置。默认 None。
            provider_budget_config (ProviderBudgetConfig)：各 LLM 厂商的预算配置。例如：OpenAI $100/天、Azure $100/天。默认 None。
            deployment_affinity_ttl_seconds (int)：user-key → deployment 亲和性映射的 TTL。默认 3600。
            ignore_invalid_deployments (bool)：忽略非法 deployment，继续使用其他有效 deployment。默认是抛错。
        返回：
            Router：litellm.Router 类的一个实例。

        使用示例：
        ```python
        from litellm import Router
        model_list = [
        {
            "model_name": "azure-gpt-3.5-turbo", # 模型别名
            "litellm_params": { # 用于 litellm completion/embedding 调用的参数
                "model": "azure/<your-deployment-name-1>",
                "api_key": <your-api-key>,
                "api_version": <your-api-version>,
                "api_base": <your-api-base>
            },
        },
        {
            "model_name": "azure-gpt-3.5-turbo", # 模型别名
            "litellm_params": { # 用于 litellm completion/embedding 调用的参数
                "model": "azure/<your-deployment-name-2>",
                "api_key": <your-api-key>,
                "api_version": <your-api-version>,
                "api_base": <your-api-base>
            },
        },
        {
            "model_name": "openai-gpt-3.5-turbo", # 模型别名
            "litellm_params": { # 用于 litellm completion/embedding 调用的参数
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
        self.tag_filtering_match_any = tag_filtering_match_any
        from litellm._service_logger import ServiceLogging

        self.service_logger_obj: ServiceLogging = ServiceLogging()
        litellm.suppress_debug_info = True  # 避免 Router 使用时打出 'Give Feedback/Get help' 提示 —— 相关 Issue：https://github.com/BerriAI/litellm/issues/5942
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
        self.guardrail_list = guardrail_list or []
        self.deployment_names: List = (
            []
        )  # litellm_params 中的模型名称列表，例如 azure/chatgpt-v-2
        self.deployment_latency_map = {}
        ### CACHING ###
        cache_type: Literal["local", "redis", "redis-semantic", "s3", "disk"] = (
            "local"  # 默认使用进程内内存缓存
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

            if redis_db is not None:
                verbose_router_logger.warning(
                    "Deprecated 'redis_db' argument used. Please remove 'redis_db' from your config/database and use 'cache_kwargs' instead."
                )
                cache_config["db"] = str(redis_db)

            # Add additional key-value pairs from cache_kwargs
            cache_config.update(cache_kwargs)
            redis_cache = self._create_redis_cache(cache_config)

        if cache_responses:
            if litellm.cache is None:
                # cache 可能已经在 proxy server 中初始化过，这里不应覆盖
                litellm.cache = litellm.Cache(type=cache_type, **cache_config)  # type: ignore
            self.cache_responses = cache_responses
        self.cache = DualCache(
            redis_cache=redis_cache, in_memory_cache=InMemoryCache()
        )  # 使用双层缓存（Redis + 内存）来跟踪冷却、用量等信息

        ### SCHEDULER ###
        self.scheduler = Scheduler(
            polling_interval=polling_interval, redis_cache=redis_cache
        )
        self.default_priority = default_priority
        self.default_deployment = None  # 当用户使用 model = * 时，用来记录默认的 deployment
        self.default_max_parallel_requests = default_max_parallel_requests
        self.provider_default_deployment_ids: List[str] = []
        self.pattern_router = PatternMatchRouter()
        self.team_pattern_routers: Dict[str, PatternMatchRouter] = (
            {}
        )  # 按 team 划分的 PatternMatchRouter，结构为 {"TEAM_ID": PatternMatchRouter}
        self.auto_routers: Dict[str, "AutoRouter"] = {}
        self.complexity_routers: Dict[str, "ComplexityRouter"] = {}

        # 由于 set_model_list 会用到 model_group_alias，这里提前初始化
        self.model_group_alias: Dict[str, Union[str, RouterModelGroupAliasItem]] = (
            model_group_alias or {}
        )  # router 使用的 model group 别名，例如 {"gpt-4": "gpt-3.5-turbo"} —— 所有请求 gpt-4 会被路由到 gpt-3.5-turbo 组

        # 初始化 model ID → deployment 索引的映射，便于 O(1) 查找
        self.model_id_to_deployment_index_map: Dict[str, int] = {}
        # 初始化 model name → deployment 索引列表的映射，便于 O(1) 查找
        # 映射关系：model_name -> model_list 中的索引列表
        self.model_name_to_deployment_indices: Dict[str, List[int]] = {}
        # 映射关系：(team_id, team_public_model_name) -> model_list 中的索引列表
        self.team_model_to_deployment_indices: Dict[Tuple[str, str], List[int]] = {}

        if model_list is not None:
            # set_model_list 会自动构建索引
            self.set_model_list(model_list)
            self.healthy_deployments: List = self.model_list  # type: ignore
            for m in model_list:
                if "model" in m["litellm_params"]:
                    self.deployment_latency_map[m["litellm_params"]["model"]] = 0
        else:
            self.model_list: List = (
                []
            )  # 初始化为空列表，以便 _add_deployment 和 delete_deployment 能正常工作

        self._access_groups_cache: Optional[Dict[str, List[str]]] = None

        if allowed_fails is not None:
            self.allowed_fails = allowed_fails
        else:
            self.allowed_fails = litellm.allowed_fails
        self.cooldown_time = cooldown_time or DEFAULT_COOLDOWN_TIME_SECONDS
        self.cooldown_cache = CooldownCache(
            cache=self.cache, default_cooldown_time=self.cooldown_time
        )
        self.disable_cooldowns = disable_cooldowns
        self.enable_health_check_routing = enable_health_check_routing
        self.health_check_ignore_transient_errors = health_check_ignore_transient_errors
        _staleness = health_check_staleness_threshold or (
            DEFAULT_HEALTH_CHECK_INTERVAL * DEFAULT_HEALTH_CHECK_STALENESS_MULTIPLIER
        )
        self.health_state_cache = DeploymentHealthCache(
            cache=self.cache, staleness_threshold=float(_staleness)
        )
        self.failed_calls = (
            InMemoryCache()
        )  # 每个 deployment 失败次数的缓存；如果 1 分钟内失败次数 > allowed_fails，就会被加入冷却列表

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

        ## 设置 FALLBACKS ##
        ### 校验格式是否正确
        _fallbacks = fallbacks or litellm.fallbacks

        self.validate_fallbacks(fallback_param=_fallbacks)
        ### 正式设置 fallbacks
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
        )  # 记录每个模型的总调用次数
        self.fail_calls: defaultdict = defaultdict(
            int
        )  # 记录每个模型的失败调用次数
        self.success_calls: defaultdict = defaultdict(
            int
        )  # 记录每个模型的成功调用次数
        self.previous_models: List = (
            []
        )  # 存储失败调用的历史（会作为 metadata 传递给下一次调用）

        # 让 Router.chat.completions.create 与 openai.chat.completions.create 保持兼容
        default_litellm_params = default_litellm_params or {}
        self.chat = litellm.Chat(params=default_litellm_params, router_obj=self)

        # litellm 的默认参数
        self.default_litellm_params = default_litellm_params
        self.default_litellm_params.setdefault("timeout", timeout)
        self.default_litellm_params.setdefault("max_retries", 0)
        self.default_litellm_params.setdefault("metadata", {}).update(
            {"caching_groups": caching_groups}
        )

        self.deployment_stats: dict = {}  # 用于调试负载均衡
        """
        deployment_stats 示例结构：
        {
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

        ### 路由配置 ###
        self.routing_strategy_init(
            routing_strategy=routing_strategy,
            routing_strategy_args=routing_strategy_args,
        )
        self.access_groups = None
        ## 用量跟踪 ##
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
        ## 冷却相关回调 ##
        if isinstance(litellm.failure_callback, list):
            litellm.logging_callback_manager.add_litellm_failure_callback(
                self.deployment_callback_on_failure
            )
        else:
            litellm.failure_callback = [self.deployment_callback_on_failure]
        self.routing_strategy_args = routing_strategy_args
        self.provider_budget_config = provider_budget_config
        self.deployment_affinity_ttl_seconds = deployment_affinity_ttl_seconds
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
            if self.retry_policy is not None:
                verbose_router_logger.info(
                    "\033[32mRouter Custom Retry Policy Set:\n{}\033[0m".format(
                        self.retry_policy.model_dump(exclude_none=True)
                    )
                )

        self.model_group_retry_policy: Optional[Dict[str, RetryPolicy]] = (
            model_group_retry_policy
        )
        self.model_group_affinity_config: Optional[Dict[str, List[str]]] = (
            model_group_affinity_config
        )

        self.allowed_fails_policy: Optional[AllowedFailsPolicy] = None
        if allowed_fails_policy is not None:
            if isinstance(allowed_fails_policy, dict):
                self.allowed_fails_policy = AllowedFailsPolicy(**allowed_fails_policy)
            elif isinstance(allowed_fails_policy, AllowedFailsPolicy):
                self.allowed_fails_policy = allowed_fails_policy

            if self.allowed_fails_policy is not None:
                verbose_router_logger.info(
                    "\033[32mRouter Custom Allowed Fails Policy Set:\n{}\033[0m".format(
                        self.allowed_fails_policy.model_dump(exclude_none=True)
                    )
                )

        self.alerting_config: Optional[AlertingConfig] = alerting_config

        if optional_pre_call_checks is not None:
            self.add_optional_pre_call_checks(optional_pre_call_checks)

        # 如果设置了 model_group_affinity_config 但没有开启任何全局亲和性检查，
        # 仍需要 DeploymentAffinityCheck 回调（所有全局开关为 False），
        # 以便按 model group 的配置单独启用亲和性。
        if self.model_group_affinity_config and not any(
            isinstance(cb, DeploymentAffinityCheck)
            for cb in (self.optional_callbacks or [])
        ):
            if self.optional_callbacks is None:
                self.optional_callbacks = []
            affinity_callback = DeploymentAffinityCheck(
                cache=self.cache,
                ttl_seconds=self.deployment_affinity_ttl_seconds,
                enable_user_key_affinity=False,
                enable_responses_api_affinity=False,
                enable_session_id_affinity=False,
                model_group_affinity_config=self.model_group_affinity_config,
            )
            self.optional_callbacks.append(affinity_callback)
            litellm.logging_callback_manager.add_litellm_callback(affinity_callback)

        if self.alerting_config is not None:
            self._initialize_alerting()

        self.initialize_assistants_endpoint()
        self.initialize_router_endpoints()
        self.apply_default_settings()

    @staticmethod
    def get_valid_args() -> List[str]:
        """
        返回 Router.__init__ 方法所支持的有效参数名列表。
        """
        arg_spec = inspect.getfullargspec(Router.__init__)
        valid_args = arg_spec.args + arg_spec.kwonlyargs
        if "self" in valid_args:
            valid_args.remove("self")
        return valid_args

    def apply_default_settings(self):
        """
        应用默认配置到 router。
        """

        default_pre_call_checks: OptionalPreCallChecks = []
        self.add_optional_pre_call_checks(default_pre_call_checks)
        return None

    def discard(self):
        """
        伪析构方法：当 router 不再使用时，用于清理全局数据结构。
        目前的功能是将 router 注册的所有回调从l所在列表中移除。
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

        # 如果存在 ForwardClientSideHeadersByModelGroup，也移除它
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
        根据 cache_config 初始化 RedisCache 或 RedisClusterCache。
        """
        startup_nodes = cache_config.get("startup_nodes")
        if not startup_nodes:
            _env_cluster_nodes = get_secret("REDIS_CLUSTER_NODES")
            if _env_cluster_nodes is not None and isinstance(_env_cluster_nodes, str):
                startup_nodes = json.loads(_env_cluster_nodes)

        if startup_nodes:
            return RedisClusterCache(**{**cache_config, "startup_nodes": startup_nodes})
        else:
            return RedisCache(**cache_config)

    def _update_redis_cache(self, cache: RedisCache):
        """
        如果 router 尚未配置 redis 缓存，将传入的 cache 设置为其 redis 缓存。

        这样 proxy 用户只需要写：
        ```yaml
        litellm_settings:
            cache: true
        ```
        缓存就能开箱即用。
        """
        if self.cache.redis_cache is None:
            self.cache.redis_cache = cache

    def routing_strategy_init(
        self, routing_strategy: Union[RoutingStrategy, str], routing_strategy_args: dict
    ):
        verbose_router_logger.info(f"Routing strategy: {routing_strategy}")

        # 校验 routing_strategy 取值，带有帮助信息地快速失败
        # 参考：https://github.com/BerriAI/litellm/issues/11330
        # 有效策略来自 RoutingStrategy 枚举 + "simple-shuffle"（默认值，未在枚举中）
        valid_strategy_strings = ["simple-shuffle"] + [s.value for s in RoutingStrategy]

        if routing_strategy is not None:
            is_valid_string = (
                isinstance(routing_strategy, str)
                and routing_strategy in valid_strategy_strings
            )
            is_valid_enum = isinstance(routing_strategy, RoutingStrategy)
            if not is_valid_string and not is_valid_enum:
                raise ValueError(
                    f"非法的 routing_strategy：'{routing_strategy}'。 "
                    f"有效选项：{valid_strategy_strings}。 "
                    f"请检查 config.yaml 中的 'router_settings.routing_strategy'，"
                    f"或者直接使用 Router SDK 时的 'routing_strategy' 参数。"
                )

        if (
            routing_strategy == RoutingStrategy.LEAST_BUSY.value
            or routing_strategy == RoutingStrategy.LEAST_BUSY
        ):
            self.leastbusy_logger = LeastBusyLoggingHandler(router_cache=self.cache)
            ## 注册回调
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
        ## 初始化直通型（pass-through）的 Assistants 端点 ##
        self.acreate_assistants = self.factory_function(litellm.acreate_assistants)
        self.adelete_assistant = self.factory_function(litellm.adelete_assistant)
        self.aget_assistants = self.factory_function(litellm.aget_assistants)
        self.acreate_thread = self.factory_function(litellm.acreate_thread)
        self.aget_thread = self.factory_function(litellm.aget_thread)
        self.a_add_message = self.factory_function(litellm.a_add_message)
        self.aget_messages = self.factory_function(litellm.aget_messages)
        self.arun_thread = self.factory_function(litellm.arun_thread)

    def _initialize_core_endpoints(self):
        """初始化 router 核心端点的辅助方法。"""
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
        self.acompact_responses = self.factory_function(
            litellm.acompact_responses, call_type="acompact_responses"
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
        self._aresponses_websocket = self.factory_function(
            litellm._aresponses_websocket, call_type="_aresponses_websocket"
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
        # 注：acancel_batch 直接在 Router 类上实现（而不使用 factory_function），
        # 以便像 acreate_batch、aretrieve_batch 一样处理模型 → 厂商的映射关系

    def _initialize_vector_store_endpoints(self):
        """初始化 vector store 相关端点。"""
        from litellm.vector_stores.main import (
            adelete,
            alist,
            aretrieve,
            asearch,
            aupdate,
            create,
            delete,
            list,
            retrieve,
            search,
            update,
        )

        self.avector_store_search = self.factory_function(
            asearch, call_type="avector_store_search"
        )
        self.vector_store_search = self.factory_function(
            search, call_type="vector_store_search"
        )
        self.vector_store_create = self.factory_function(
            create, call_type="vector_store_create"
        )
        self.avector_store_retrieve = self.factory_function(
            aretrieve, call_type="avector_store_retrieve"
        )
        self.vector_store_retrieve = self.factory_function(
            retrieve, call_type="vector_store_retrieve"
        )
        self.avector_store_list = self.factory_function(
            alist, call_type="avector_store_list"
        )
        self.vector_store_list = self.factory_function(
            list, call_type="vector_store_list"
        )
        self.avector_store_update = self.factory_function(
            aupdate, call_type="avector_store_update"
        )
        self.vector_store_update = self.factory_function(
            update, call_type="vector_store_update"
        )
        self.avector_store_delete = self.factory_function(
            adelete, call_type="avector_store_delete"
        )
        self.vector_store_delete = self.factory_function(
            delete, call_type="vector_store_delete"
        )

    def _initialize_vector_store_file_endpoints(self):
        """初始化 vector store file 相关端点。"""
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
        """初始化 Google GenAI 相关端点。"""
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
        """初始化 OCR 与搜索相关端点。"""
        from litellm.ocr import aocr, ocr

        self.aocr = self.factory_function(aocr, call_type="aocr")
        self.ocr = self.factory_function(ocr, call_type="ocr")

        from litellm.search import asearch, search

        self.asearch = self.factory_function(asearch, call_type="asearch")
        self.search = self.factory_function(search, call_type="search")

    def _initialize_video_endpoints(self):
        """初始化视频相关端点。"""
        from litellm.videos import (
            avideo_content,
            avideo_create_character,
            avideo_edit,
            avideo_extension,
            avideo_generation,
            avideo_get_character,
            avideo_list,
            avideo_remix,
            avideo_status,
            video_content,
            video_create_character,
            video_edit,
            video_extension,
            video_generation,
            video_get_character,
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
        self.avideo_create_character = self.factory_function(
            avideo_create_character, call_type="avideo_create_character"
        )
        self.video_create_character = self.factory_function(
            video_create_character, call_type="video_create_character"
        )
        self.avideo_get_character = self.factory_function(
            avideo_get_character, call_type="avideo_get_character"
        )
        self.video_get_character = self.factory_function(
            video_get_character, call_type="video_get_character"
        )
        self.avideo_edit = self.factory_function(avideo_edit, call_type="avideo_edit")
        self.video_edit = self.factory_function(video_edit, call_type="video_edit")
        self.avideo_extension = self.factory_function(
            avideo_extension, call_type="avideo_extension"
        )
        self.video_extension = self.factory_function(
            video_extension, call_type="video_extension"
        )

    def _initialize_container_endpoints(self):
        """初始化 container 相关端点。"""
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

        # 自动注册由 JSON 生成的 container file 端点
        for name, func in container_file_endpoints.items():
            setattr(self, name, self.factory_function(func, call_type=name))  # type: ignore[arg-type]

    def _initialize_skills_endpoints(self):
        """初始化 Anthropic Skills API 相关端点。"""
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
        """初始化 Google Interactions API 相关端点。"""
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
        """初始化专项类端点（vector store、OCR、search、video、container、skills、interactions）的辅助方法。"""
        self._initialize_vector_store_endpoints()
        self._initialize_vector_store_file_endpoints()
        self._initialize_google_genai_endpoints()
        self._initialize_ocr_search_endpoints()
        # 用 router 感知的实现覆盖默认的 vector store 方法
        self._override_vector_store_methods_for_router()
        self._initialize_video_endpoints()
        self._initialize_container_endpoints()
        self._initialize_skills_endpoints()
        self._initialize_interactions_endpoints()

    def initialize_router_endpoints(self):
        self._initialize_core_endpoints()
        self._initialize_specialized_endpoints()

    def validate_fallbacks(self, fallback_param: Optional[List]):
        """
        校验 fallbacks 参数的格式是否合法。
        """
        if fallback_param is None:
            return
        for fallback_dict in fallback_param:
            if not isinstance(fallback_dict, dict):
                raise ValueError(f"元素 '{fallback_dict}' 不是一个字典。")
            if len(fallback_dict) != 1:
                raise ValueError(
                    f"字典 '{fallback_dict}' 必须有且仅有一个 key，当前 key 个数为 {len(fallback_dict)}。"
                )

    def add_optional_pre_call_checks(
        self, optional_pre_call_checks: Optional[OptionalPreCallChecks]
    ):
        if optional_pre_call_checks is None:
            return

        # ---------------------------------------------------------------------
        # 统一的 deployment 亲和性（会话粘性）
        # ---------------------------------------------------------------------
        enable_user_key_affinity = "deployment_affinity" in optional_pre_call_checks
        enable_responses_api_affinity = (
            "responses_api_deployment_check" in optional_pre_call_checks
        )
        enable_session_id_affinity = "session_affinity" in optional_pre_call_checks
        if (
            enable_user_key_affinity
            or enable_responses_api_affinity
            or enable_session_id_affinity
        ):
            if self.optional_callbacks is None:
                self.optional_callbacks = []

            existing_affinity_callback: Optional[DeploymentAffinityCheck] = None
            for cb in self.optional_callbacks:
                if isinstance(cb, DeploymentAffinityCheck):
                    existing_affinity_callback = cb
                    break

            if existing_affinity_callback is not None:
                existing_affinity_callback.enable_user_key_affinity = (
                    existing_affinity_callback.enable_user_key_affinity
                    or enable_user_key_affinity
                )
                existing_affinity_callback.enable_responses_api_affinity = (
                    existing_affinity_callback.enable_responses_api_affinity
                    or enable_responses_api_affinity
                )
                existing_affinity_callback.enable_session_id_affinity = (
                    existing_affinity_callback.enable_session_id_affinity
                    or enable_session_id_affinity
                )
                existing_affinity_callback.ttl_seconds = (
                    self.deployment_affinity_ttl_seconds
                )
                if self.model_group_affinity_config:
                    existing_affinity_callback.model_group_affinity_config = (
                        self.model_group_affinity_config
                    )
            else:
                affinity_callback = DeploymentAffinityCheck(
                    cache=self.cache,
                    ttl_seconds=self.deployment_affinity_ttl_seconds,
                    enable_user_key_affinity=enable_user_key_affinity,
                    enable_responses_api_affinity=enable_responses_api_affinity,
                    enable_session_id_affinity=enable_session_id_affinity,
                    model_group_affinity_config=self.model_group_affinity_config,
                )
                self.optional_callbacks.append(affinity_callback)
                litellm.logging_callback_manager.add_litellm_callback(affinity_callback)

        # ---------------------------------------------------------------------
        # 加密内容亲和性
        # ---------------------------------------------------------------------
        if "encrypted_content_affinity" in optional_pre_call_checks:
            from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
                EncryptedContentAffinityCheck,
            )

            if self.optional_callbacks is None:
                self.optional_callbacks = []

            already_registered = any(
                isinstance(cb, EncryptedContentAffinityCheck)
                for cb in self.optional_callbacks
            )
            if not already_registered:
                ec_callback = EncryptedContentAffinityCheck()
                self.optional_callbacks.append(ec_callback)
                litellm.logging_callback_manager.add_litellm_callback(ec_callback)

        # ---------------------------------------------------------------------
        # 其他可选的预调用检查
        # ---------------------------------------------------------------------
        for pre_call_check in optional_pre_call_checks:
            _callback: Optional[CustomLogger] = None
            if pre_call_check in (
                "deployment_affinity",
                "responses_api_deployment_check",
                "session_affinity",
                "encrypted_content_affinity",
            ):
                continue
            if pre_call_check == "prompt_caching":
                _callback = PromptCachingDeploymentCheck(cache=self.cache)
            elif pre_call_check == "router_budget_limiting":
                _callback = RouterBudgetLimiting(
                    dual_cache=self.cache,
                    provider_budget_config=self.provider_budget_config,
                    model_list=self.model_list,
                )
            elif pre_call_check == "enforce_model_rate_limits":
                _callback = ModelRateLimitingCheck(dual_cache=self.cache)

            if _callback is None:
                continue

            if self.optional_callbacks is None:
                self.optional_callbacks = []
            self.optional_callbacks.append(_callback)
            litellm.logging_callback_manager.add_litellm_callback(_callback)

    def print_deployment(self, deployment: dict):
        """
        返回 deployment 的一份拷贝，其中 api key 被脱敏处理。

        仅保留 api key 的前 2 个字符，其余部分用 10 个 * 代替。
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
                f"打印 deployment 时发生错误 - {str(e)}"
            )
            raise e

    ### COMPLETION、EMBEDDING、IMAGE GENERATION 相关方法

    def completion(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        使用示例：
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
        deployment = None
        try:
            # 在选择 deployment 之前记录一份 kwargs。
            # 这样流式 fallback 迭代器就能用原始的 model group 重新派发请求。
            input_kwargs_for_streaming_fallback = kwargs.copy()
            input_kwargs_for_streaming_fallback["model"] = model

            # 按策略（例如最低 TPM/RPM）选择一个可用的 deployment
            deployment = self.get_available_deployment(
                model=model,
                messages=messages,
                specific_deployment=kwargs.pop("specific_deployment", None),
                request_kwargs=kwargs,
            )
            # 检查是否配置了“静默模型实验”
            # 复制一份 litellm_params 避免修改到 Router 的状态
            litellm_params = deployment["litellm_params"].copy()
            silent_model = litellm_params.pop("silent_model", None)

            if silent_model is not None:
                # 镜像流量到第二个模型
                # 使用 threading.Thread（而非 ThreadPoolExecutor）：executor.submit()
                # 需要 pickle 参数，但 deployment 里的 kwargs 可能包含不可 pickle 的对象
                # （例如 OTEL span 内的 _thread.RLock、logger 等）。
                thread = threading.Thread(
                    target=self._silent_experiment_completion,
                    args=(silent_model, messages),
                    kwargs=kwargs,
                    daemon=True,
                )
                thread.start()

            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)
            kwargs.pop("silent_model", None)  # 确保 silent_model 也不残留在 kwargs 中
            model_name = litellm_params["model"]
            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs
            )
            # 检查传入的 key 是否与 client 的 key 一致 #
            dynamic_api_key = kwargs.get("api_key", None)
            if (
                dynamic_api_key is not None
                and potential_model_client is not None
                and dynamic_api_key != potential_model_client.api_key
            ):
                model_client = None
            else:
                model_client = potential_model_client

            ### deployment 级别的预调用检查（例如预先更新 rpm；若超限则抛错） ###
            ## 仅当传入的是 model group（而非具体 model id）时执行
            if not self.has_model_id(model):
                self.routing_strategy_pre_call_checks(deployment=deployment)

            input_kwargs = {
                **litellm_params,
                "messages": messages,
                "caching": self.cache_responses,
                "client": model_client,
                **kwargs,
            }
            response = litellm.completion(**input_kwargs)
            verbose_router_logger.info(
                f"litellm.completion(model={model_name})\033[32m 200 OK\033[0m"
            )

            ## 检查内容过滤（content filter）错误 ##
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

            # 包装流式响应：如果迭代过程中抛出 MidStreamFallbackError，
            # 则触发 Router 的 fallback 链。
            if isinstance(response, CustomStreamWrapper):
                return self._completion_streaming_iterator(
                    model_response=response,
                    messages=messages,
                    initial_kwargs=input_kwargs_for_streaming_fallback,
                )

            return response
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.completion(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            # 在异常上设置当前 deployment 的 num_retries，用于重试逻辑
            if deployment is not None:
                self._set_deployment_num_retries_on_exception(e, deployment)
            raise e

    def _get_silent_experiment_kwargs(self, **kwargs) -> dict:
        """
        为静默实验准备 kwargs，确保它与主调用彻底隔离。

        确保 metadata 隔离：safe_deep_copy 在 deepcopy 失败时
        （例如 metadata 中含有带 parent_otel_span 的 UserAPIKeyAuth，
        OTel Span 无法 deepcopy）会退化到返回原始引用。
        这里强制对 metadata dict 做一次浅拷贝，防止修改（model_group、
        is_silent_experiment）污染到主调用的 metadata。
        """
        from litellm.litellm_core_utils.core_helpers import safe_deep_copy

        silent_kwargs = safe_deep_copy(kwargs)

        # safe_deep_copy 在 deepcopy 失败时（UserAPIKeyAuth.parent_otel_span 不可 deepcopy）
        # 会退回原始 metadata 引用。这里用身份比较检测这种情况，
        # 并强制做一次浅拷贝，这样给 silent_kwargs 设置
        # model_group / is_silent_experiment 时就不会污染主调用的 metadata。
        original_metadata = kwargs.get("metadata")
        if (
            original_metadata is not None
            and silent_kwargs.get("metadata") is original_metadata
        ):
            silent_kwargs["metadata"] = dict(original_metadata)

        if "metadata" not in silent_kwargs:
            silent_kwargs["metadata"] = {}

        # OTel span 在不同事件循环间使用是不安全的。
        # 静默实验会在新事件循环中运行，这里剔除 span，
        # 避免跨循环追踪竞态或 span 状态损坏。
        silent_kwargs["metadata"].pop("litellm_parent_otel_span", None)

        silent_kwargs["metadata"]["is_silent_experiment"] = True

        # 强制 stream=False，保证响应被完整消费且回调能触发
        silent_kwargs["stream"] = False

        # 移除日志对象和 call ID，保证静默实验有全新的日志上下文
        # 避免与 Proxy 数据库（spend_logs）中的主调用记录发生冲突
        silent_kwargs.pop("litellm_call_id", None)
        silent_kwargs.pop("litellm_logging_obj", None)
        silent_kwargs.pop("standard_logging_object", None)
        # 注意：不要移除 proxy_server_request —— 它是费用日志 metadata 所需的

        return silent_kwargs

    def _silent_experiment_completion(
        self, silent_model: str, messages: List[Any], **kwargs
    ):
        """
        在后台（新线程）中运行一次静默实验。
        """
        try:
            # 防止无限递归：若静默模型自身又挂有静默模型，则直接返回
            if kwargs.get("metadata", {}).get("is_silent_experiment", False):
                return

            messages = copy.deepcopy(messages)

            verbose_router_logger.info(
                f"开始针对模型 {silent_model} 的静默实验"
            )

            silent_kwargs = self._get_silent_experiment_kwargs(**kwargs)

            # 覆盖 model_group，保证指标归类到静默模型
            silent_kwargs["metadata"]["model_group"] = silent_model

            # 为本线程创建新的事件循环，以便异步成功回调
            # （如 _ProxyDBLogger）可以调度并写入数据库。
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:

                async def _run_silent_completion():
                    await self.acompletion(
                        model=silent_model,
                        messages=cast(List[AllMessageValues], messages),
                        **silent_kwargs,
                    )
                    # 排干 acompletion 中通过 asyncio.create_task
                    # 创建的“发射后不管”任务（例如告警钩子）。
                    pending = asyncio.all_tasks()
                    current = asyncio.current_task()
                    if current is not None:
                        pending.discard(current)
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                loop.run_until_complete(_run_silent_completion())
            finally:
                loop.close()
        except Exception as e:
            verbose_router_logger.error(
                f"针对模型 {silent_model} 的静默实验失败：{str(e)}"
            )

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

    # 函数的真正实现
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

    @staticmethod
    def _combine_fallback_usage(
        fallback_item: ModelResponseStream,
        complete_response_object_usage: Optional[Usage],
    ) -> None:
        """将部分流的 usage 与 fallback 流的 usage 合并到当前 chunk 上。"""
        from litellm.cost_calculator import BaseTokenUsageProcessor

        usage = cast(Optional[Usage], getattr(fallback_item, "usage", None))
        usage_objects = [usage] if usage is not None else []
        if (
            complete_response_object_usage is not None
            and hasattr(complete_response_object_usage, "usage")
            and complete_response_object_usage.usage is not None  # type: ignore
        ):
            usage_objects.append(complete_response_object_usage)
        combined_usage = BaseTokenUsageProcessor.combine_usage_objects(
            usage_objects=usage_objects
        )
        setattr(fallback_item, "usage", combined_usage)

    async def _acompletion_streaming_iterator(
        self,
        model_response: CustomStreamWrapper,
        messages: List[Dict[str, str]],
        initial_kwargs: dict,
    ) -> CustomStreamWrapper:
        """
        遍历流式响应的辅助方法。

        捕获迭代过程中的异常，交给 router 的 fallback 系统处理。
        """
        from litellm.exceptions import MidStreamFallbackError

        class FallbackStreamWrapper(CustomStreamWrapper):
            def __init__(self, async_generator: AsyncGenerator):
                # 从原始 model_response 复制属性
                super().__init__(
                    completion_stream=async_generator,
                    model=model_response.model,
                    custom_llm_provider=model_response.custom_llm_provider,
                    logging_obj=model_response.logging_obj,
                )
                self._async_generator = async_generator
                # 保留原始响应的隐式参数（包含 litellm_overhead_time_ms 等）
                if hasattr(model_response, "_hidden_params"):
                    self._hidden_params = model_response._hidden_params.copy()

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await self._async_generator.__anext__()

        async def stream_with_fallbacks():
            fallback_response = None  # 用于在 finally 中清理
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
                    # 使用 router 的 fallback 系统
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
                    if e.is_pre_first_chunk or not e.generated_content:
                        # 错误发生前还未生成任何内容（例如首个 chunk 就遭遇 429 限流）。
                        # 用原始 messages 重新重试——此时添加“续写”提示反而
                        # 会浪费 token 并使模型困惑。
                        initial_kwargs["messages"] = messages
                    else:
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

                    # 如果 fallback 返回的是流式响应，则对其迭代
                    if hasattr(fallback_response, "__aiter__"):
                        async for fallback_item in fallback_response:  # type: ignore
                            if (
                                fallback_item
                                and isinstance(fallback_item, ModelResponseStream)
                                and hasattr(fallback_item, "usage")
                            ):
                                self._combine_fallback_usage(
                                    fallback_item, complete_response_object_usage
                                )
                            yield fallback_item
                    else:
                        # fallback 返回的是非流式响应，则返回 None
                        yield None

                except Exception as fallback_error:
                    # fallback 也失败时，记录日志并再度抛出异常
                    verbose_router_logger.error(
                        f"Fallback also failed: {fallback_error}"
                    )
                    raise fallback_error
            finally:
                # 关闭底层流：当生成器被关闭时（例如客户端断连），
                # 将 HTTP 连接释放回连接池。
                # 使用 anyio 的 CancelScope(shield=True) 避免被取消，让 await 能完成。
                with anyio.CancelScope(shield=True):
                    if hasattr(model_response, "aclose"):
                        try:
                            await model_response.aclose()
                        except BaseException as e:
                            verbose_router_logger.debug(
                                "stream_with_fallbacks: error closing model_response: %s",
                                e,
                            )
                    if fallback_response is not None and hasattr(
                        fallback_response, "aclose"
                    ):
                        try:
                            await fallback_response.aclose()
                        except BaseException as e:
                            verbose_router_logger.debug(
                                "stream_with_fallbacks: error closing fallback_response: %s",
                                e,
                            )

        return FallbackStreamWrapper(stream_with_fallbacks())

    def _completion_streaming_iterator(  # noqa: PLR0915
        self,
        model_response: CustomStreamWrapper,
        messages: List[Dict[str, str]],
        initial_kwargs: dict,
    ) -> CustomStreamWrapper:
        """
        _acompletion_streaming_iterator 的同步版本。

        包装同步流式响应，使 MidStreamFallbackError（由
        CustomStreamWrapper.__next__ 抛出）能触发 Router 的 fallback
        链，而不会直接报给调用方。
        """
        from litellm.exceptions import MidStreamFallbackError

        class SyncFallbackStreamWrapper(CustomStreamWrapper):
            def __init__(self, sync_generator: Generator):
                super().__init__(
                    completion_stream=sync_generator,
                    model=model_response.model,
                    custom_llm_provider=model_response.custom_llm_provider,
                    logging_obj=model_response.logging_obj,
                )
                self._sync_generator = sync_generator
                if hasattr(model_response, "_hidden_params"):
                    self._hidden_params = model_response._hidden_params.copy()

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._sync_generator)

        router_self = self

        def stream_with_fallbacks():
            fallback_response = None
            try:
                for item in model_response:
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
                    model_group = cast(str, initial_kwargs.get("model"))
                    fallbacks: Optional[List] = initial_kwargs.get(
                        "fallbacks", router_self.fallbacks
                    )
                    context_window_fallbacks: Optional[List] = initial_kwargs.get(
                        "context_window_fallbacks",
                        router_self.context_window_fallbacks,
                    )
                    content_policy_fallbacks: Optional[List] = initial_kwargs.get(
                        "content_policy_fallbacks",
                        router_self.content_policy_fallbacks,
                    )
                    initial_kwargs["original_function"] = router_self._completion
                    if e.is_pre_first_chunk or not e.generated_content:
                        initial_kwargs["messages"] = messages
                    else:
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
                    router_self._update_kwargs_before_fallbacks(
                        model=model_group, kwargs=initial_kwargs
                    )
                    fallback_response = router_self.function_with_fallbacks(
                        **initial_kwargs,
                        fallbacks=fallbacks,
                        context_window_fallbacks=context_window_fallbacks,
                        content_policy_fallbacks=content_policy_fallbacks,
                    )

                    if hasattr(fallback_response, "__iter__"):
                        for fallback_item in fallback_response:
                            if (
                                fallback_item
                                and isinstance(fallback_item, ModelResponseStream)
                                and hasattr(fallback_item, "usage")
                            ):
                                router_self._combine_fallback_usage(
                                    fallback_item, complete_response_object_usage
                                )
                            yield fallback_item
                    else:
                        yield None

                except Exception as fallback_error:
                    verbose_router_logger.error(
                        f"Fallback also failed: {fallback_error}"
                    )
                    raise fallback_error
            finally:
                if hasattr(model_response, "close"):
                    try:
                        model_response.close()  # type: ignore[reportAttributeAccessIssue]
                    except BaseException as close_err:
                        verbose_router_logger.debug(
                            "stream_with_fallbacks: error closing model_response: %s",
                            close_err,
                        )
                if fallback_response is not None and hasattr(
                    fallback_response, "close"
                ):
                    try:
                        fallback_response.close()
                    except BaseException as close_err:
                        verbose_router_logger.debug(
                            "stream_with_fallbacks: error closing fallback_response: %s",
                            close_err,
                        )

        return SyncFallbackStreamWrapper(stream_with_fallbacks())

    async def _silent_experiment_acompletion(
        self, silent_model: str, messages: List[Any], **kwargs
    ):
        """
        在后台异步运行一次静默实验。
        """
        try:
            # 防止无限递归：若静默模型本身又挂有静默模型，则直接返回
            if kwargs.get("metadata", {}).get("is_silent_experiment", False):
                return

            messages = copy.deepcopy(messages)

            verbose_router_logger.info(
                f"开始针对模型 {silent_model} 的静默实验"
            )

            silent_kwargs = self._get_silent_experiment_kwargs(**kwargs)
            # 覆盖 model_group，保证指标归类到静默模型
            silent_kwargs["metadata"]["model_group"] = silent_model

            # 触发静默请求
            await self.acompletion(
                model=silent_model,
                messages=cast(List[AllMessageValues], messages),
                **silent_kwargs,
            )
        except Exception as e:
            verbose_router_logger.error(
                f"针对模型 {silent_model} 的静默实验失败：{str(e)}"
            )

    async def _acompletion(  # noqa: PLR0915
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Union[
        ModelResponse,
        CustomStreamWrapper,
    ]:
        """
        - 获取一个可用的 deployment
        - 用信号量（semaphore）并发保护调用
        - 信号量与 deployment 的 rpm 一一对应
        - 在信号量内部，执行前还会对本地 rpm 做一次预检查
        """
        model_name = None
        deployment = None
        _timeout_debug_deployment_dict = (
            {}
        )  # 临时字典，用于调试超时问题
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

            # 记录当前 deployment 被选中的频率，便于调试

            self._track_deployment_metrics(
                deployment=deployment, parent_otel_span=parent_otel_span
            )

            # 检查是否配置了静默模型实验
            # 拷贝一份 litellm_params，避免修改 Router 的状态
            litellm_params = deployment["litellm_params"].copy()
            silent_model = litellm_params.pop("silent_model", None)

            if silent_model is not None:
                # 镜像流量到第二个模型
                # 这是静默实验，因此不能阻塞主请求
                asyncio.create_task(
                    self._silent_experiment_acompletion(
                        silent_model=silent_model,
                        messages=messages,  # 使用 messages 而不是 *args
                        **kwargs,
                    )
                )

            self._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)
            kwargs.pop("silent_model", None)  # 确保 silent_model 也不残留在 kwargs 中

            model_name = litellm_params["model"]

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            input_kwargs = {
                **litellm_params,
                "messages": messages,
                "caching": self.cache_responses,
                "client": model_client,
                **kwargs,
            }
            input_kwargs.pop("silent_model", None)

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
                    - 调用前检查 rpm 是否超限
                    - 若允许，则增加 rpm 计数（允许全局值更新，并发安全）
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

            ## 检查内容过滤（content filter）错误 ##
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
            # 记录 deployment 被选中的频率，便于调试
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
            # 在异常上设置当前 deployment 的 num_retries，用于重试逻辑
            if deployment is not None:
                self._set_deployment_num_retries_on_exception(e, deployment)
            raise e
        except Exception as e:
            verbose_router_logger.info(
                f"litellm.acompletion(model={model_name})\033[31m Exception {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            # 在异常上设置当前 deployment 的 num_retries，用于重试逻辑
            if deployment is not None:
                self._set_deployment_num_retries_on_exception(e, deployment)
            raise e

    def _update_kwargs_before_fallbacks(
        self,
        model: str,
        kwargs: dict,
        metadata_variable_name: Optional[str] = "metadata",
    ) -> None:
        """
        向 kwargs 添加或更新：
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

    def _set_deployment_num_retries_on_exception(
        self, exception: Exception, deployment: dict
    ) -> None:
        """
        将 deployment 的 litellm_params 中配置的 num_retries 设置到异常上。

        这使得 async_function_with_retries 中的重试逻辑能使用
        deployment 级别的重试配置，而不是全局的默认值。
        """
        # 仅在异常尚未设置 num_retries 时才设置
        if hasattr(exception, "num_retries") and exception.num_retries is not None:  # type: ignore
            return

        litellm_params = deployment.get("litellm_params", {})
        dep_num_retries = litellm_params.get("num_retries")
        if dep_num_retries is not None:
            try:
                exception.num_retries = int(dep_num_retries)  # type: ignore  # 同时兼容 int 和 str
            except (ValueError, TypeError):
                pass  # 若无法转为 int，则跳过

    def _update_kwargs_with_default_litellm_params(
        self, kwargs: dict, metadata_variable_name: Optional[str] = "metadata"
    ) -> None:
        """
        如果配置了默认的 litellm 参数，将其合并到 kwargs 中。

        根据 metadata_variable_name，选择插入为 "metadata" 或 "litellm_metadata"。
        """
        # 1) 拷贝默认值，并取出其中的 metadata
        defaults = self.default_litellm_params.copy()
        metadata_defaults = defaults.pop("metadata", {}) or {}

        # 2) 将非 metadata 默认值中尚未出现在 kwargs 中的填入
        for key, value in defaults.items():
            if value is None:
                continue
            kwargs.setdefault(key, value)

        # 3) 合并 metadata（根据 metadata_variable_name 决定插入到 "metadata" 还是 "litellm_metadata"）
        kwargs.setdefault(metadata_variable_name, {}).update(metadata_defaults)

    def _handle_clientside_credential(
        self, deployment: dict, kwargs: dict, function_name: Optional[str] = None
    ) -> Deployment:
        """
        处理客户端凭据（clientside credential）。
        """
        model_info = deployment.get("model_info", {}).copy()
        litellm_params = deployment["litellm_params"].copy()
        dynamic_litellm_params = get_dynamic_litellm_params(
            litellm_params=litellm_params, request_kwargs=kwargs
        )
        # 使用 deployment 的 model_name 作为 model_group，生成 model_id
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
        )  # 将新 deployment 加入 router
        return deployment_pydantic_obj

    @staticmethod
    def _merge_tools_from_deployment(deployment: dict, kwargs: dict) -> None:
        """
        将 deployment 的 litellm_params 中的 tools 与请求 kwargs 中的 tools 合并。
        如果两者都有 tools，则拼接（deployment tools 在前，请求 tools 在后）。
        tool_choice：优先用请求中的值，否则用 deployment 的。
        """
        dep_params_raw = deployment.get("litellm_params", {}) or {}
        if isinstance(dep_params_raw, dict):
            dep_params = dep_params_raw
        else:
            dep_params = dep_params_raw.model_dump(exclude_none=True)
        dep_tools = dep_params.get("tools") or []
        req_tools = kwargs.get("tools") or []
        if dep_tools or req_tools:
            merged = list(dep_tools) + list(req_tools)
            kwargs["tools"] = merged
        if "tool_choice" not in kwargs and dep_params.get("tool_choice") is not None:
            kwargs["tool_choice"] = dep_params["tool_choice"]

    def _update_kwargs_with_deployment(
        self,
        deployment: dict,
        kwargs: dict,
        function_name: Optional[str] = None,
    ) -> None:
        """
        做三件事：
        - 将选中的 deployment、model_info、api_base 写入 kwargs["metadata"]（用于日志）
        - 如果配置了默认的 litellm 参数，将其合并到 kwargs
        - 合并 deployment 和请求中的 tools（proxy 配置的 tools + 请求中的 tools）
        """
        self._merge_tools_from_deployment(deployment=deployment, kwargs=kwargs)

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

        ## DEPLOYMENT 级别的 TAGS
        deployment_tags = deployment.get("litellm_params", {}).get("tags")
        if deployment_tags:
            existing_tags = kwargs[metadata_variable_name].get("tags") or []
            merged_tags = list(existing_tags)
            for tag in deployment_tags:
                if tag not in merged_tags:
                    merged_tags.append(tag)
            kwargs[metadata_variable_name]["tags"] = merged_tags

        ## 将凭据名称作为 TAG
        credential_name = deployment.get("litellm_params", {}).get(
            "litellm_credential_name"
        )
        if credential_name:
            credential_tag = f"Credential: {credential_name}"
            existing_tags = kwargs[metadata_variable_name].get("tags") or []
            if credential_tag not in existing_tags:
                existing_tags.append(credential_tag)
            kwargs[metadata_variable_name]["tags"] = existing_tags

        kwargs["model_info"] = model_info

        kwargs["timeout"] = self._get_timeout(
            kwargs=kwargs, data=deployment["litellm_params"]
        )

        self._update_kwargs_with_default_litellm_params(
            kwargs=kwargs, metadata_variable_name=metadata_variable_name
        )

    def _get_async_openai_model_client(self, deployment: dict, kwargs: dict):
        """
        获取为当前 deployment 创建的 AsyncOpenAI 或 AsyncAzureOpenAI client 的辅助方法。

        生产环境中会复用同一个 OpenAI client 以优化延迟/性能。

        如果传入了动态 api key：
            则不复用 client，传 model_client=None。OpenAI/AzureOpenAI client 会在对应厂商的 handler 中重建。
        """
        potential_model_client = self._get_client(
            deployment=deployment, kwargs=kwargs, client_type="async"
        )

        # 检查传入的 key 是否与 client 的 key 一致 #
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
        """从 kwargs 或 deployment 参数中获取流式超时的辅助方法。"""
        return (
            kwargs.get("stream_timeout", None)  # 用户动态传入的参数
            or data.get(
                "stream_timeout", None
            )  # deployment 的 litellm_params 上设置的超时
            or self.stream_timeout  # router 上设置的超时
            or self.default_litellm_params.get("stream_timeout", None)
        )

    def _get_non_stream_timeout(
        self, kwargs: dict, data: dict
    ) -> Optional[Union[float, int]]:
        """从 kwargs 或 deployment 参数中获取非流式超时的辅助方法。"""
        timeout = (
            kwargs.get("timeout", None)  # 用户动态传入的参数
            or kwargs.get("request_timeout", None)  # 用户动态传入的参数
            or data.get(
                "timeout", None
            )  # deployment 的 litellm_params 上设置的超时
            or data.get(
                "request_timeout", None
            )  # deployment 的 litellm_params 上设置的超时
            or self.timeout  # router 上设置的超时
            or self.default_litellm_params.get("timeout", None)
        )
        return timeout

    def _get_timeout(self, kwargs: dict, data: dict) -> Optional[Union[float, int]]:
        """从 kwargs 或 deployment 参数中获取超时时间的辅助方法。"""
        timeout: Optional[Union[float, int]] = None
        if kwargs.get("stream", False):
            timeout = self._get_stream_timeout(kwargs=kwargs, data=data)
        if timeout is None:
            timeout = self._get_non_stream_timeout(
                kwargs=kwargs, data=data
            )  # 若未设置流式专用超时，则默认使用非流式超时
        return timeout

    async def abatch_completion(
        self,
        models: List[str],
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        **kwargs,
    ):
        """
        异步批量 completion。支持两种场景：
        1. 在 litellm.Router 上将 1 个请求批量分发到 N 个模型。此时把 messages 作为 List[Dict[str, str]] 传入。
        2. 在 litellm.Router 上将 N 个请求批量分发到 M 个模型。此时把 messages 作为 List[List[Dict[str, str]]] 传入。

        示例：1 个请求分发到 N 个模型：
        ```
            response = await router.abatch_completion(
                models=["gpt-3.5-turbo", "groq-llama"],
                messages=[
                    {"role": "user", "content": "is litellm becoming a better product ?"}
                ],
                max_tokens=15,
            )
        ```


        示例：N 个请求分发到 M 个模型：
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
        ############## 异步 completion 的辅助函数 ##################

        async def _async_completion_no_exceptions(
            model: str, messages: List[AllMessageValues], **kwargs
        ):
            """
            对 self.async_completion 的包装器，捕获异常并将其作为结果返回
            """
            try:
                return await self.acompletion(model=model, messages=messages, **kwargs)
            except Exception as e:
                return e

        async def _async_completion_no_exceptions_return_idx(
            model: str,
            messages: List[AllMessageValues],
            idx: int,  # 该响应所对应的 message 的下标
            **kwargs,
        ):
            """
            对 self.async_completion 的包装器，捕获异常并将其作为结果返回
            """
            try:
                return (
                    await self.acompletion(model=model, messages=messages, **kwargs),
                    idx,
                )
            except Exception as e:
                return e, idx

        ############## 异步 completion 的辅助函数 ##################

        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            _tasks = []
            for model in models:
                # 逐个添加任务；即便某个任务失败也不影响其他任务
                _tasks.append(_async_completion_no_exceptions(model=model, messages=messages, **kwargs))  # type: ignore
            response = await asyncio.gather(*_tasks)
            return response
        elif isinstance(messages, list) and all(isinstance(m, list) for m in messages):
            _tasks = []
            for idx, message in enumerate(messages):
                for model in models:
                    # 第 X 个请求，第 Y 个模型
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
        异步批量 completion —— 在 litellm.Router 上将多条 message 批量发送到一个 model_group。

        适用于向同一个模型发送多个请求的场景。

        参数：
            model (List[str]): 模型组
            messages (List[List[Dict[str, str]]]): message 列表，列表中的每个元素代表一次请求
            **kwargs: 其他参数
        用法示例：
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
            对 self.async_completion 的包装器，捕获异常并将其作为结果返回
            """
            try:
                return await self.acompletion(model=model, messages=messages, **kwargs)
            except Exception as e:
                return e

        _tasks = []
        for message_request in messages:
            # 逐个添加任务；即便某个任务失败也不影响其他任务
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
        model —— 以逗号分隔的模型名列表。例如 model="gpt-4, gpt-3.5-turbo"

        从多个模型中返回最先成功响应的那一个。是一个 OpenAI 兼容端点。
        """
        models = [m.strip() for m in model.split(",")]

        async def _async_completion_no_exceptions(
            model: str, messages: List[Dict[str, str]], stream: bool, **kwargs: Any
        ) -> Union[ModelResponse, CustomStreamWrapper, Exception]:
            """
            对 self.acompletion 的包装器，捕获异常并将其作为结果返回
            """
            try:
                result = await self.acompletion(model=model, messages=messages, stream=stream, **kwargs)  # type: ignore
                return result
            except asyncio.CancelledError:
                verbose_router_logger.debug(
                    "收到 'task.cancel'，取消对 model={} 的调用。".format(model)
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
                        "收到成功响应，取消其他所有正在进行的 LLM API 调用。"
                    )
                    # 一旦收到期望的响应，取消其他所有仍在等待的任务
                    for t in pending_tasks:
                        t.cancel()
                    return result
            except Exception:
                # 忽略异常，交给外层循环处理
                pass
            finally:
                # 任务结束时将其从 pending tasks 中移除
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

        # 等待第一个成功完成的任务
        while pending_tasks:
            done, pending_tasks = await asyncio.wait(  # type: ignore
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for completed_task in done:
                result = await check_response(completed_task)

                if result is not None:
                    # 返回第一个成功的结果
                    result._hidden_params["fastest_response_batch_completion"] = True
                    return result

        # 如果循环结束还没有返回，说明所有任务都失败了
        raise Exception("All tasks failed")

    ### 调度器（SCHEDULER） ###

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
        ### 构造队列中的 FLOW ITEM ###
        _request_id = str(uuid.uuid4())
        item = FlowItem(
            priority=priority,  # 👈 设置请求优先级
            request_id=_request_id,  # 👈 设置请求 ID
            model_name=model,  # 👈 与 'Router' 保持一致
        )
        ### [完成] ###

        ## 将请求加入队列 ##
        await self.scheduler.add_request(request=item)

        ## 轮询队列
        end_time = time.monotonic() + self.timeout
        curr_time = time.monotonic()
        poll_interval = self.scheduler.polling_interval  # 默认每 3ms 轮询一次
        make_request = False

        while curr_time < end_time:
            _healthy_deployments, _ = await self._async_get_healthy_deployments(
                model=model, parent_otel_span=parent_otel_span
            )
            make_request = await self.scheduler.poll(  ## 轮询队列 ## ——有健康 deployment 或当前请求已处于队头时返回 True
                id=item.request_id,
                model_name=item.model_name,
                health_deployments=_healthy_deployments,
            )
            if make_request:  ## 为 True 则发起请求
                break
            else:  ## 否则继续循环，直到达到 default_timeout
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
            # 抛出超时异常之前，先从调度器队列中清理该请求
            await self.scheduler.remove_request(
                request_id=item.request_id, model_name=item.model_name
            )
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
        ### 构造队列中的 FLOW ITEM ###
        _request_id = str(uuid.uuid4())
        item = FlowItem(
            priority=priority,  # 👈 设置请求优先级
            request_id=_request_id,  # 👈 设置请求 ID
            model_name=model,  # 👈 与 'Router' 保持一致
        )
        ### [完成] ###

        ## 将请求加入队列 ##
        await self.scheduler.add_request(request=item)

        ## 轮询队列
        end_time = time.monotonic() + self.timeout
        curr_time = time.monotonic()
        poll_interval = self.scheduler.polling_interval  # 默认每 3ms 轮询一次
        make_request = False

        while curr_time < end_time:
            _healthy_deployments, _ = await self._async_get_healthy_deployments(
                model=model, parent_otel_span=parent_otel_span
            )
            make_request = await self.scheduler.poll(  ## 轮询队列 ## ——有健康 deployment 或当前请求已处于队头时返回 True
                id=item.request_id,
                model_name=item.model_name,
                health_deployments=_healthy_deployments,
            )
            if make_request:  ## 为 True 则发起请求
                break
            else:  ## 否则继续循环，直到达到 default_timeout
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
            # 抛出超时异常之前，先从调度器队列中清理该请求
            await self.scheduler.remove_request(
                request_id=item.request_id, model_name=item.model_name
            )
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

        # litellm_agent/ 前缀仅用于剥离模型名称，不需要 prompt_id
        is_litellm_agent_model = isinstance(
            litellm_model, str
        ) and litellm_model.startswith("litellm_agent/")

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

        if not is_litellm_agent_model and (
            prompt_id is None or not isinstance(prompt_id, str)
        ):
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

        # 合并前先从 data 中过滤掉 prompt management 专用的参数
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
        if _model_list is None or len(_model_list) == 0:  # 直接调用模型而非通过 Router
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
                f"进入 _image_generation() — model: {model}; kwargs: {kwargs}"
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

            ### 特定 deployment 的调用前检查　### （例如在调用前更新 rpm；若该 deployment 已超限则抛出异常）
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
                f"进入 _image_generation() — model: {model}; kwargs: {kwargs}"
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
                    - 调用前先检查 rpm 限额
                    - 如果通过检查，自增 rpm 计数（修改全局值，并发安全）
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
                f"litellm.aimage_generation(model={model_name})\033[31m 异常 {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    async def atranscription(self, file: FileTypes, model: str, **kwargs):
        """
        用法示例：

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
                f"进入 _atranscription() — model: {model}; kwargs: {kwargs}"
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

            ### 并发安全的 RPM 检查 ###
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
                    - 调用前先检查 rpm 限额
                    - 如果通过检查，自增 rpm 计数（修改全局值，并发安全）
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
                f"litellm.atranscription(model={model_name})\033[31m 异常 {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    async def aspeech(self, model: str, input: str, voice: str, **kwargs):
        """
        用法示例：

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
                ):  # 模型级参数优先于路由器默认参数
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            potential_model_client = self._get_client(
                deployment=deployment, kwargs=kwargs, client_type="async"
            )
            # 检查传入的 key 是否与客户端中的 key 一致 #
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
                f"进入 _rerank() — model: {model}; kwargs: {kwargs}"
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
                f"litellm.arerank(model={model_name})\033[31m 异常 {str(e)}\033[0m"
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

            # 选择一个可用的 deployment（TPM/RPM 最低的）
            deployment = self.get_available_deployment(
                model=model,
                messages=messages,
                specific_deployment=kwargs.pop("specific_deployment", None),
            )

            data = deployment["litellm_params"].copy()
            for k, v in self.default_litellm_params.items():
                if (
                    k not in kwargs
                ):  # 模型级参数优先于路由器默认参数
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            # 通过 litellm.completion() 发起调用
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
                f"进入 _atext_completion() — model: {model}; kwargs: {kwargs}"
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
                    - 调用前先检查 rpm 限额
                    - 如果通过检查，自增 rpm 计数（修改全局值，并发安全）
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
                f"litellm.atext_completion(model={model})\033[31m 异常 {str(e)}\033[0m"
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
                f"进入 _aadapter_completion() — model: {model}; kwargs: {kwargs}"
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
                    - 调用前先检查 rpm 限额
                    - 如果通过检查，自增 rpm 计数（修改全局值，并发安全）
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
                f"litellm.aadapter_completion(model={model})\033[31m 异常 {str(e)}\033[0m"
            )
            if model is not None:
                self.fail_calls[model] += 1
            raise e

    async def _asearch_with_fallbacks(self, original_function: Callable, **kwargs):
        """
        通过 Router 发起 search API 调用的辅助函数，支持负载均衡与回退。
        复用 Router 原有的重试/回退基础设施。
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
        search API 调用的辅助函数 ——选择一个 search 工具并调用原始函数。
        由 async_function_with_fallbacks 在每次重试时调用。
        """
        from litellm.router_utils.search_api_router import SearchAPIRouter

        return await SearchAPIRouter.async_search_with_fallbacks_helper(
            router_instance=self,
            model=model,
            original_generic_function=original_generic_function,
            **kwargs,
        )

    async def aguardrail(
        self,
        guardrail_name: str,
        original_function: Callable,
        **kwargs,
    ):
        """
        带负载均衡和回退地执行一个 guardrail。

        参数：
            guardrail_name: 要执行的 guardrail 名称
            original_function: guardrail 的执行函数（例如 async_pre_call_hook）
            **kwargs: 传给 guardrail 的其他参数

        返回：
            guardrail 执行后的结果
        """
        kwargs["model"] = guardrail_name  # 为了兼容回退系统，那一套以 model 为键
        kwargs["original_generic_function"] = original_function
        kwargs["original_function"] = self._aguardrail_helper
        self._update_kwargs_before_fallbacks(
            model=guardrail_name,
            kwargs=kwargs,
            metadata_variable_name="litellm_metadata",
        )
        verbose_router_logger.debug(
            f"进入 aguardrail() — guardrail_name: {guardrail_name}; kwargs: {kwargs}"
        )
        response = await self.async_function_with_fallbacks(**kwargs)
        return response

    async def _aguardrail_helper(
        self,
        model: str,
        original_generic_function: Callable,
        **kwargs,
    ):
        """
        aguardrail 的辅助函数 ——选择一个 guardrail 部署并执行。
        由 async_function_with_fallbacks 在每次重试时调用。

        参数：
            model: 实际是 guardrail_name（为保持与回退系统的占位符兼容而命名为 'model'）
            original_generic_function: guardrail 的执行函数
            **kwargs: 其他参数
        """
        guardrail_name = model
        selected_guardrail = self.get_available_guardrail(
            guardrail_name=guardrail_name,
        )

        verbose_router_logger.debug(
            f"已选中的 guardrail 部署：{selected_guardrail.get('litellm_params', {}).get('guardrail')}"
        )

        # 将所选中的 guardrail 配置传递给原始函数
        kwargs["selected_guardrail"] = selected_guardrail
        response = await original_generic_function(**kwargs)
        return response

    def get_available_guardrail(
        self,
        guardrail_name: str,
    ) -> "GuardrailTypedDict":
        """
        基于 Router 的负载均衡策略选择一个 guardrail 部署。

        参数：
            guardrail_name: 要选择的 guardrail 名称

        返回：
            选中的 guardrail 配置字典
        """
        from litellm.router_strategy.simple_shuffle import simple_shuffle

        healthy_deployments = [
            g for g in self.guardrail_list if g.get("guardrail_name") == guardrail_name
        ]

        if not healthy_deployments:
            raise ValueError(f"No guardrail found with name: {guardrail_name}")

        if len(healthy_deployments) == 1:
            return healthy_deployments[0]

        # 使用 simple_shuffle 做权重选择
        return cast(
            GuardrailTypedDict,
            simple_shuffle(
                llm_router_instance=self,
                healthy_deployments=healthy_deployments,
                model=guardrail_name,
            ),
        )

    async def _ageneric_api_call_with_fallbacks(
        self, model: str, original_function: Callable, **kwargs
    ):
        """
        通过 Router 发起通用 LLM API 调用的辅助函数，以便在任意接口上使用 litellm router 的重试/回退能力。
        """
        try:
            kwargs["model"] = model
            kwargs["original_generic_function"] = original_function
            kwargs["original_function"] = self._ageneric_api_call_with_fallbacks_helper
            self._update_kwargs_before_fallbacks(
                model=model, kwargs=kwargs, metadata_variable_name="litellm_metadata"
            )
            verbose_router_logger.debug(
                f"进入 ageneric_api_call_with_fallbacks() — model: {model}; kwargs: {kwargs}"
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
        在 LLM passthrough 路由下，把 deployment 实际模型名填入 endpoint 中。

        例如 bedrock invoke 的用户可能传入 endpoint 为 /model/special-bedrock-model/invoke，
        实际向下游发送时应将其替换为 /model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke。
        """
        if "endpoint" in kwargs and kwargs["endpoint"]:
            # 对于特定厂商的 endpoint，需从 model_name 中剥除厂商前缀
            # 例如 "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0" -> "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
            from litellm import get_llm_provider

            try:
                # get_llm_provider 返回的是 (去前缀模型名, provider, api_key, api_base)
                stripped_model_name, _, _, _ = get_llm_provider(
                    model=model_name,
                    custom_llm_provider=kwargs.get("custom_llm_provider"),
                    api_base=kwargs.get("api_base"),
                )
                replacement_model_name = stripped_model_name
            except Exception:
                # 若 get_llm_provider 失败，则回退为直接使用 model_name
                replacement_model_name = model_name

            kwargs["endpoint"] = kwargs["endpoint"].replace(
                model, replacement_model_name
            )
        return kwargs

    async def _ageneric_api_call_with_fallbacks_helper(
        self, model: str, original_generic_function: Callable, **kwargs
    ):
        """
        通过 Router 发起通用 LLM API 调用的辅助函数，以便在任意接口上使用 litellm router 的重试/回退能力。
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
            
            # 从 deployment 参数中获取 custom_llm_provider
            try:
                custom_llm_provider = data.get("custom_llm_provider")
                _, inferred_custom_llm_provider, _, _ = get_llm_provider(
                    model=data["model"],
                    custom_llm_provider=custom_llm_provider,
                )
                custom_llm_provider = custom_llm_provider or inferred_custom_llm_provider
            except Exception:
                custom_llm_provider = None
            
            # 构造下游响应调用的 kwargs
            response_kwargs = {
                **data,
                "caching": self.cache_responses,
                **kwargs,
            }
            # 仅在非 None 时才设置 custom_llm_provider
            if custom_llm_provider is not None:
                response_kwargs["custom_llm_provider"] = custom_llm_provider
            
            response = original_generic_function(**response_kwargs)

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
                    - 调用前先检查 rpm 限额
                    - 如果通过检查，自增 rpm 计数（修改全局值，并发安全）
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
                f"ageneric_api_call_with_fallbacks(model={model})\033[31m 异常 {str(e)}\033[0m"
            )
            if model is not None:
                self.fail_calls[model] += 1
            raise e

    def _generic_api_call_with_fallbacks(
        self, model: str, original_function: Callable, **kwargs
    ):
        """
        通过 Router 发起通用 LLM API 调用，以便在任意接口上使用 litellm router 的重试/回退能力。
        参数：
            model: 要使用的模型
            original_function: 实际要调用的处理函数（例如 litellm.completion）
            **kwargs: 传给处理函数的其他参数
        返回：
            处理函数的调用结果
        """
        handler_name = original_function.__name__
        metadata_variable_name = _get_router_metadata_variable_name(
            function_name="generic_api_call"
        )
        try:
            verbose_router_logger.debug(
                f"进入 _generic_api_call() — handler: {handler_name}, model: {model}; kwargs: {kwargs}"
            )
            self._update_kwargs_before_fallbacks(
                model=model,
                kwargs=kwargs,
                metadata_variable_name=metadata_variable_name,
            )
            deployment = self.get_available_deployment(
                model=model,
                messages=kwargs.get("messages", None),
                specific_deployment=kwargs.pop("specific_deployment", None),
                request_kwargs=kwargs,
            )
            self._update_kwargs_with_deployment(
                deployment=deployment, kwargs=kwargs, function_name="generic_api_call"
            )

            data = deployment["litellm_params"].copy()
            model_name = data["model"]

            self.total_calls[model_name] += 1

            # 对于 passthrough 路由，需要用 deployment 的真实模型名，
            # 并将 endpoint 中的逻辑模型名替换为真实模型名
            if "endpoint" in kwargs and kwargs["endpoint"]:
                kwargs["endpoint"] = kwargs["endpoint"].replace(model, model_name)
            kwargs["model"] = model_name

            # 执行路由策略的调用前检查
            self.routing_strategy_pre_call_checks(deployment=deployment)

            try:
                custom_llm_provider = data.get("custom_llm_provider")
                _, inferred_custom_llm_provider, _, _ = get_llm_provider(
                    model=data["model"],
                    custom_llm_provider=custom_llm_provider,
                )
                custom_llm_provider = custom_llm_provider or inferred_custom_llm_provider
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
                f"{handler_name}(model={model})\033[31m 异常 {str(e)}\033[0m"
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
            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)
            response = self.function_with_fallbacks(**kwargs)
            return response
        except Exception as e:
            raise e

    def _embedding(self, input: Union[str, List], model: str, **kwargs):
        model_name = None
        try:
            verbose_router_logger.debug(
                f"进入 embedding() — model: {model}; kwargs: {kwargs}"
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
            # 检查传入的 key 是否与客户端中的 key 一致 #
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

            ### 特定 deployment 的调用前检查　### （例如在调用前更新 rpm；若该 deployment 已超限则抛出异常）
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
                f"litellm.embedding(model={model_name})\033[31m 异常 {str(e)}\033[0m"
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
                f"进入 _aembedding() — model: {model}; kwargs: {kwargs}"
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

            ### 并发安全的 RPM 检查 ###
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
                    - 调用前先检查 rpm 限额
                    - 如果通过检查，自增 rpm 计数（修改全局值，并发安全）
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
                f"litellm.aembedding(model={model_name})\033[31m 异常 {str(e)}\033[0m"
            )
            if model_name is not None:
                self.fail_calls[model_name] += 1
            raise e

    #### 文件（FILES）API ####
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

    async def _acreate_file(  # noqa: PLR0915
        self,
        model: str,
        **kwargs,
    ) -> OpenAIFileObject:
        try:
            from litellm.router_utils.common_utils import add_model_file_id_mappings

            verbose_router_logger.debug(
                f"进入 _atext_completion() — model: {model}; kwargs: {kwargs}"
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

                ## 将文件中的模型名替换为所选中 deployment 的模型名 ##
                # 对于配置/数据库下发的 deployment，优先使用 deployment 参数中的 provider
                custom_llm_provider = data.get("custom_llm_provider")
                stripped_model, inferred_custom_llm_provider, _, _ = get_llm_provider(
                    model=data["model"],
                    custom_llm_provider=custom_llm_provider,
                )
                # 保留显式存储的 provider，其次才回退到推断值
                custom_llm_provider = custom_llm_provider or inferred_custom_llm_provider

                ## 将文件中的模型名替换为所选中 deployment 的模型名 ##
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
                if (
                    "gcs_bucket_name" in data
                ):  # TODO: 待有更好的处理方式后移除：问题在于 create_file 调用需要将 gcs_bucket_name 传给 router，但它并不会自动出现在这里
                    kwargs_copy.setdefault("litellm_metadata", {})[
                        "gcs_bucket_name"
                    ] = data["gcs_bucket_name"]
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
                        - 调用前先检查 rpm 限额
                        - 如果通过检查，自增 rpm 计数（修改全局值，并发安全）
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
                raise Exception("未找到任何健康的 deployment。")

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
                f"litellm.acreate_file(model={model}, {kwargs})\033[31m 异常 {str(e)}\033[0m"
            )
            if model is not None:
                self.fail_calls[model] += 1
            raise e

    #### 向量存储（VECTOR STORES）API ####
    async def avector_store_create(
        self,
        model: Union[str, None],
        **kwargs,
    ):
        """
        为指定模型创建一个向量存储。

        参数：
            model: router 配置中的模型名
            **kwargs: 向量存储创建参数

        返回：
            VectorStoreCreateResponse
        """
        try:
            # 如果 model 为 None，则采用工厂函数方式（直接 SDK 调用）
            if model is None:
                from litellm.vector_stores.main import acreate

                # 用工厂函数来处理本次调用
                factory_fn = self.factory_function(
                    acreate, call_type="avector_store_create"
                )
                return await factory_fn(**kwargs)

            from litellm.vector_stores import acreate as avector_store_create_sdk

            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "vector-store-api-fake-text"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
                request_kwargs=kwargs,
            )
            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            self._update_kwargs_with_deployment(
                deployment=deployment,
                kwargs=kwargs,
                function_name="avector_store_create",
            )

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            # 从 deployment 参数中获取 custom provider
            custom_llm_provider = data.get("custom_llm_provider")
            _, inferred_custom_llm_provider, _, _ = get_llm_provider(
                model=data["model"],
                custom_llm_provider=custom_llm_provider,
            )
            custom_llm_provider = custom_llm_provider or inferred_custom_llm_provider

            response = avector_store_create_sdk(
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
                f"litellm.avector_store_create(model={model_name})\033[32m 200 OK\033[0m"
            )

            return response
        except Exception as e:
            verbose_router_logger.exception(
                f"litellm.avector_store_create(model={model})\033[31m 异常 {str(e)}\033[0m"
            )
            if model is not None:
                self.fail_calls[model] += 1
            raise e

    def _override_vector_store_methods_for_router(self):
        """
        用 router 感知的实现覆盖由工厂生成的向量存储相关方法。
        在 _initialize_vector_store_endpoints() 之后调用，确保所使用的是
        我们自定义的、能够处理 deployment 选择和凭证注入的方法，
        而不是通用的工厂生成方法。
        """
        # 保留对上面定义的自定义方法的引用
        # 这些方法已经处理好了经由 deployment 的路由逻辑
        pass  # 这些方法已在上方作为实例方法定义

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
                f"进入 _acreate_batch() — model: {model}; kwargs: {kwargs}"
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

            ## 将 custom provider 设为所选中 deployment 的值 ##
            custom_llm_provider = data.get("custom_llm_provider")
            _, inferred_custom_llm_provider, _, _ = get_llm_provider(
                model=data["model"],
                custom_llm_provider=custom_llm_provider,
            )
            custom_llm_provider = custom_llm_provider or inferred_custom_llm_provider

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
                    - 调用前先检查 rpm 限额
                    - 如果通过检查，自增 rpm 计数（修改全局值，并发安全）
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
                f"litellm._acreate_batch(model={model}, {kwargs})\033[31m 异常 {str(e)}\033[0m"
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
        遍历模型组中的所有模型，查找对应的 batch。

        未来改进 —— 缓存查找结果。
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
                raise Exception("Router 尚未初始化。")

            receieved_exceptions = []

            async def try_retrieve_batch(model_name: DeploymentTypedDict):
                try:
                    from litellm.litellm_core_utils.core_helpers import safe_deep_copy

                    model = model_name["litellm_params"].get("model")
                    data = model_name["litellm_params"].copy()
                    custom_llm_provider = data.get("custom_llm_provider")
                    if model is None:
                        raise Exception(
                            f"deployment 的 litellm_params 中未找到 model：{model_name}"
                        )
                    # 根据当前模型名，更新 kwargs 以包含模型专有的调整
                    ## 将 custom provider 设为所选中 deployment 的值 ##
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

            # 并行查询所有模型
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
                raise Exception("未找到任何健康的 deployment。")

            # 检查成功响应并处理异常
            if results is not None:
                if isinstance(results, LiteLLMBatch):
                    return results
                elif isinstance(results, list):
                    for result in results:
                        if isinstance(result, LiteLLMBatch):
                            return result

            # 如果未找到有效的 Batch 响应，就抛出遇到的第一个异常
            if receieved_exceptions:
                raise receieved_exceptions[0]  # 抛出遇到的第一个异常

            # 如果没有收集到异常，则抛出一个通用异常
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

    async def acancel_batch(
        self,
        model: str,
        **kwargs,
    ) -> LiteLLMBatch:
        """
        通过 router 取消一个 batch，同时维护正确的模型到 provider 的映射。
        """
        try:
            kwargs["model"] = model
            kwargs["original_function"] = self._acancel_batch
            kwargs["num_retries"] = kwargs.get("num_retries", self.num_retries)
            metadata_variable_name = _get_router_metadata_variable_name(
                function_name="_acancel_batch"
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

    async def _acancel_batch(
        self,
        model: str,
        **kwargs,
    ) -> LiteLLMBatch:
        try:
            verbose_router_logger.debug(
                f"进入 _acancel_batch() — model: {model}; kwargs: {kwargs}"
            )
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "batch-api-fake-text"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
                request_kwargs=kwargs,
            )

            data = deployment["litellm_params"].copy()
            model_name = data["model"]
            self._update_kwargs_with_deployment(
                deployment=deployment, kwargs=kwargs, function_name="_acancel_batch"
            )

            model_client = self._get_async_openai_model_client(
                deployment=deployment,
                kwargs=kwargs,
            )
            self.total_calls[model_name] += 1

            ## 将 custom provider 设为所选中 deployment 的值 ##
            custom_llm_provider = data.get("custom_llm_provider")
            _, inferred_custom_llm_provider, _, _ = get_llm_provider(
                model=data["model"],
                custom_llm_provider=custom_llm_provider,
            )
            custom_llm_provider = custom_llm_provider or inferred_custom_llm_provider

            response = litellm.acancel_batch(
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
                    - 调用前先检查 rpm 限额
                    - 如果通过检查，自增 rpm 计数（修改全局值，并发安全）
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
                f"litellm.acancel_batch(model={model_name})\033[32m 200 OK\033[0m"
            )

            return response  # type: ignore
        except Exception as e:
            verbose_router_logger.exception(
                f"litellm._acancel_batch(model={model}, {kwargs})\033[31m 异常 {str(e)}\033[0m"
            )
            if model is not None:
                self.fail_calls[model] += 1
            raise e

    async def alist_batches(
        self,
        model: str,
        **kwargs,
    ):
        """
        返回一个模型组下所有 deployment 的 batch。
        """

        filtered_model_list = self.get_model_list(model_name=model)
        if filtered_model_list is None:
            raise Exception("Router 尚未初始化。")

        async def try_retrieve_batch(model: DeploymentTypedDict):
            try:
                # 根据当前模型名，更新 kwargs 以包含模型专有的调整
                return await litellm.alist_batches(
                    **{**model["litellm_params"], **kwargs}
                )
            except Exception:
                return None

        # 并行查询所有模型
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
                ## 检查 batch id
                if final_results["first_id"] is None and hasattr(result, "first_id"):
                    final_results["first_id"] = getattr(result, "first_id")
                final_results["last_id"] = getattr(result, "last_id")
                final_results["data"].extend(result.data)  # type: ignore

                ## 检查 'has_more'
                if getattr(result, "has_more", False) is True:
                    final_results["has_more"] = True

        return final_results

    #### 透传（PASSTHROUGH）API ####

    async def _pass_through_moderation_endpoint_factory(
        self,
        original_function: Callable,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        # 将 model_group 添加进 kwargs
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
            "acompact_responses",
            "responses",
            "aget_responses",
            "adelete_responses",
            "afile_delete",
            "afile_content",
            "_arealtime",
            "_aresponses_websocket",
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
            "avector_store_retrieve",
            "avector_store_list",
            "avector_store_update",
            "avector_store_delete",
            "avector_store_file_create",
            "avector_store_file_list",
            "avector_store_file_retrieve",
            "avector_store_file_content",
            "avector_store_file_update",
            "avector_store_file_delete",
            "vector_store_search",
            "vector_store_create",
            "vector_store_retrieve",
            "vector_store_list",
            "vector_store_update",
            "vector_store_delete",
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
            "avideo_create_character",
            "video_create_character",
            "avideo_get_character",
            "video_get_character",
            "avideo_edit",
            "video_edit",
            "avideo_extension",
            "video_extension",
            "acreate_container",
            "create_container",
            "alist_containers",
            "list_containers",
            "aretrieve_container",
            "retrieve_container",
            "adelete_container",
            "delete_container",
            "aupload_container_file",
            "upload_container_file",
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
        针对不同的 API 调用类型，创建合适的包装函数。

        返回：
            - 对于同步调用类型，返回同步函数
            - 对于异步调用类型，返回异步函数
        """
        # 处理同步调用类型
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
            "vector_store_retrieve",
            "vector_store_list",
            "vector_store_update",
            "vector_store_delete",
        ):

            def vector_store_sync_wrapper(
                custom_llm_provider: Optional[str] = None,
                client: Optional[Any] = None,
                **kwargs,
            ):
                if custom_llm_provider and "custom_llm_provider" not in kwargs:
                    kwargs["custom_llm_provider"] = custom_llm_provider
                if kwargs.get("model"):
                    return self._generic_api_call_with_fallbacks(
                        original_function=original_function, **kwargs
                    )
                return original_function(**kwargs)

            return vector_store_sync_wrapper

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

        # 处理异步调用类型
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
                "_aresponses_websocket",
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
                "avideo_create_character",
                "avideo_get_character",
                "avideo_edit",
                "avideo_extension",
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
                "aupload_container_file",
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
                "acompact_responses",
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
                "avector_store_retrieve",
                "avector_store_list",
                "avector_store_update",
                "avector_store_delete",
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
        在 router 上初始化向量存储（Vector Store）API 端点。

        如果 kwargs 中传入了 model，则使用基于 model 的路由来获取 deployment 凭证；
        否则直接调用原始函数。
        """
        if custom_llm_provider and "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = custom_llm_provider

        # 若传入 model，则使用通用 API 调用+回退的逻辑走正确路由
        if kwargs.get("model"):
            return await self._ageneric_api_call_with_fallbacks(
                original_function=original_function,
                **kwargs,
            )

        # 否则直接调用原始函数
        return await original_function(**kwargs)

    async def _init_containers_api_endpoints(
        self,
        original_function: Callable,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        """
        在 router 上初始化 Containers API 端点。

        Container 相关操作不需要基于 model 的路由，所以我们直接带上 custom_llm_provider
        调用原始函数。
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
        在 router 上初始化 Responses API 端点。

        GET/DELETE/CANCEL 类 Responses API 请求会将 model_id 编码在 response_id 中，
        此函数会解码 response_id 并将 model 设置为解出的 model_id。
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
        在 router 上初始化 Interactions API 端点。

        GET/DELETE/CANCEL 类 Interactions API 请求不需要基于 model 的路由，
        所以我们直接带上 custom_llm_provider 调用原始函数。
        """
        if custom_llm_provider and "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = custom_llm_provider
        # Interactions API 默认使用 gemini
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
        """透传到 assistants 端点的内部辅助函数"""
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

    #### [结束] ASSISTANTS API ####

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
        async_function_with_fallbacks 的公共工具方法
        """
        verbose_router_logger.debug(f"异常堆栈：{traceback.format_exc()}")
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

        # 基于 order 的回退：把更高 order 层级添加到回退列表前面
        # 对于有专用回退处理器的异常类型，跳过此逻辑
        _skip_order_fallback = isinstance(
            e,
            (litellm.ContextWindowExceededError, litellm.ContentPolicyViolationError),
        )
        _request_team_id: Optional[str] = (
            kwargs.get("metadata", {}) or {}
        ).get("user_api_key_team_id")
        all_deployments = self._get_all_deployments(
            model_name=original_model_group, team_id=_request_team_id
        )
        _order_set: set = {
            litellm.utils._get_deployment_order(d)
            for d in all_deployments
            if litellm.utils._get_deployment_order(d) is not None
        }
        order_values: list = sorted(_order_set)
        if len(order_values) > 1 and not _skip_order_fallback:
            # 判断哪些 order 层级已经尝试过
            current_target = kwargs.get("_target_order")
            skip_up_to = (
                current_target if current_target is not None else order_values[0]
            )
            # 构建基于 order 的回退条目（跳过已试过的层级）
            order_fallback_entries: List = [
                {"model": original_model_group, "_target_order": o}
                for o in order_values
                if o > skip_up_to
            ]
            # 获取外部回退 —— 同时处理标准和非标准格式
            external_fallback_group: Optional[List] = None
            if fallbacks is not None and model_group is not None:
                if _check_non_standard_fallback_format(fallbacks=fallbacks):
                    # 非标准格式（例如 ["claude-3-haiku"] 或
                    # [{"model": "...", "messages": [...]}]）直接透传
                    external_fallback_group = fallbacks
                else:
                    external_fallback_group, generic_idx = get_fallback_model_group(
                        fallbacks=fallbacks,
                        model_group=cast(str, model_group),
                    )
                    if external_fallback_group is None and generic_idx is not None:
                        external_fallback_group = fallbacks[generic_idx]["*"]

            # 合并后的列表：order 回退在前，外部回退在后
            combined_fallbacks = order_fallback_entries + (
                external_fallback_group or []
            )

            if combined_fallbacks:
                input_kwargs.update(
                    {
                        "fallback_model_group": combined_fallbacks,
                        "original_model_group": original_model_group,
                    }
                )
                response = await run_async_fallback(
                    *args,
                    **input_kwargs,
                )
                return response

        try:
            verbose_router_logger.info("尝试在模型之间回退")

            # 检查是否使用了客户端定义的回退（例如 fallbacks = ["gpt-3.5-turbo", "claude-3-haiku"]，或 fallbacks=[{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hey, how's it going?"}]}]）
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
                        msg="捕获到 'ContextWindowExceededError'，但未配置 context_window_fallback，\
                        如有 fallbacks 则回退到其上。{}".format(
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
                        msg="捕获到 'ContentPolicyViolationError'，但未配置 content_policy_fallback，\
                        如有 fallbacks 则回退到其上。{}".format(
                            error_message
                        )
                    )

                    e.message += "\n{}".format(error_message)
            if fallbacks is not None and model_group is not None:
                verbose_router_logger.debug(f"进入 model fallbacks 分支：{fallbacks}")
                (
                    fallback_model_group,
                    generic_fallback_idx,
                ) = get_fallback_model_group(
                    fallbacks=fallbacks,  # 例如 fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}]
                    model_group=cast(str, model_group),
                )
                ## 若未命中具体配置，再查看是否有通用回退
                if fallback_model_group is None and generic_fallback_idx is not None:
                    fallback_model_group = fallbacks[generic_fallback_idx]["*"]

                if fallback_model_group is None:
                    verbose_router_logger.info(
                        f"未为原始模型组 model_group={model_group} 找到可用的回退模型组。Fallbacks={fallbacks}"
                    )
                    if hasattr(original_exception, "message"):
                        original_exception.message += f"未为原始模型组 model_group={model_group} 找到可用的回退模型组。Fallbacks={fallbacks}"  # type: ignore
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
                "litellm.router.py::async_function_with_fallbacks() — 执行回退时发生错误 — {}\n{}\n\n调试信息：\n当前冷却中的 Deployments={}".format(
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
            # 将可用的回退信息追加到异常信息中
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
        先尝试调用 function_with_retries；
        如果在 num_retries 轮重试后仍失败，则回退到另一个模型组。
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
            if verbose_router_logger.isEnabledFor(logging.DEBUG):
                verbose_router_logger.debug(f"异步响应：{response}")
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
        辅助函数：用于 mock 测试时主动抛出 litellm 错误。

        抛出异常的情况：
            litellm.InternalServerError: 请求参数中传入 `mock_testing_fallbacks=True` 时
            litellm.ContextWindowExceededError: 请求参数中传入 `mock_testing_context_fallbacks=True` 时
            litellm.ContentPolicyViolationError: 请求参数中传入 `mock_testing_content_policy_fallbacks=True` 时
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
        verbose_router_logger.debug("进入 async_function_with_retries。")
        original_function = kwargs.pop("original_function")
        fallbacks = kwargs.pop("fallbacks", self.fallbacks)
        parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
        context_window_fallbacks = kwargs.pop(
            "context_window_fallbacks", self.context_window_fallbacks
        )
        content_policy_fallbacks = kwargs.pop(
            "content_policy_fallbacks", self.content_policy_fallbacks
        )
        # 支持按请求级别覆盖 model_group_retry_policy（可来自 key/team 的设置）
        model_group_retry_policy = kwargs.pop(
            "model_group_retry_policy", self.model_group_retry_policy
        )
        model_group: Optional[str] = kwargs.get("model")
        num_retries = kwargs.pop("num_retries")

        ## 将 model group 大小写入 metadata ——用于 model_group_rate_limit_error 追踪
        _metadata: dict = kwargs.get("litellm_metadata", kwargs.get("metadata")) or {}
        if "model_group" in _metadata and isinstance(_metadata["model_group"], str):
            model_list = self.get_model_list(model_name=_metadata["model_group"])
            if model_list is not None:
                _metadata.update({"model_group_size": len(model_list)})

        verbose_router_logger.debug(
            f"异步函数重试：original_function={original_function}, num_retries={num_retries}"
        )
        ## 将重试追踪信息写入 metadata ——用于耗费日志的重试追踪
        _metadata["attempted_retries"] = 0
        _metadata["max_retries"] = (
            num_retries  # 在异常处理分支中被覆盖后再更新
        )
        try:
            self._handle_mock_testing_rate_limit_error(
                model_group=model_group, kwargs=kwargs
            )
            # 若调用成功，不会抛出异常，从而跳出循环
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
            重试逻辑
            """
            (
                _healthy_deployments,
                _all_deployments,
            ) = await self._async_get_healthy_deployments(
                model=kwargs.get("model") or "",
                parent_otel_span=parent_otel_span,
            )

            # 在 should_retry_this_error 之前先检查 retry policy
            # 这样 retry policy 可以覆盖“有无健康 deployment”的检查
            _retry_policy_applies = False
            if self.retry_policy is not None or model_group_retry_policy is not None:
                # 从 retry policy 中取 num_retries
                # 优先使用函数开头捕获到的 model_group，其次从 metadata 或 kwargs 中取
                # 此时 kwargs.get("model") 已经是具体 deployment 的模型名，而不是模型组名
                _model_group_for_retry_policy = (
                    model_group or _metadata.get("model_group") or kwargs.get("model")
                )
                # 如果请求级别传入了 model_group_retry_policy 则优先使用，否则使用 router 实例自带的
                _retry_policy_retries = _get_num_retries_from_retry_policy(
                    exception=original_exception,
                    model_group=_model_group_for_retry_policy,
                    model_group_retry_policy=model_group_retry_policy,
                    retry_policy=self.retry_policy,
                )
                if _retry_policy_retries is not None:
                    num_retries = _retry_policy_retries
                    _retry_policy_applies = True

            # 若该错误不应重试，则直接抛出异常
            # 若 retry policy 已生效，则跳过这一检查（retry policy 优先）
            if not _retry_policy_applies:
                self.should_retry_this_error(
                    error=e,
                    healthy_deployments=_healthy_deployments,
                    all_deployments=_all_deployments,
                    context_window_fallbacks=context_window_fallbacks,
                    regular_fallbacks=fallbacks,
                    content_policy_fallbacks=content_policy_fallbacks,
                )
            # 在经过 deployment_num_retries / retry_policy 覆盖之后更新 max_retries
            _metadata["max_retries"] = num_retries

            ## 日志记录
            if num_retries > 0:
                kwargs = self.log_retry(kwargs=kwargs, e=original_exception)
            else:
                raise

            verbose_router_logger.debug(
                f"即将重试请求，num_retries={num_retries}"
            )
            # 决定重试前要 sleep 多久
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
                    # 每次重试前更新重试追踪信息
                    _metadata["attempted_retries"] = current_attempt + 1
                    _metadata["max_retries"] = num_retries
                    # 若调用成功，不会抛出异常，从而跳出循环
                    response = await self.make_call(original_function, *args, **kwargs)
                    if coroutine_checker.is_async_callable(
                        response
                    ):  # 异步错误通常以 coroutine 形式返回
                        response = await response

                    response = add_retry_headers_to_response(
                        response=response,
                        attempted_retries=current_attempt + 1,
                        max_retries=num_retries,
                    )
                    return response

                except Exception as e:
                    # 始终记录最新遇到的错误，这样最终抛出的是最后一次失败的异常，
                    # 而不是最初的那一次。
                    original_exception = e

                    ## 日志记录
                    kwargs = self.log_retry(kwargs=kwargs, e=e)
                    remaining_retries = num_retries - current_attempt - 1
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

                    # 检查本次错误是否不可重试（例如 400 context window exceeded）。
                    # 若不可重试，则立即抛出异常而不继续重试循环。
                    # 同时尊重 retry policy 的优先级——仅在 retry policy 不适用时才做此检查。
                    if not _retry_policy_applies:
                        try:
                            self.should_retry_this_error(
                                error=e,
                                healthy_deployments=_healthy_deployments,
                                all_deployments=_all_deployments,
                                context_window_fallbacks=context_window_fallbacks,
                                regular_fallbacks=fallbacks,
                                content_policy_fallbacks=content_policy_fallbacks,
                            )
                        except Exception:
                            raise e

                    _timeout = self._time_to_sleep_before_retry(
                        e=e,
                        remaining_retries=remaining_retries,
                        num_retries=num_retries,
                        healthy_deployments=_healthy_deployments,
                        all_deployments=_all_deployments,
                    )
                    await asyncio.sleep(_timeout)

            if type(original_exception) in litellm.LITELLM_EXCEPTION_TYPES:
                setattr(original_exception, "max_retries", num_retries)
                # current_attempt 是从 0 开始的索引（0 到 num_retries-1），所以循环结束后
                # 它表示最后一次尝试的索引。实际已重试次数 = current_attempt + 1，
                # 当所有重试用尽时它正好等于 num_retries。
                # 因为在进入循环前已经验证 num_retries > 0，所以走到这里时
                # current_attempt 一定会被赋值，不会为 None。
                actual_retries_attempted = (
                    current_attempt + 1 if current_attempt is not None else num_retries
                )
                setattr(original_exception, "num_retries", actual_retries_attempted)

            raise original_exception

    async def make_call(self, original_function: Any, *args, **kwargs):
        """
        调用 .completion()/.embeddings()/等函数的统一入口。
        """
        model_group = kwargs.get("model")
        response = original_function(*args, **kwargs)
        if coroutine_checker.is_async_callable(response) or inspect.isawaitable(
            response
        ):
            response = await response
        ## 处理响应头
        response = await self.set_response_headers(
            response=response, model_group=model_group
        )

        return response

    def _handle_mock_testing_rate_limit_error(
        self, kwargs: dict, model_group: Optional[str] = None
    ):
        """
        辅助函数：用于测试时主动抛出模拟的 litellm.RateLimitError 错误。

        抛出异常的情况：
            请求参数中传入 `mock_testing_rate_limit_error=True` 时抛出 litellm.RateLimitError
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
                f"litellm.router.py::_mock_rate_limit_error() — 为 model={model_group} 抛出模拟的 RateLimitError"
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
        1. 若 context_window_fallbacks 不为 None，碰到 ContextWindowExceededError 时抛出异常
        2. 若 content_policy_fallbacks 不为 None，碰到 ContentPolicyViolationError 时抛出异常

        2. 碰到 RateLimitError 时，如果满足以下任一条件则抛出：
            - 没有配置 fallbacks
            - 同一模型组内没有健康的 deployment
        """
        _num_healthy_deployments = 0
        if healthy_deployments is not None and isinstance(healthy_deployments, list):
            _num_healthy_deployments = len(healthy_deployments)

        _num_all_deployments = 0
        if all_deployments is not None and isinstance(all_deployments, list):
            _num_all_deployments = len(all_deployments)

        ### 判断是否为 RateLimit / ContextWindowExceeded / ContentPolicyViolation（且已配置 fallbacks）/ BadRequest 等错误
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

        status_code = getattr(error, "status_code", None)
        if status_code is not None and not litellm._should_retry(status_code):
            # 401/403 是特殊情况：如果多个 deployment 可用则允许重试（在下方处理）
            if status_code not in (401, 403):
                raise error

        if isinstance(error, litellm.NotFoundError):
            raise error
        # 仅在有其他 deployment 可用时才重试的错误
        if isinstance(error, openai.RateLimitError):
            if (
                _num_healthy_deployments <= 0  # 如果没有健康的 deployment
                and regular_fallbacks is not None  # 且没有配置 fallbacks
                and len(regular_fallbacks) > 0
            ):
                raise error  # 则抛出异常

        if isinstance(error, openai.AuthenticationError):
            """
            - 如果该模型组还有其他 deployment，则重试
            - 否则直接抛出异常
            """
            if (
                _num_all_deployments <= 1
            ):  # 若该模型组只有 1 个 deployment 则不重试
                raise error  # 抛出异常

        # 如果没有健康的 deployment，直接抛出异常
        if _num_healthy_deployments <= 0:  # 没有健康的 deployment
            raise error

        return True

    def function_with_fallbacks(self, *args, **kwargs):
        """
        async_function_with_fallbacks 的同步包装器。

        包装的目的是减少重复代码、防止维护时出现 bug。
        """
        return run_async_function(self.async_function_with_fallbacks, *args, **kwargs)

    def _get_fallback_model_group_from_fallbacks(
        self,
        fallbacks: List[Dict[str, List[str]]],
        model_group: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        返回指定模型组对应的回退模型列表。

        如果未找到对应的回退模型组，返回 None。

        示例：
            fallbacks = [{"gpt-3.5-turbo": ["gpt-4"]}, {"gpt-4o": ["gpt-3.5-turbo"]}]
            model_group = "gpt-3.5-turbo"
            returns: ["gpt-4"]
        """
        if model_group is None:
            return None

        fallback_model_group: Optional[List[str]] = None
        for item in fallbacks:  # 例如 [{"gpt-3.5-turbo": ["gpt-4"]}]
            if list(item.keys())[0] == model_group:
                fallback_model_group = item[model_group]
                break
        return fallback_model_group

    def _get_first_default_fallback(self) -> Optional[str]:
        """
        如果 default_fallbacks 存在，返回其中的第一个模型。
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
        计算退避时间，然后重试。

        只有在下面两种情形下才应立即重试：
            1. 同一个模型组中还有健康的 deployment
            2. completion 调用配置了 fallbacks
        """

        ## 基础情形 —— 只有 1 个 deployment
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

    ### 辅助函数

    async def deployment_callback_on_success(
        self,
        kwargs,  # completion 的 kwargs
        completion_response,  # completion 返回的响应
        start_time,
        end_time,  # 开始/结束时间
    ):
        """
        追踪 model_list 中每个模型剩余的 tpm/rpm 额度
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
                )  # 稳定名称，即便对于通配符路由也能正确工作
                # 和同步版本一样，从 kwargs 中获取 model_group 和 id
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
                        deployment=deployment_info,
                        received_model_name=model_group,
                    )
                    # 从 deployment 信息中获取 tpm/rpm
                    tpm = deployment_info.get("tpm", None)
                    rpm = deployment_info.get("rpm", None)

                    ## 检查 litellm_params 中的 tpm/rpm
                    tpm_litellm_params = deployment_info.litellm_params.tpm
                    rpm_litellm_params = deployment_info.litellm_params.rpm

                    ## 检查 model_info 中的 tpm/rpm
                    tpm_model_info = deployment_model_info.get("tpm", None)
                    rpm_model_info = deployment_model_info.get("rpm", None)

                # 不管是否设置了 TPM/RPM 限额，都始终追踪 deployment 的成功次数，供冷却逻辑使用
                increment_deployment_successes_for_current_minute(
                    litellm_router_instance=self,
                    deployment_id=id,
                )

                ## 如果全为 None，则返回 —— 未设置 tpm/rpm 的模型无需追踪当前用量
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
                # 准备各种值
                # ------------
                dt = get_utc_datetime()
                current_minute = dt.strftime(
                    "%H-%M"
                )  # 无论系统时区如何，统一使用 UTC 时间

                tpm_key = RouterCacheEnum.TPM.value.format(
                    id=id, current_minute=current_minute, model=deployment_name
                )
                # ------------
                # 更新用量
                # ------------
                # 更新缓存
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
                "litellm.router.Router::deployment_callback_on_success() — 发生异常：{}".format(
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
        使用内存缓存追踪当前分钟内某个 deployment 的成功次数。

        返回：
        - key: str，用于自增缓存的 key
        - None：未找到 key 时
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
        该方法有两个职责：
        - 使用内存缓存追踪当前分钟内某个 deployment 的失败次数
        - 若超过每分钟允许的失败次数，将该 deployment 置于冷却状态

        返回：
        - True：应将该 deployment 置于冷却状态
        - False：不应将该 deployment 置于冷却状态
        """
        verbose_router_logger.debug("Router：进入 'deployment_callback_on_failure'")
        try:
            exception = kwargs.get("exception", None)
            exception_status = getattr(exception, "status_code", "")

            # 缓存 litellm_params，避免重复的字典查找
            litellm_params = kwargs.get("litellm_params", {})
            _model_info = litellm_params.get("model_info", {})

            exception_headers = litellm.litellm_core_utils.exception_mapping_utils._get_response_headers(
                original_exception=exception
            )

            # 确定冷却时间，优先级从高到低：deployment 配置 > 响应头 > router 默认值
            deployment_cooldown = litellm_params.get("cooldown_time", None)

            header_cooldown = None
            if exception_headers is not None:
                header_cooldown = litellm.utils._get_retry_after_from_exception_header(
                    response_headers=exception_headers
                )
            ##############################################
            # 冷却时间的决策逻辑
            # 1. 优先查看 deployment 配置中是否设置了冷却时间
            # 2. 其次查看响应头中是否返回了冷却时间
            # 3. 都没有时则使用 router 的默认冷却时间
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
                )  # 将 deployment_id 写入冷却清单

                return result
            else:
                verbose_router_logger.debug(
                    "Router：未找到 model_info，退出 'deployment_callback_on_failure' 且未进入冷却。"
                )
                return False

        except Exception as e:
            raise e

    async def async_deployment_callback_on_failure(
        self, kwargs, completion_response: Optional[Any], start_time, end_time
    ):
        """
        更新 deployment 的 RPM 使用量
        """
        deployment_name = kwargs["litellm_params"]["metadata"].get(
            "deployment", None
        )  # 兼容通配符路由 ——返回的是当初传给 `litellm.completion` 的原始名称
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
        )  # 无论系统时区如何，统一使用 UTC 时间

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
        辅助方法：返回请求数据中“元数据字段”应该使用的名称。

        - 新端点返回 `litellm_metadata`
        - 旧端点返回 `metadata`

        背景：
        - LiteLLM 最初使用 `metadata` 作为内部存储元数据的字段
        - 之后 OpenAI 开始用该字段保存它们的元数据
        - 现在 LiteLLM 已逐步切换为使用 `litellm_metadata` 来保存我们自己的元数据
        """
        return get_metadata_variable_name_from_kwargs(kwargs)

    def log_retry(self, kwargs: dict, e: Exception) -> dict:
        """
        发生重试或回退时，记录刚刚失败的模型调用细节 —— 类似 Sentry 的 breadcrumb。
        """
        try:
            _metadata_var = (
                "litellm_metadata" if "litellm_metadata" in kwargs else "metadata"
            )
            # 将失败的模型记录为“上一个模型”
            previous_model = {
                "exception_type": type(e).__name__,
                "exception_string": str(e),
            }
            for (
                k,
                v,
            ) in (
                kwargs.items()
            ):  # 除了旧的 previous_models 值以外，将 kwargs 中全部东西写入日志——避免嵌套
                if k not in [_metadata_var, "messages", "original_function"]:
                    previous_model[k] = v
                elif k == _metadata_var and isinstance(v, dict):
                    previous_model[_metadata_var] = {}  # type: ignore
                    for metadata_k, metadata_v in kwargs[_metadata_var].items():
                        if metadata_k != "previous_models":
                            previous_model[k][metadata_k] = metadata_v  # type: ignore

            # 检查 self.previous_models 的长度，超过 3 时移除最前面那个
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
        更新当前分钟内某个 deployment 的 rpm。

        返回：
        - int：当前请求计数
        """
        rpm_key = deployment_id

        request_count = self.cache.get_cache(
            key=rpm_key, parent_otel_span=parent_otel_span, local_only=True
        )
        if request_count is None:
            request_count = 1
            self.cache.set_cache(
                key=rpm_key, value=request_count, local_only=True, ttl=60
            )  # 只保存 60 秒
        else:
            request_count += 1
            self.cache.set_cache(
                key=rpm_key, value=request_count, local_only=True
            )  # 不修改现有的 ttl

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
        判断是否应抛出 content policy 错误。

        仅在存在可用的回退方案时才抛出。
        否则直接返回原始响应。
        """
        if response.choices and len(response.choices) > 0:
            if response.choices[0].finish_reason != "content_filter":
                return False

        content_policy_fallbacks = kwargs.get(
            "content_policy_fallbacks", self.content_policy_fallbacks
        )

        ### 仅当存在 content policy fallback 时才抛出异常 ###
        if content_policy_fallbacks is not None:
            fallback_model_group = None
            for item in content_policy_fallbacks:  # 例如 [{"gpt-3.5-turbo": ["gpt-4"]}]
                if list(item.keys())[0] == model:
                    fallback_model_group = item[model]
                    break

            if fallback_model_group is not None:
                return True
        elif self._has_default_fallbacks():  # 设置了默认回退
            return True

        verbose_router_logger.debug(
            "碰到 Content Policy 错误，但无可用回退，返回原始响应。model={}, content_policy_fallbacks={}".format(
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
        返回一个元组：
        - Tuple[List[Dict], List[Dict]]:
            1. healthy_deployments：健康 deployment 的列表
            2. all_deployments：所有 deployment 的列表
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
        模仿 'async_routing_strategy_pre_call_checks' 的同步版本。

        用于 'usage-based-routing-v2' 中保持 rpm 更新逻辑的一致性。

        返回：
        - None

        抛出：
        - RateLimit 异常：如果 deployment 超出其 tpm/rpm 限额
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
        用于 usage-based-routing-v2：在信号量内、发起调用前执行 rpm 检查。

        -> 当为 deployment 设置了 rpm 限额时，可以保证调用的并发安全。

        返回：
        - None

        抛出：
        - RateLimit 异常：如果 deployment 超出其 tpm/rpm 限额
        """
        for _callback in litellm.callbacks:
            if isinstance(_callback, CustomLogger):
                try:
                    await _callback.async_pre_call_check(deployment, parent_otel_span)
                except litellm.RateLimitError as e:
                    ## 记录失败事件
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
                        ).start()  # 记录响应
                    _set_cooldown_deployments(
                        litellm_router_instance=self,
                        exception_status=e.status_code,
                        original_exception=e,
                        deployment=deployment["model_info"]["id"],
                        time_to_cooldown=self.cooldown_time,
                    )
                    raise e
                except Exception as e:
                    ## 记录失败事件
                    if logging_obj is not None:
                        asyncio.create_task(
                            logging_obj.async_failure_handler(
                                exception=e,
                                traceback_exception=traceback.format_exc(),
                                end_time=time.time(),
                            )
                        )
                        ## 日志
                        threading.Thread(
                            target=logging_obj.failure_handler,
                            args=(e, traceback.format_exc()),
                        ).start()  # 记录响应
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
        用于 usage-based-routing-v2：在信号量内、发起调用前执行 rpm 检查。

        -> 当为 deployment 设置了 rpm 限额时，可以保证调用的并发安全。

        返回：
        - None

        抛出：
        - RateLimit 异常：如果 deployment 超出其 tpm/rpm 限额
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
                    ## 记录失败事件
                    if logging_obj is not None:
                        asyncio.create_task(
                            logging_obj.async_failure_handler(
                                exception=e,
                                traceback_exception=traceback.format_exc(),
                                end_time=time.time(),
                            )
                        )
                        ## 日志
                        threading.Thread(
                            target=logging_obj.failure_handler,
                            args=(e, traceback.format_exc()),
                        ).start()  # 记录响应
                    raise e
        return returned_healthy_deployments

    def _generate_model_id(self, model_group: str, litellm_params: dict):
        """
        辅助函数：为同一个 deployment 一致性地生成相同的 id。

        - 把所有 litellm 参数拼成一个字符串
        - 做哈希
        - 使用哈希值作为 id
        """
        # 性能优化：在循环中使用 list + join 代替字符串拼接
        # 避免创建大量临时字符串对象（复杂度 O(n) 而非 O(n²)）
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
        创建一个 deployment 对象，并将其加入 model_list。

        如果该 deployment 在当前环境下未启用，则会被忽略。

        返回：
        - Deployment：创建的 deployment 对象
        - None：当该 deployment 在当前环境下未启用时（即 litellm_params 中配置了 'supported_environments'）
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

            ## 将 model info 注册到 litellm 的 model cost 映射表中
            model_id = deployment.model_info.id
            if model_id is not None:
                litellm.register_model(
                    model_cost={
                        model_id: _model_info,
                    }
                )

            ## 旧版模型注册逻辑 ## 保留以避免破坏兼容性
            _model_name = deployment.litellm_params.model
            if deployment.litellm_params.custom_llm_provider is not None:
                _model_name = (
                    deployment.litellm_params.custom_llm_provider + "/" + _model_name
                )

            # 对于共享的后端模型 key，剥离 custom pricing 字段，
            # 防止一个 deployment 的定价覆盖污染到使用相同后端模型名的
            # 另一个 deployment。
            # 每个 deployment 的完整定价信息已经通过上面其唯一 model_id 保存。
            _custom_pricing_fields = CustomPricingLiteLLMParams.model_fields.keys()
            _shared_model_info = {
                k: v for k, v in _model_info.items() if k not in _custom_pricing_fields
            }
            litellm.register_model(
                model_cost={
                    _model_name: _shared_model_info,
                }
            )

            ## 检查该 LLM Deployment 在当前环境下是否被允许
            if (
                self.deployment_is_active_for_environment(deployment=deployment)
                is not True
            ):
                verbose_router_logger.warning(
                    f"忽略 deployment {deployment.model_name}，因为它在当前环境下未启用：{deployment.model_info['supported_environments']}"
                )
                return None

            # 在添加 deployment 之前先校验 tag_regex 的正则表达式，
            # 这样一来，即使某个 pattern 无效，也不会让 router 处于部分初始化的状态。
            _tag_regex = deployment.litellm_params.get("tag_regex") or []
            for pattern in _tag_regex:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"model '{deployment.model_name}' 的 tag_regex 中存在无效正则表达式："
                        f"{pattern!r} — {exc}"
                    ) from exc

            deployment = self._add_deployment(deployment=deployment)

            model = deployment.to_json(exclude_none=True)

            self._add_model_to_list_and_index_map(
                model=model, model_id=deployment.model_info.id
            )
            return deployment
        except Exception as e:
            if self.ignore_invalid_deployments:
                verbose_router_logger.exception(
                    f"创建 deployment 失败：{e}，忽略该 deployment 并继续处理其余 deployment。"
                )
                return None
            else:
                raise e

    def _is_auto_router_deployment(self, litellm_params: LiteLLM_Params) -> bool:
        """
        判断该 deployment 是否为 auto-router（基于语义的路由）deployment。

        当 litellm_params.model 以 "auto_router/" 开头且
        不以 "auto_router/complexity_router" 开头时返回 True（后者走 complexity 路由）。
        """
        if litellm_params.model.startswith("auto_router/complexity_router"):
            return False  # This is handled by complexity_router
        if litellm_params.model.startswith("auto_router/"):
            return True
        return False

    def init_auto_router_deployment(self, deployment: Deployment):
        """
        初始化 auto-router 类型的 deployment。

        会创建 auto-router 实例并将其加入 self.auto_routers 字典。
        """
        from litellm.router_strategy.auto_router.auto_router import AutoRouter

        auto_router_config_path: Optional[str] = (
            deployment.litellm_params.auto_router_config_path
        )
        auto_router_config: Optional[str] = deployment.litellm_params.auto_router_config
        if auto_router_config_path is None and auto_router_config is None:
            raise ValueError(
                "auto-router 类型的 deployment 必须配置 auto_router_config_path 或 auto_router_config，请在 litellm_params 中设置。"
            )

        default_model: Optional[str] = (
            deployment.litellm_params.auto_router_default_model
        )
        if default_model is None:
            raise ValueError(
                "auto-router 类型的 deployment 必须配置 auto_router_default_model，请在 litellm_params 中设置。"
            )

        embedding_model: Optional[str] = (
            deployment.litellm_params.auto_router_embedding_model
        )
        if embedding_model is None:
            raise ValueError(
                "auto-router 类型的 deployment 必须配置 auto_router_embedding_model，请在 litellm_params 中设置。"
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
                f"auto-router deployment {deployment.model_name} 已经存在，请使用不同的 model_name。"
            )
        self.auto_routers[deployment.model_name] = autor_router

    def _is_complexity_router_deployment(self, litellm_params: LiteLLM_Params) -> bool:
        """
        判断该 deployment 是否为 complexity-router（基于复杂度的路由）deployment。

        当 litellm_params.model 以 "auto_router/complexity_router" 开头时返回 True。
        """
        if litellm_params.model.startswith("auto_router/complexity_router"):
            return True
        return False

    def init_complexity_router_deployment(self, deployment: Deployment):
        """
        初始化 complexity-router 类型的 deployment。

        会创建 complexity-router 实例并将其加入 self.complexity_routers 字典。
        """
        # 在函数内导入以避免循环引用 —— ComplexityRouter 是 CustomLogger 的子类，
        # 它会 import 依赖 router.py 的 litellm 内部模块。
        # 这里的处理方式与上面 init_auto_router_deployment 中的 AutoRouter 相同。
        from litellm.router_strategy.complexity_router.complexity_router import (
            ComplexityRouter,
        )

        complexity_router_config: Optional[dict] = (
            deployment.litellm_params.complexity_router_config
        )

        default_model: Optional[str] = (
            deployment.litellm_params.complexity_router_default_model
        )

        # 如果没有指定默认模型，尝试从 config 的 tiers 中取
        if default_model is None and complexity_router_config:
            tiers = complexity_router_config.get("tiers", {})
            # 使用 MEDIUM 层级作为兜底的默认值
            default_model = tiers.get("MEDIUM") or tiers.get("SIMPLE")

        if default_model is None:
            raise ValueError(
                "complexity-router 类型的 deployment 必须配置 complexity_router_default_model，"
                "或在 complexity_router_config 中配置 tiers，请在 litellm_params 中设置。"
            )

        complexity_router: ComplexityRouter = ComplexityRouter(
            model_name=deployment.model_name,
            default_model=default_model,
            litellm_router_instance=self,
            complexity_router_config=complexity_router_config,
        )
        if deployment.model_name in self.complexity_routers:
            raise ValueError(
                f"complexity-router deployment {deployment.model_name} 已经存在，请使用不同的 model_name。"
            )
        self.complexity_routers[deployment.model_name] = complexity_router

    def deployment_is_active_for_environment(self, deployment: Deployment) -> bool:
        """
        判断某个 llm deployment 在当前环境下是否启用。可以在多个环境中复用同一份 config.yaml。

        需要在 .env 中设置 `LITELLM_ENVIRONMENT`。合法的取值：
            - development
            - staging
            - production

        抛出：
        - ValueError：.env 中未设置 LITELLM_ENVIRONMENT，或其值不在合法集合内
        - ValueError：model_info 中未设置 supported_environments，或其中的值不在合法集合内
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
                "已为模型设置 'supported_environments'，但 .env 中未设置 'LITELLM_ENVIRONMENT'。"
            )

        if litellm_environment not in VALID_LITELLM_ENVIRONMENTS:
            raise ValueError(
                f"LITELLM_ENVIRONMENT 必须是 {VALID_LITELLM_ENVIRONMENTS} 中的一个，但当前值是：{litellm_environment}"
            )

        for _env in deployment.model_info["supported_environments"]:
            if _env not in VALID_LITELLM_ENVIRONMENTS:
                raise ValueError(
                    f"supported_environments 中的每个值都必须属于 {VALID_LITELLM_ENVIRONMENTS}，但对于 deployment {deployment}，出现了非法值：{_env}"
                )

        if litellm_environment in deployment.model_info["supported_environments"]:
            return True
        return False

    def set_model_list(self, model_list: list):
        original_model_list = copy.deepcopy(model_list)
        self.model_list = []
        self.model_id_to_deployment_index_map = {}  # 重置索引
        self.model_name_to_deployment_indices = {}  # 重置 model_name 索引
        self.team_model_to_deployment_indices = {}  # 重置 team_model 索引
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()
        # 给每个模型加上 api_base/api_key，这样在 api_base1、api_base2 上对 azure/gpt 做负载均衡才能正常工作

        for model in original_model_list:
            _model_name = model.pop("model_name")
            _litellm_params = model.pop("litellm_params")
            ## 检查 litellm 参数中是否引用了 os.environ
            if isinstance(_litellm_params, dict):
                for k, v in _litellm_params.items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        _litellm_params[k] = get_secret(v)

            _model_info: dict = model.pop("model_info", {})

            # 检查 model_info 中是否已经有 id
            if "id" not in _model_info:
                _id = self._generate_model_id(_model_name, _litellm_params)
                _model_info["id"] = _id

            if _litellm_params.get("organization", None) is not None and isinstance(
                _litellm_params["organization"], list
            ):  # 修复 https://github.com/BerriAI/litellm/issues/3949
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
            f"\n已初始化 Model List：{self.get_model_names()}"
        )
        self.model_names = {m["model_name"] for m in model_list}

        # 注：model_name_to_deployment_indices 已经通过
        # _create_deployment -> _add_model_to_list_and_index_map 逐条建立好了

    def _add_deployment(self, deployment: Deployment) -> Deployment:
        import os

        #### 校验模型 ########
        # 在按照 LLM provider 做校验前，先判断是否为 prompt 管理模型
        litellm_model = deployment.litellm_params.model
        is_prompt_management_model = False

        if "/" in litellm_model:
            split_litellm_model = litellm_model.split("/")[0]
            if split_litellm_model in litellm._known_custom_logger_compatible_callbacks:
                is_prompt_management_model = True

        if is_prompt_management_model:
            # 对于 prompt 管理模型，跳过 LLM provider 校验
            # 实际使用的模型名将在运行时从 prompt 文件中解析
            _model = litellm_model
            custom_llm_provider = None
            dynamic_api_key = None
            api_base = None
        else:
            # 检查 provider 是否在支持列表中
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
            # model["litellm_params"] 解析完毕
            # 检查 provider 是否被支持：位于枚举列表或通过 JSON 配置注册
            if (
                custom_llm_provider not in litellm.provider_list
                and not JSONProviderRegistry.exists(custom_llm_provider)
            ):
                raise Exception(f"不支持的 provider - {custom_llm_provider}")

        #### 初始化 DEPLOYMENT NAMES ########
        self.deployment_names.append(deployment.litellm_params.model)
        ############ 用户可以将 tpm/rpm 作为 litellm_param 传入，也可以作为 router 参数 ###########
        # 在 get_available_deployment 中，使用 litellm_param["rpm"]
        # 这一段逻辑也会确保 rpm 被设置为一个 litellm_param
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

        # 检查用户是否试图使用 model_name == "*"
        # 这是针对该 api key 的通配模型
        # if deployment.model_name == "*":
        #     if deployment.litellm_params.model == "*":
        #         # 用户希望将未知 deployment 的所有请求透传给 litellm.acompletion
        #         self.router_general_settings.pass_through_all_models = True
        #     else:
        #         self.default_deployment = deployment.to_json(exclude_none=True)
        # 检查用户是否使用了 provider 级别的通配路由
        # 例如 model_name = "databricks/*" 或 model_name = "anthropic/*"
        if "*" in deployment.model_name:
            # 将其存为正则表达式，所有命中该 pattern 的 deployment 都会被路由到这里
            # 将 deployment.model_name 当作正则 pattern 存储
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

        # Azure GPT-Vision 增强功能配置，用户可以使用 os.environ/ 引用环境变量
        data_sources = deployment.litellm_params.get("dataSources", []) or []

        for data_source in data_sources:
            params = data_source.get("parameters", {})
            for param_key in ["endpoint", "key"]:
                # 如果 Azure GPT Vision Enhancements 中设置了 endpoint 或 key，检查是否使用了环境变量
                if param_key in params and params[param_key].startswith("os.environ/"):
                    env_name = params[param_key].replace("os.environ/", "")
                    params[param_key] = os.environ.get(env_name, "")

        # # 初始化 OpenAI / Azure 客户端
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
        # 判断该 deployment 是否为 auto-router
        #########################################################
        if self._is_auto_router_deployment(litellm_params=deployment.litellm_params):
            self.init_auto_router_deployment(deployment=deployment)

        #########################################################
        # 判断该 deployment 是否为 complexity-router
        #########################################################
        if self._is_complexity_router_deployment(
            litellm_params=deployment.litellm_params
        ):
            self.init_complexity_router_deployment(deployment=deployment)

        return deployment

    def _initialize_deployment_for_pass_through(
        self, deployment: Deployment, custom_llm_provider: str, model: str
    ):
        """
        可选：当 `deployment.litellm_params.use_in_pass_through` 为 True 时，为透传端点初始化 deployment。

        不同 provider 的透传端点使用不同的 .env 变量，这个辅助方法会根据 deployment 的凭证设置相应的 .env 变量。
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
                        "透传端点需要在 litellm_params 中设置 vertex_project 和 vertex_location。"
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
        参数：
        - deployment: Deployment - 要添加到 Router 的 deployment

        返回：
        - 添加进去的 deployment
        - 或 None（当该 deployment 已经存在时）
        """
        # 检查该 deployment 是否已存在

        _deployment_model_id = deployment.model_info.id
        if _deployment_model_id and self.has_model_id(_deployment_model_id):
            return None

        # 加入 model_list
        _deployment = deployment.to_json(exclude_none=True)
        # 初始化 client
        self._add_deployment(deployment=deployment)

        # 将 custom pricing 注册到 litellm.model_cost 中。
        # 与 _create_deployment() 的逻辑保持一致，确保动态添加的 deployment
        # （例如从数据库加载的）也能正确注册自定义定价。
        # 否则 _is_model_cost_zero() 无法识别显式配置为 0 的模型，
        # 会导致预算检查错误地拦截免费模型。
        _model_id = deployment.model_info.id
        if _model_id is not None:
            _model_info_dict: dict = deployment.model_info.model_dump(exclude_none=True)
            for field in CustomPricingLiteLLMParams.model_fields.keys():
                field_value = deployment.litellm_params.get(field)
                if field_value is not None:
                    _model_info_dict[field] = field_value
            litellm.register_model(model_cost={_model_id: _model_info_dict})

        # 加入 model_names
        self._add_model_to_list_and_index_map(
            model=_deployment, model_id=deployment.model_info.id
        )
        self.model_names.add(deployment.model_name)
        return deployment

    def _update_deployment_indices_after_removal(
        self, model_id: str, removal_idx: int
    ) -> None:
        """
        辅助方法：当某个 deployment 从 model_list 中移除后，更新相关的 deployment 索引。

        参数：
        - model_id: str - 被移除 deployment 的 id
        - removal_idx: int - 被移除 deployment 在 model_list 中的原索引位置
        """
        # 更新所有位于被删除元素之后的模型索引
        for deployment_id, idx in self.model_id_to_deployment_index_map.items():
            if idx > removal_idx:
                self.model_id_to_deployment_index_map[deployment_id] = idx - 1
        # 将被删除模型从索引中移除
        if model_id in self.model_id_to_deployment_index_map:
            del self.model_id_to_deployment_index_map[model_id]

        # 更新 model_name_to_deployment_indices
        for model_name, indices in list(self.model_name_to_deployment_indices.items()):
            # 构建新列表，不直接修改原列表
            updated_indices = []
            for idx in indices:
                if idx == removal_idx:
                    # 跳过被删除的索引
                    continue
                elif idx > removal_idx:
                    # 对于位于被删除元素之后的索引递减
                    updated_indices.append(idx - 1)
                else:
                    # 对于位于被删除元素之前的索引，保持不变
                    updated_indices.append(idx)

            # 更新或删除该条目
            if len(updated_indices) > 0:
                self.model_name_to_deployment_indices[model_name] = updated_indices
            else:
                del self.model_name_to_deployment_indices[model_name]

        # 更新 team_model_to_deployment_indices
        for key, indices in list(self.team_model_to_deployment_indices.items()):
            # 构建新列表，不直接修改原列表
            updated_indices = []
            for idx in indices:
                if idx == removal_idx:
                    # 跳过被删除的索引
                    continue
                elif idx > removal_idx:
                    # 对于位于被删除元素之后的索引递减
                    updated_indices.append(idx - 1)
                else:
                    # 对于位于被删除元素之前的索引，保持不变
                    updated_indices.append(idx)

            # 更新或删除该条目
            if len(updated_indices) > 0:
                self.team_model_to_deployment_indices[key] = updated_indices
            else:
                del self.team_model_to_deployment_indices[key]

    def _update_team_model_index(self, model: dict, idx: int) -> None:
        """
        辅助方法：为单个 deployment 更新 team_model_to_deployment_indices。

        参数：
        - model: dict - 要索引的 deployment
        - idx: int - 在 model_list 中的索引位置
        """
        team_id = (model.get("model_info") or {}).get("team_id")
        team_public_model_name = (model.get("model_info") or {}).get(
            "team_public_model_name"
        )
        if team_id and team_public_model_name:
            key = (team_id, team_public_model_name)
            if key not in self.team_model_to_deployment_indices:
                self.team_model_to_deployment_indices[key] = []
            if idx not in self.team_model_to_deployment_indices[key]:
                self.team_model_to_deployment_indices[key].append(idx)

    def _add_model_to_list_and_index_map(
        self, model: dict, model_id: Optional[str] = None
    ) -> None:
        """
        辅助方法：将一个 model 添加到 model_list 并同时更新两个索引。

        参数：
        - model: dict - 要加入列表的 model
        - model_id: Optional[str] - 用于索引的 model ID。如果为 None，尝试从 model["model_info"]["id"] 中获取
        """
        idx = len(self.model_list)
        self.model_list.append(model)
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()

        # 更新 model_id 索引，以实现 O(1) 查找
        if model_id is not None:
            self.model_id_to_deployment_index_map[model_id] = idx
        elif model.get("model_info", {}).get("id") is not None:
            self.model_id_to_deployment_index_map[model["model_info"]["id"]] = idx

        # 更新 model_name 索引，以实现 O(1) 查找
        model_name = model.get("model_name")
        if model_name:
            if model_name not in self.model_name_to_deployment_indices:
                self.model_name_to_deployment_indices[model_name] = []
            self.model_name_to_deployment_indices[model_name].append(idx)

        # 更新 team_model 索引，以实现按 team 范围的 O(1) 查找
        self._update_team_model_index(model, idx)

    def upsert_deployment(self, deployment: Deployment) -> Optional[Deployment]:
        """
        添加或更新 deployment
        参数：
        - deployment: Deployment - 要添加/更新的 deployment

        返回：
        - 添加/更新后的 deployment
        """
        try:
            # 检查该 deployment 是否已存在
            _deployment_model_id = deployment.model_info.id or ""

            _deployment_on_router: Optional[Deployment] = self.get_deployment(
                model_id=_deployment_model_id
            )
            if _deployment_on_router is not None:
                # router 中已存在使用相同 model_id 的 deployment
                if (
                    deployment.litellm_params == _deployment_on_router.litellm_params
                    and deployment.model_info == _deployment_on_router.model_info
                ):
                    # 配置完全相同，无需更新
                    return None

                # 存在新的 litellm 参数 -> 需要更新该 deployment
                # 先移除旧的 deployment
                removal_idx: Optional[int] = None
                deployment_id = deployment.model_info.id
                deployment_fast_mapping = self.model_id_to_deployment_index_map

                if deployment_id in deployment_fast_mapping:
                    removal_idx = deployment_fast_mapping[deployment_id]

                    if removal_idx is not None:
                        self.model_list.pop(removal_idx)
                        self._invalidate_model_group_info_cache()
                        self._invalidate_access_groups_cache()
                        self._update_deployment_indices_after_removal(
                            model_id=deployment_id, removal_idx=removal_idx
                        )

            # 如果 router 中不存在该 model_id，直接添加
            self.add_deployment(deployment=deployment)
            return deployment
        except Exception as e:
            if self.ignore_invalid_deployments:
                verbose_router_logger.debug(
                    f"upsert deployment 失败：{e}，忽略该 deployment 并继续处理其余。"
                )
                return None
            else:
                raise e

    def delete_deployment(self, id: str) -> Optional[Deployment]:
        """
        参数：
        - id: str - 要删除的 deployment 的 id

        返回：
        - 被删除的 deployment
        - 或 None（当未找到要删除的 deployment 时）
        """
        deployment_idx = None
        if id in self.model_id_to_deployment_index_map:
            deployment_idx = self.model_id_to_deployment_index_map[id]

        try:
            if deployment_idx is not None:
                # 先从列表中弹出该条目
                item = self.model_list.pop(deployment_idx)
                self._invalidate_model_group_info_cache()
                self._invalidate_access_groups_cache()
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
        返回 -> Deployment 或 None

        抛出异常 -> 如果查找到的 model 格式不合法
        """
        # 仅通过 model_id_to_deployment_index_map 实现 O(1) 查找
        if model_id in self.model_id_to_deployment_index_map:
            idx = self.model_id_to_deployment_index_map[model_id]
            model = self.model_list[idx]
            if isinstance(model, dict):
                return Deployment(**model)
            elif isinstance(model, Deployment):
                return model
            else:
                raise Exception("Model 格式不合法 - {}".format(type(model)))

        return None

    def get_deployment_credentials(self, model_id: str) -> Optional[dict]:
        """
        返回 -> 指定 model id 对应的凭证字典
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
        返回 -> Deployment 或 None

        抛出异常 -> 如果查找到的 model 格式不合法

        使用索引实现 O(1) 查找，避免 O(n) 的线性扫描。
        """
        # model_name 索引上的 O(1) 查找
        if model_group_name in self.model_name_to_deployment_indices:
            indices = self.model_name_to_deployment_indices[model_group_name]
            if indices:
                # 返回该 model_name 下的第一个 deployment
                model = self.model_list[indices[0]]
                if isinstance(model, dict):
                    return Deployment(**model)
                elif isinstance(model, Deployment):
                    return model
                else:
                    raise Exception("Model Name 格式不合法 - {}".format(type(model)))
        return None

    def get_deployment_credentials_with_provider(
        self, model_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        根据 model_list 中的 model 名称，获取 API 凭证和 provider 信息。
        对于需要凭证的透传端点（files、batches 等）很有用。

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

        # If still not found, check for wildcard pattern matches
        if deployment is None:
            potential_wildcard_models = self.pattern_router.route(model_id) or []
            if potential_wildcard_models:
                # Use the first matching wildcard deployment
                deployment_dict = potential_wildcard_models[0]
                if isinstance(deployment_dict, dict):
                    deployment = Deployment(**deployment_dict)
                elif isinstance(deployment_dict, Deployment):
                    deployment = deployment_dict

        if deployment is None:
            return None

        # Get basic credentials
        credentials = CredentialLiteLLMParams(
            **deployment.litellm_params.model_dump(exclude_none=True)
        ).model_dump(exclude_none=True)

        # Resolve litellm_credential_name to actual credentials
        if deployment.litellm_params.litellm_credential_name is not None:
            credential_values = CredentialAccessor.get_credential_values(
                deployment.litellm_params.litellm_credential_name
            )
            if not credential_values:
                verbose_router_logger.warning(
                    f"Credential '{deployment.litellm_params.litellm_credential_name}' not found in credential_list"
                )
            credentials.update(credential_values)
            # Remove the credential name since we've resolved it
            credentials.pop("litellm_credential_name", None)

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
        self,
        deployment: Union[dict, "Deployment"],
        received_model_name: str,
        id: None = None,
    ) -> ModelMapInfo:
        pass

    @overload
    def get_router_model_info(
        self, deployment: None, received_model_name: str, id: str
    ) -> ModelMapInfo:
        pass

    def get_router_model_info(
        self,
        deployment: Optional[Union[dict, "Deployment"]],
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
                deployment = _deployment

        if deployment is None:
            raise ValueError("Deployment not found")

        ## GET BASE MODEL
        base_model = (deployment.get("model_info") or {}).get("base_model", None)
        if base_model is None:
            base_model = (deployment.get("litellm_params") or {}).get(
                "base_model", None
            )

        model = base_model

        ## GET PROVIDER - reuse LiteLLM_Params if already constructed
        litellm_params_data = deployment.get("litellm_params")
        litellm_params: LiteLLM_Params
        if isinstance(litellm_params_data, LiteLLM_Params):
            litellm_params = litellm_params_data
        elif isinstance(litellm_params_data, dict) and "model" in litellm_params_data:
            litellm_params = LiteLLM_Params(**litellm_params_data)
        else:
            raise ValueError(
                f"Deployment missing valid litellm_params. "
                f"Got: {type(litellm_params_data).__name__}, "
                f"deployment_id: {(deployment.get('model_info') or {}).get('id', 'unknown')}"
            )
        _model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=litellm_params.model,
            litellm_params=litellm_params,
        )

        ## SET MODEL TO 'model=' - if base_model is None + not azure
        if custom_llm_provider == "azure" and base_model is None:
            verbose_router_logger.error(
                f"Could not identify azure model '{_model}'. Set azure 'base_model' for accurate max tokens, cost tracking, etc.- https://docs.litellm.ai/docs/proxy/cost_tracking#spend-tracking-for-azure-openai-models"
            )
        elif custom_llm_provider != "azure":
            model = _model

            if "*" in model:  # only call pattern_router for wildcard models
                potential_models = self.pattern_router.route(received_model_name)
                if potential_models is not None:
                    for potential_model in potential_models:
                        try:
                            if (potential_model.get("model_info") or {}).get("id") == (
                                deployment.get("model_info") or {}
                            ).get("id"):
                                model = (
                                    potential_model.get("litellm_params") or {}
                                ).get("model")
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
        user_model_info = deployment.get("model_info") or {}

        if model_info is not None:
            model_info.update(cast(ModelInfo, user_model_info))

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
                    input_cost_per_token=None,
                    output_cost_per_token=None,
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
                    or (model_info["input_cost_per_token"] or 0.0)
                    > (model_group_info.input_cost_per_token or 0.0)
                ):
                    model_group_info.input_cost_per_token = model_info[
                        "input_cost_per_token"
                    ]
                if model_info.get("output_cost_per_token", None) is not None and (
                    model_group_info.output_cost_per_token is None
                    or (model_info["output_cost_per_token"] or 0.0)
                    > (model_group_info.output_cost_per_token or 0.0)
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
        self.team_model_to_deployment_indices.clear()

        for idx, model in enumerate(model_list):
            model_name = model.get("model_name")
            if model_name:
                if model_name not in self.model_name_to_deployment_indices:
                    self.model_name_to_deployment_indices[model_name] = []
                self.model_name_to_deployment_indices[model_name].append(idx)

            self._update_team_model_index(model, idx)

    def _build_model_id_to_deployment_index_map(self, model_list: list):
        """
        Build model index from model list to enable O(1) lookups immediately.
        This is called during initialization to avoid the race condition where
        requests arrive before model_id_to_deployment_index_map is populated.
        """
        # First populate the model_list
        self.model_list = []
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()
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

    def resolve_model_name_from_model_id(
        self, model_id: Optional[str]
    ) -> Optional[str]:
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
        Check if team_model_name resolves to team-specific deployments.

        Returns the public model name (unchanged) so the router can find all
        sibling deployments via team_id filtering, instead of collapsing to a
        single internal model_name.

        Returns:
        - str: the team_model_name if team deployments exist for this team
        - None: if no team-specific model is found
        """
        models = self.get_model_list(model_name=team_model_name, team_id=team_id)
        if not models:
            return None
        for model in models:
            if model.get("model_info", {}).get("team_id") == team_id:
                return team_model_name

        # No team-scoped deployment found; wildcard/pattern routes are
        # handled downstream by the pattern_router in _common_checks_available_deployment.
        return None

    def should_include_deployment(
        self, model_name: str, model: dict, team_id: Optional[str] = None
    ) -> bool:
        """
        Get the team-specific model name if team_id matches the deployment.
        """
        if (
            team_id is not None
            and (model.get("model_info") or {}).get("team_id") == team_id
            and model_name
            == (model.get("model_info") or {}).get("team_public_model_name")
        ):
            return True
        elif model_name is not None and model["model_name"] == model_name:
            # Fallback: check by internal model_name for non-team deployments
            # or deployments that haven't been migrated to team_public_model_name yet
            model_team_id = (model.get("model_info") or {}).get("team_id")
            if (
                team_id is None  # requester has no team constraint
                or model_team_id is None  # global deployment - accessible to all teams
                or model_team_id == team_id  # deployment belongs to requester's team
            ):
                return True
        # No match: deployment is for a different team or doesn't match the requested model
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

        Note: when team_id is provided, O(1) lookup in
        `team_model_to_deployment_indices` only applies when `model_name` is the
        team public model name. If a caller passes an internal deployment model
        name (for example, `model_name_<team_id>_<uuid>`), this method falls back
        to the standard model-name index / scan path.
        """
        returned_models: List[DeploymentTypedDict] = []

        # O(1) lookup in team_model index when team_id is provided
        if team_id is not None:
            key = (team_id, model_name)
            if key in self.team_model_to_deployment_indices:
                indices = self.team_model_to_deployment_indices[key]
                # O(k) where k = team deployments for this model_name (typically 1-10)
                for idx in indices:
                    model = self.model_list[idx]
                    if not self.should_include_deployment(
                        model_name=model_name, model=model, team_id=team_id
                    ):
                        continue
                    if model_alias is not None:
                        alias_model = model.copy()
                        alias_model["model_name"] = model_alias
                        returned_models.append(alias_model)
                    else:
                        returned_models.append(model)
                if returned_models:
                    return returned_models

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

        if model_name is not None:
            # Fast path: direct dict lookup avoids scanning all aliases for non-alias model names.
            if model_name not in self.model_group_alias:
                return returned_models
            alias_items = [(model_name, self.model_group_alias[model_name])]
        else:
            alias_items = list(self.model_group_alias.items())

        for model_alias, model_value in alias_items:
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

    def _invalidate_model_group_info_cache(self) -> None:
        """Invalidate the cached model group info.

        Call this whenever self.model_list is modified to ensure the cache is rebuilt.
        """
        self._cached_get_model_group_info.cache_clear()

    def _invalidate_access_groups_cache(self) -> None:
        """Invalidate the cached access groups.

        Call this whenever self.model_list is modified to ensure the cache is rebuilt.
        """
        self._access_groups_cache = None

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
        # Check if this is the no-args hot path (cacheable)
        _use_cache = (
            model_name is None and model_access_group is None and team_id is None
        )

        # Return cached result for the no-args hot path
        if _use_cache and self._access_groups_cache is not None:
            return self._access_groups_cache

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

        # Cache the result for the no-args hot path
        if _use_cache:
            self._access_groups_cache = dict(access_groups)
            return self._access_groups_cache

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
            _deployment_model = None  # per-deployment model name (avoids overwriting the outer `model` group name)
            try:
                base_model = _model_info.get("base_model", None)
                if base_model is None:
                    base_model = _litellm_params.get("base_model", None)
                model_info = self.get_router_model_info(
                    deployment=deployment, received_model_name=model
                )
                _deployment_model = base_model or _litellm_params.get("model", None)

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
                                _deployment_model,
                                model_info["max_input_tokens"],
                                input_tokens,
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
                # get supported params — use per-deployment model to avoid overwriting the outer model group name
                _dep_model_for_params = _deployment_model or model
                (
                    _dep_model_for_params,
                    custom_llm_provider,
                    _,
                    _,
                ) = litellm.get_llm_provider(
                    model=_dep_model_for_params,
                    litellm_params=LiteLLM_Params(**_litellm_params),
                )

                supported_openai_params = litellm.get_supported_openai_params(
                    model=_dep_model_for_params,
                    custom_llm_provider=custom_llm_provider,
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
            # Check for team-specific deployments by team_public_model_name.
            # This intentionally takes priority over team pattern routers below,
            # so that named team deployments shadow wildcard/pattern routes.
            if request_team_id is not None:
                team_deployments = self._get_all_deployments(
                    model_name=model, team_id=request_team_id
                )
                if team_deployments:
                    return model, team_deployments

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
        healthy_deployments = self._get_all_deployments(
            model_name=model, team_id=request_team_id
        )

        if len(healthy_deployments) == 0:
            # check if the user sent in a deployment name instead
            healthy_deployments = self._get_deployment_by_litellm_model(model=model)

        if verbose_router_logger.isEnabledFor(logging.DEBUG):
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
                    healthy_deployments = self._get_all_deployments(
                        model_name=model, team_id=request_team_id
                    )

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

        if verbose_router_logger.isEnabledFor(logging.DEBUG):
            verbose_router_logger.debug(
                f"healthy_deployments after team filter: {healthy_deployments}"
            )

        healthy_deployments = filter_web_search_deployments(
            healthy_deployments=healthy_deployments,
            request_kwargs=request_kwargs,
        )

        if verbose_router_logger.isEnabledFor(logging.DEBUG):
            verbose_router_logger.debug(
                f"healthy_deployments after web search filter: {healthy_deployments}"
            )

        if isinstance(healthy_deployments, dict):
            return healthy_deployments

        # Health-check-based filtering (before cooldown)
        healthy_deployments = (
            await self._async_filter_health_check_unhealthy_deployments(
                healthy_deployments=healthy_deployments,
                parent_otel_span=parent_otel_span,
            )
        )

        cooldown_deployments = await _async_get_cooldown_deployments(
            litellm_router_instance=self, parent_otel_span=parent_otel_span
        )
        if verbose_router_logger.isEnabledFor(logging.DEBUG):
            verbose_router_logger.debug(f"cooldown deployments: {cooldown_deployments}")
        _pre_cooldown_deployments = healthy_deployments
        healthy_deployments = self._filter_cooldown_deployments(
            healthy_deployments=healthy_deployments,
            cooldown_deployments=cooldown_deployments,
        )
        # Safety net: only bypass cooldown filter when health-check routing is
        # driving cooldown (i.e. allowed_fails_policy is set). Without a policy,
        # cooldowns are from real request failures and must not be bypassed.
        if (
            not healthy_deployments
            and self.enable_health_check_routing
            and self.allowed_fails_policy is not None
        ):
            verbose_router_logger.warning(
                "All deployments in cooldown via health-check routing, bypassing cooldown filter"
            )
            healthy_deployments = _pre_cooldown_deployments

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

        ## ORDER FILTERING ## -> if user set 'order' in deployments, return deployments with lowest order (e.g. order=1 > order=2)
        _target_order = (request_kwargs or {}).pop("_target_order", None)
        healthy_deployments = litellm.utils._get_order_filtered_deployments(
            cast(List[Dict], healthy_deployments), target_order=_target_order
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

            # When encrypted content affinity pins to a specific deployment,
            if (
                request_kwargs.get("_encrypted_content_affinity_pinned")
                and len(healthy_deployments) == 1
            ):
                return healthy_deployments[0]

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

    async def async_get_available_deployment_for_pass_through(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ):
        """
        Async version of get_available_deployment_for_pass_through

        Only returns deployments configured with use_in_pass_through=True
        """
        try:
            parent_otel_span = _get_parent_otel_span_from_kwargs(request_kwargs)

            # 1. Execute pre-routing hook
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

            # 2. Get healthy deployments
            healthy_deployments = await self.async_get_healthy_deployments(
                model=model,
                request_kwargs=request_kwargs,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
                parent_otel_span=parent_otel_span,
            )

            # 3. If specific deployment returned, verify if it supports pass-through
            if isinstance(healthy_deployments, dict):
                litellm_params = healthy_deployments.get("litellm_params", {})
                if litellm_params.get("use_in_pass_through"):
                    return healthy_deployments
                else:
                    raise litellm.BadRequestError(
                        message=f"Deployment {healthy_deployments.get('model_info', {}).get('id')} does not support pass-through endpoint (use_in_pass_through=False)",
                        model=model,
                        llm_provider="",
                    )

            # 4. Filter deployments that support pass-through
            pass_through_deployments = self._filter_pass_through_deployments(
                healthy_deployments=healthy_deployments
            )

            if len(pass_through_deployments) == 0:
                raise litellm.BadRequestError(
                    message=f"Model {model} has no deployments configured with use_in_pass_through=True. Please add use_in_pass_through: true to the deployment configuration",
                    model=model,
                    llm_provider="",
                )

            # 5. Apply load balancing strategy
            start_time = time.perf_counter()
            if (
                self.routing_strategy == "usage-based-routing-v2"
                and self.lowesttpm_logger_v2 is not None
            ):
                deployment = (
                    await self.lowesttpm_logger_v2.async_get_available_deployments(
                        model_group=model,
                        healthy_deployments=pass_through_deployments,  # type: ignore
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
                        healthy_deployments=pass_through_deployments,  # type: ignore
                        messages=messages,
                        input=input,
                        request_kwargs=request_kwargs,
                    )
                )
            elif self.routing_strategy == "simple-shuffle":
                return simple_shuffle(
                    llm_router_instance=self,
                    healthy_deployments=pass_through_deployments,
                    model=model,
                )
            elif (
                self.routing_strategy == "least-busy"
                and self.leastbusy_logger is not None
            ):
                deployment = (
                    await self.leastbusy_logger.async_get_available_deployments(
                        model_group=model,
                        healthy_deployments=pass_through_deployments,  # type: ignore
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
                f"async_get_available_deployment_for_pass_through model: {model}, selected deployment: {self.print_deployment(deployment)}"
            )

            end_time = time.perf_counter()
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
            if request_kwargs is not None:
                logging_obj = request_kwargs.get("litellm_logging_obj", None)
                if logging_obj is not None:
                    threading.Thread(
                        target=logging_obj.failure_handler,
                        args=(e, traceback_exception),
                    ).start()
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

        #########################################################
        # Check if any complexity-router should be used
        #########################################################
        if model in self.complexity_routers:
            return await self.complexity_routers[model].async_pre_routing_hook(
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

        # Health-check-based filtering (before cooldown)
        healthy_deployments = self._filter_health_check_unhealthy_deployments(
            healthy_deployments=healthy_deployments,
            parent_otel_span=parent_otel_span,
        )

        cooldown_deployments = _get_cooldown_deployments(
            litellm_router_instance=self, parent_otel_span=parent_otel_span
        )
        _pre_cooldown_deployments = healthy_deployments
        healthy_deployments = self._filter_cooldown_deployments(
            healthy_deployments=healthy_deployments,
            cooldown_deployments=cooldown_deployments,
        )
        if (
            not healthy_deployments
            and self.enable_health_check_routing
            and self.allowed_fails_policy is not None
        ):
            verbose_router_logger.warning(
                "All deployments in cooldown via health-check routing, bypassing cooldown filter"
            )
            healthy_deployments = _pre_cooldown_deployments

        # filter pre-call checks
        if self.enable_pre_call_checks and messages is not None:
            healthy_deployments = self._pre_call_checks(
                model=model,
                healthy_deployments=healthy_deployments,
                messages=messages,
                request_kwargs=request_kwargs,
            )

        ## ORDER FILTERING ## -> if user set 'order' in deployments, return deployments with lowest order (e.g. order=1 > order=2)
        _target_order = (request_kwargs or {}).pop("_target_order", None)
        healthy_deployments = litellm.utils._get_order_filtered_deployments(
            healthy_deployments, target_order=_target_order
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

    def get_available_deployment_for_pass_through(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Returns deployments available for pass-through endpoints (based on load balancing strategy)

        Similar to get_available_deployment, but only returns deployments with use_in_pass_through=True

        Args:
            model: Model name
            messages: Optional list of messages
            input: Optional input data
            specific_deployment: Whether to find a specific deployment
            request_kwargs: Optional request parameters

        Returns:
            Dict: Selected deployment configuration

        Raises:
            BadRequestError: If no deployment is configured with use_in_pass_through=True
            RouterRateLimitError: If no pass-through deployments are available
        """
        # 1. Perform common checks to get healthy deployments list
        model, healthy_deployments = self._common_checks_available_deployment(
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )

        # 2. If the returned is a specific deployment (Dict), verify and return directly
        if isinstance(healthy_deployments, dict):
            litellm_params = healthy_deployments.get("litellm_params", {})
            if litellm_params.get("use_in_pass_through"):
                return healthy_deployments
            else:
                # Specific deployment does not support pass-through
                raise litellm.BadRequestError(
                    message=f"Deployment {healthy_deployments.get('model_info', {}).get('id')} does not support pass-through endpoint (use_in_pass_through=False)",
                    model=model,
                    llm_provider="",
                )

        # 3. Filter deployments that support pass-through
        pass_through_deployments = self._filter_pass_through_deployments(
            healthy_deployments=healthy_deployments
        )

        if len(pass_through_deployments) == 0:
            # No deployments support pass-through
            raise litellm.BadRequestError(
                message=f"Model {model} has no deployment configured with use_in_pass_through=True. Please add use_in_pass_through: true in the deployment configuration",
                model=model,
                llm_provider="",
            )

        # 4. Apply health-check and cooldown filtering
        parent_otel_span: Optional[Span] = _get_parent_otel_span_from_kwargs(
            request_kwargs
        )
        pass_through_deployments = self._filter_health_check_unhealthy_deployments(
            healthy_deployments=pass_through_deployments,
            parent_otel_span=parent_otel_span,
        )
        cooldown_deployments = _get_cooldown_deployments(
            litellm_router_instance=self, parent_otel_span=parent_otel_span
        )
        pass_through_deployments = self._filter_cooldown_deployments(
            healthy_deployments=pass_through_deployments,
            cooldown_deployments=cooldown_deployments,
        )

        # 5. Apply pre-call checks (if enabled)
        if self.enable_pre_call_checks and messages is not None:
            pass_through_deployments = self._pre_call_checks(
                model=model,
                healthy_deployments=pass_through_deployments,
                messages=messages,
                request_kwargs=request_kwargs,
            )

        if len(pass_through_deployments) == 0:
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

        # 6. Apply load balancing strategy
        if self.routing_strategy == "least-busy" and self.leastbusy_logger is not None:
            deployment = self.leastbusy_logger.get_available_deployments(
                model_group=model, healthy_deployments=pass_through_deployments  # type: ignore
            )
        elif self.routing_strategy == "simple-shuffle":
            return simple_shuffle(
                llm_router_instance=self,
                healthy_deployments=pass_through_deployments,
                model=model,
            )
        elif (
            self.routing_strategy == "latency-based-routing"
            and self.lowestlatency_logger is not None
        ):
            deployment = self.lowestlatency_logger.get_available_deployments(
                model_group=model,
                healthy_deployments=pass_through_deployments,  # type: ignore
                request_kwargs=request_kwargs,
            )
        elif (
            self.routing_strategy == "usage-based-routing"
            and self.lowesttpm_logger is not None
        ):
            deployment = self.lowesttpm_logger.get_available_deployments(
                model_group=model,
                healthy_deployments=pass_through_deployments,  # type: ignore
                messages=messages,
                input=input,
            )
        elif (
            self.routing_strategy == "usage-based-routing-v2"
            and self.lowesttpm_logger_v2 is not None
        ):
            deployment = self.lowesttpm_logger_v2.get_available_deployments(
                model_group=model,
                healthy_deployments=pass_through_deployments,  # type: ignore
                messages=messages,
                input=input,
            )
        else:
            deployment = None

        if deployment is None:
            verbose_router_logger.info(
                f"get_available_deployment_for_pass_through model: {model}, no available deployments"
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
            f"get_available_deployment_for_pass_through model: {model}, selected deployment: {self.print_deployment(deployment)}"
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
        if verbose_router_logger.isEnabledFor(logging.DEBUG):
            verbose_router_logger.debug(f"cooldown deployments: {cooldown_deployments}")
        # Convert to set for O(1) lookup and use list comprehension for O(n) filtering
        cooldown_set = set(cooldown_deployments)
        return [
            deployment
            for deployment in healthy_deployments
            if deployment["model_info"]["id"] not in cooldown_set
        ]

    async def _async_filter_health_check_unhealthy_deployments(
        self,
        healthy_deployments: List[Dict],
        parent_otel_span: Optional[Span] = None,
    ) -> List[Dict]:
        """
        Filter out deployments marked unhealthy by background health checks.
        No-op when enable_health_check_routing is False.
        Returns all deployments if health state is unavailable, stale, or would
        exclude every candidate (safety net).
        """
        if not self.enable_health_check_routing:
            return healthy_deployments

        # When allowed_fails_policy is set, cooldown is the sole routing exclusion
        # mechanism -- skip the binary health check filter so the policy threshold
        # is respected before any deployment is excluded.
        if self.allowed_fails_policy is not None:
            return healthy_deployments

        unhealthy_ids = (
            await self.health_state_cache.async_get_unhealthy_deployment_ids(
                parent_otel_span=parent_otel_span
            )
        )
        if not unhealthy_ids:
            return healthy_deployments

        filtered = [
            d for d in healthy_deployments if d["model_info"]["id"] not in unhealthy_ids
        ]

        if not filtered:
            verbose_router_logger.warning(
                "All deployments marked unhealthy by health checks, bypassing health filter"
            )
            return healthy_deployments

        return filtered

    def _filter_health_check_unhealthy_deployments(
        self,
        healthy_deployments: List[Dict],
        parent_otel_span: Optional[Span] = None,
    ) -> List[Dict]:
        """Sync version of _async_filter_health_check_unhealthy_deployments."""
        if not self.enable_health_check_routing:
            return healthy_deployments

        if self.allowed_fails_policy is not None:
            return healthy_deployments

        unhealthy_ids = self.health_state_cache.get_unhealthy_deployment_ids(
            parent_otel_span=parent_otel_span
        )
        if not unhealthy_ids:
            return healthy_deployments

        filtered = [
            d for d in healthy_deployments if d["model_info"]["id"] not in unhealthy_ids
        ]

        if not filtered:
            verbose_router_logger.warning(
                "All deployments marked unhealthy by health checks, bypassing health filter"
            )
            return healthy_deployments

        return filtered

    def _filter_pass_through_deployments(
        self, healthy_deployments: List[Dict]
    ) -> List[Dict]:
        """
        Filter out deployments configured with use_in_pass_through=True

        Args:
            healthy_deployments: List of healthy deployments

        Returns:
            List[Dict]: Only includes a list of deployments that support pass-through
        """
        verbose_router_logger.debug(
            f"Filter pass-through deployments from {len(healthy_deployments)} healthy deployments"
        )

        pass_through_deployments = [
            deployment
            for deployment in healthy_deployments
            if deployment.get("litellm_params", {}).get("use_in_pass_through", False)
        ]

        verbose_router_logger.debug(
            f"Found {len(pass_through_deployments)} deployments with pass-through enabled"
        )

        return pass_through_deployments

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
        if (
            isinstance(exception, litellm.BadRequestError)
            and allowed_fails_policy.BadRequestErrorAllowedFails is not None
        ):
            return allowed_fails_policy.BadRequestErrorAllowedFails

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

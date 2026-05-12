"""
Router 冷却（cooldown）处理模块
- _set_cooldown_deployments: 将某个部署（deployment）加入冷却列表
- get_cooldown_deployments: 返回当前处于冷却状态的部署列表
- async_get_cooldown_deployments: 异步版本——返回当前处于冷却状态的部署列表

"""

import asyncio
import math
from typing import TYPE_CHECKING, Any, List, Optional, Union

import litellm
from litellm._logging import verbose_router_logger
from litellm.constants import (
    DEFAULT_COOLDOWN_TIME_SECONDS,
    DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS,
    DEFAULT_FAILURE_THRESHOLD_PERCENT,
    SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD,
)
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback
from .router_callbacks.track_deployment_metrics import (
    get_deployment_failures_for_current_minute,
    get_deployment_successes_for_current_minute,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.router import Router as _Router

    LitellmRouter = _Router
    Span = Union[_Span, Any]
else:
    LitellmRouter = Any
    Span = Any


def _is_cooldown_required(
    litellm_router_instance: LitellmRouter,
    model_id: str,
    exception_status: Union[str, int],
    exception_str: Optional[str] = None,
) -> bool:
    """
    根据异常状态码判断是否需要进入冷却。

    参数：
        model_id (str)：模型列表中对应模型的 id
        exception_status (Union[str, int])：异常对应的状态码

    返回：
        bool：需要冷却则返回 True，否则返回 False。
    """
    try:
        # 以下字符串出现在异常信息中时，不触发冷却（属于 litellm 自身的连接错误）
        ignored_strings = ["APIConnectionError"]
        if (
            exception_str is not None
        ):  # 对于 litellm 自身的 API 连接错误，不做冷却处理
            for ignored_string in ignored_strings:
                if ignored_string in exception_str:
                    return False

        # 如果状态码是字符串类型，转成 int；空字符串视为不需要冷却
        if isinstance(exception_status, str):
            if len(exception_status) == 0:
                return False
            exception_status = int(exception_status)

        if exception_status >= 400 and exception_status < 500:
            if exception_status == 429:
                # 429 限流错误 —— 需要冷却
                return True

            elif exception_status == 401:
                # 401 鉴权错误 —— 需要冷却
                return True

            elif exception_status == 408:
                # 408 请求超时 —— 需要冷却
                return True

            elif exception_status == 404:
                # 404 未找到 —— 需要冷却
                return True

            else:
                # 其他 4XX 错误一律不冷却
                return False

        else:
            # 5XX 或其他状态码 —— 都做冷却处理
            return True

    except Exception:
        # 兜底逻辑 —— 解析过程中出现任何异常，默认进行冷却
        return True


def _should_run_cooldown_logic(
    litellm_router_instance: LitellmRouter,
    deployment: Optional[str],
    exception_status: Union[str, int],
    original_exception: Any,
    time_to_cooldown: Optional[float] = None,
) -> bool:
    """
    判断是否需要执行冷却逻辑的辅助函数。
    返回 False 表示不应执行冷却逻辑。

    以下场景不会执行冷却逻辑：
    - router.disable_cooldowns 为 True（即全局关闭了冷却功能）
    - deployment 为 None
    - _is_cooldown_required() 返回 False
    - deployment 存在于 litellm_router_instance.provider_default_deployment_ids 中
    - 异常状态码属于不应立即重试的类型（例如 401）
    """
    if (
        deployment is None
        or litellm_router_instance.get_model_group(id=deployment) is None
    ):
        verbose_router_logger.debug(
            "不执行冷却逻辑：deployment id 为 None，或找不到对应的 model group。"
        )
        return False

    #########################################################
    # 如果 time_to_cooldown 为 0 或接近 0，不执行冷却逻辑
    #########################################################
    if time_to_cooldown is not None and math.isclose(
        a=time_to_cooldown, b=0.0, abs_tol=1e-9
    ):
        verbose_router_logger.debug(
            "不执行冷却逻辑：time_to_cooldown 实质上为 0"
        )
        return False

    if litellm_router_instance.disable_cooldowns:
        verbose_router_logger.debug(
            "不执行冷却逻辑：disable_cooldowns 为 True"
        )
        return False

    if deployment is None:
        verbose_router_logger.debug("不执行冷却逻辑：deployment 为 None")
        return False

    if not _is_cooldown_required(
        litellm_router_instance=litellm_router_instance,
        model_id=deployment,
        exception_status=exception_status,
        exception_str=str(original_exception),
    ):
        verbose_router_logger.debug(
            "不执行冷却逻辑：_is_cooldown_required 返回 False"
        )
        return False

    if deployment in litellm_router_instance.provider_default_deployment_ids:
        verbose_router_logger.debug(
            "不执行冷却逻辑：deployment 属于 provider_default_deployment_ids（厂商默认部署）"
        )
        return False

    return True


def _should_cooldown_deployment(
    litellm_router_instance: LitellmRouter,
    deployment: str,
    exception_status: Union[str, int],
    original_exception: Any,
) -> bool:
    """
    判断是否应该把某个 deployment 加入冷却列表的辅助函数。

    返回 True 表示应当冷却；返回 False 表示不冷却。

    deployment 在以下情况下会被冷却：
    - v2 逻辑（当前使用）：
        - 收到 LLM API 返回的 429 错误
        - 当 %失败数/%(成功数 + 失败数) 超过 ALLOWED_FAILURE_RATE_PER_MINUTE 阈值
        - 收到 401 鉴权错误 / 404 未找到错误（由 litellm._should_retry() 判定）

    - v1 逻辑（遗留）：如果设置了 allowed_fails 或 allowed_fails_policy，
      当本分钟内失败次数超过允许值时进行冷却
    """
    ## 特殊情况 —— model group 中只有一个 deployment
    model_group = litellm_router_instance.get_model_group(id=deployment)
    is_single_deployment_model_group = False
    if model_group is not None and len(model_group) == 1:
        is_single_deployment_model_group = True
    if (
        litellm_router_instance.allowed_fails_policy is None
        and _is_allowed_fails_set_on_router(
            litellm_router_instance=litellm_router_instance
        )
        is False
    ):
        # 获取本分钟内该 deployment 的成功次数和失败次数
        num_successes_this_minute = get_deployment_successes_for_current_minute(
            litellm_router_instance=litellm_router_instance, deployment_id=deployment
        )
        num_fails_this_minute = get_deployment_failures_for_current_minute(
            litellm_router_instance=litellm_router_instance, deployment_id=deployment
        )

        total_requests_this_minute = num_successes_this_minute + num_fails_this_minute
        percent_fails = 0.0
        if total_requests_this_minute > 0:
            percent_fails = num_fails_this_minute / (
                num_successes_this_minute + num_fails_this_minute
            )
        verbose_router_logger.debug(
            "deployment = %s 的失败率统计 —— 失败占比 = %s，成功次数 = %s，失败次数 = %s",
            deployment,
            percent_fails,
            num_successes_this_minute,
            num_fails_this_minute,
        )

        exception_status_int = cast_exception_status_to_int(exception_status)
        if exception_status_int == 429 and not is_single_deployment_model_group:
            # 429 限流：只要不是单部署的 model group，就进入冷却
            return True
        elif (
            percent_fails == 1.0
            and total_requests_this_minute
            >= SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD
        ):
            # 请求全部失败且流量达到一定阈值时，冷却该 deployment
            return True
        elif (
            percent_fails > DEFAULT_FAILURE_THRESHOLD_PERCENT
            and total_requests_this_minute >= DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS
            and not is_single_deployment_model_group  # 单部署 model group 默认不做失败率冷却
        ):
            # 只有请求数足够多时才基于失败率冷却（避免小样本噪声）
            return True

        elif (
            litellm._should_retry(
                status_code=cast_exception_status_to_int(exception_status)
            )
            is False
        ):
            # 状态码属于不应重试的类型（如 401），直接冷却
            return True

        return False
    else:
        # 走遗留的 allowed_fails_policy 分支
        return should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=litellm_router_instance,
            deployment=deployment,
            original_exception=original_exception,
        )

    return False


def _set_cooldown_deployments(
    litellm_router_instance: LitellmRouter,
    original_exception: Any,
    exception_status: Union[str, int],
    deployment: Optional[str] = None,
    time_to_cooldown: Optional[float] = None,
) -> bool:
    """
    如果某个模型本分钟内的失败数超过允许值，
    或者异常属于不应立即重试的类型（例如 401），
    则将其加入本分钟的冷却列表。

    返回值：
    - True：该 deployment 已被加入冷却
    - False：该 deployment 未被加入冷却
    """
    verbose_router_logger.debug("开始检查 'should_run_cooldown_logic'")

    if (
        _should_run_cooldown_logic(
            litellm_router_instance=litellm_router_instance,
            deployment=deployment,
            exception_status=exception_status,
            original_exception=original_exception,
            time_to_cooldown=time_to_cooldown,
        )
        is False
        or deployment is None
    ):
        verbose_router_logger.debug("should_run_cooldown_logic 返回 False")
        return False

    exception_status_int = cast_exception_status_to_int(exception_status)
    verbose_router_logger.debug(f"尝试将 {deployment} 加入冷却列表")

    if _should_cooldown_deployment(
        litellm_router_instance=litellm_router_instance,
        deployment=deployment,
        exception_status=exception_status,
        original_exception=original_exception,
    ):
        # 写入冷却缓存
        litellm_router_instance.cooldown_cache.add_deployment_to_cooldown(
            model_id=deployment,
            original_exception=original_exception,
            exception_status=exception_status_int,
            cooldown_time=time_to_cooldown,
        )

        # 触发冷却事件回调（异步任务）
        asyncio.create_task(
            router_cooldown_event_callback(
                litellm_router_instance=litellm_router_instance,
                deployment_id=deployment,
                exception_status=exception_status,
                cooldown_time=time_to_cooldown,
            )
        )
        return True
    return False


async def _async_get_cooldown_deployments(
    litellm_router_instance: LitellmRouter,
    parent_otel_span: Optional[Span],
) -> List[str]:
    """
    '_get_cooldown_deployments' 的异步实现。
    """
    model_ids = litellm_router_instance.get_model_ids()
    cooldown_models = (
        await litellm_router_instance.cooldown_cache.async_get_active_cooldowns(
            model_ids=model_ids,
            parent_otel_span=parent_otel_span,
        )
    )

    cached_value_deployment_ids = []
    if (
        cooldown_models is not None
        and isinstance(cooldown_models, list)
        and len(cooldown_models) > 0
        and isinstance(cooldown_models[0], tuple)
    ):
        # 缓存中存储的是 (deployment_id, 其他信息) 的元组，这里只取 id
        cached_value_deployment_ids = [cv[0] for cv in cooldown_models]

    verbose_router_logger.debug(f"已获取处于冷却中的模型：{cooldown_models}")
    return cached_value_deployment_ids


async def _async_get_cooldown_deployments_with_debug_info(
    litellm_router_instance: LitellmRouter,
    parent_otel_span: Optional[Span],
) -> List[tuple]:
    """
    '_get_cooldown_deployments' 的异步实现 —— 同时返回调试信息。
    """
    model_ids = litellm_router_instance.get_model_ids()
    cooldown_models = (
        await litellm_router_instance.cooldown_cache.async_get_active_cooldowns(
            model_ids=model_ids, parent_otel_span=parent_otel_span
        )
    )

    verbose_router_logger.debug(f"已获取处于冷却中的模型：{cooldown_models}")
    return cooldown_models


def _get_cooldown_deployments(
    litellm_router_instance: LitellmRouter, parent_otel_span: Optional[Span]
) -> List[str]:
    """
    获取本分钟内处于冷却状态的模型列表。
    """
    # 获取当前分钟的冷却列表

    # ----------------------
    # 返回处于冷却中的模型 id
    # ----------------------
    model_ids = litellm_router_instance.get_model_ids()

    cooldown_models = litellm_router_instance.cooldown_cache.get_active_cooldowns(
        model_ids=model_ids, parent_otel_span=parent_otel_span
    )

    cached_value_deployment_ids = []
    if (
        cooldown_models is not None
        and isinstance(cooldown_models, list)
        and len(cooldown_models) > 0
        and isinstance(cooldown_models[0], tuple)
    ):
        cached_value_deployment_ids = [cv[0] for cv in cooldown_models]

    return cached_value_deployment_ids


def should_cooldown_based_on_allowed_fails_policy(
    litellm_router_instance: LitellmRouter,
    deployment: str,
    original_exception: Any,
) -> bool:
    """
    检查失败次数是否在允许范围内，并更新失败计数。

    返回值：
    - True：失败次数超过允许值（应当冷却）
    - False：失败次数仍在允许范围内（不应冷却）
    """
    # 优先使用策略（policy）中针对特定异常的允许值，否则使用 router 全局的 allowed_fails
    allowed_fails = (
        litellm_router_instance.get_allowed_fails_from_policy(
            exception=original_exception,
        )
        or litellm_router_instance.allowed_fails
    )
    cooldown_time = (
        litellm_router_instance.cooldown_time or DEFAULT_COOLDOWN_TIME_SECONDS
    )

    # 当前累计失败次数 +1
    current_fails = litellm_router_instance.failed_calls.get_cache(key=deployment) or 0
    updated_fails = current_fails + 1

    if updated_fails > allowed_fails:
        return True
    else:
        # 未超限，仅更新缓存（带 TTL），不做冷却
        litellm_router_instance.failed_calls.set_cache(
            key=deployment, value=updated_fails, ttl=cooldown_time
        )

    return False


def _is_allowed_fails_set_on_router(
    litellm_router_instance: LitellmRouter,
) -> bool:
    """
    检查 Router.allowed_fails 是否被用户设置过（即非默认值）。

    返回值：
    - True：Router.allowed_fails 已被设置为非默认值
    - False：Router.allowed_fails 为 None 或等于默认值
    """
    if litellm_router_instance.allowed_fails is None:
        return False
    if litellm_router_instance.allowed_fails != litellm.allowed_fails:
        return True
    return False


def cast_exception_status_to_int(exception_status: Union[str, int]) -> int:
    """将异常状态码统一转换为 int；无法转换时默认为 500。"""
    if isinstance(exception_status, str):
        try:
            exception_status = int(exception_status)
        except Exception:
            verbose_router_logger.debug(
                f"无法将异常状态码 {exception_status} 转换为 int，默认使用 status=500。"
            )
            exception_status = 500
    return exception_status

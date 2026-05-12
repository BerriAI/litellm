"""
Base class across routing strategies to abstract commmon functions like batch incrementing redis

路由策略通用基类：抽取各路由策略（按 provider 预算路由、限流路由、RPM/TPM 路由等）都会用到的公共能力。

核心职责：
1. 在 "内存缓存 + Redis" 的双层缓存架构下，统一管理 spend / RPM / TPM 等计数值的累加。
2. 提供 "批量写 Redis" 的能力：把每次请求的增量先落到内存，再通过异步定时任务合并后
   以 pipeline 的形式批量推到 Redis，避免每请求一次 Redis 往返带来的延迟（目标 < 100ms）。
3. 在多实例（多 worker / 多 Pod）场景下，通过 Redis 做跨实例的状态同步。

典型继承者：ProviderBudgetLimiting、各类 RateLimiter 等。
"""

import asyncio
from abc import ABC
from typing import Dict, List, Optional, Set, Tuple, Union

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.constants import DEFAULT_REDIS_SYNC_INTERVAL


class BaseRoutingStrategy(ABC):
    """
    所有路由策略的抽象基类。

    设计模式：**双层缓存（in-memory + redis）+ 批量异步同步**。
    - 读写热路径：只操作内存（亚毫秒级），对单请求延迟友好
    - 后台任务：周期性把内存里的增量批量 flush 到 Redis，再把 Redis 最新值拉回来合并
    - 多实例一致性：所有实例都往同一个 Redis key 累加，再各自拉回来同步，实现最终一致
    """

    def __init__(
        self,
        dual_cache: DualCache,
        should_batch_redis_writes: bool,
        default_sync_interval: Optional[Union[int, float]],
    ):
        """
        Args:
            dual_cache: LiteLLM 的双层缓存封装（同时持有 in_memory_cache 和 redis_cache）。
            should_batch_redis_writes: 是否启用"批量异步写 Redis"。
                - True：启动后台 sync 协程，定期 flush 增量到 Redis
                - False：纯本地内存模式（单实例够用，不需要跨实例同步）
            default_sync_interval: 后台同步任务的间隔秒数；None 时使用 DEFAULT_REDIS_SYNC_INTERVAL。
        """
        # 双层缓存实例：后续所有累加/查询都通过它来走
        self.dual_cache = dual_cache

        # Redis 增量操作队列：每次 in-memory 累加都会在这里追加一条对应的 Redis 累加操作，
        # 等后台任务来了再统一 flush，减少网络往返
        self.redis_increment_operation_queue: List[RedisPipelineIncrementOperation] = []

        # 后台同步任务的句柄，cleanup 时用来 cancel
        self._sync_task: Optional[asyncio.Task[None]] = None

        # 如果需要批量写 Redis，则启动后台周期性同步协程
        if should_batch_redis_writes:
            self.setup_sync_task(default_sync_interval)

        # 需要"在下一次 sync 时从 Redis 拉最新值回内存"的 key 集合
        # 注：注释里写的 max size 1000 只是预期约束，Python 的 set 本身并没有强制容量限制
        self.in_memory_keys_to_update: set[
            str
        ] = set()  # Set with max size of 1000 keys

    def setup_sync_task(self, default_sync_interval: Optional[Union[int, float]]):
        """Setup the sync task in a way that's compatible with FastAPI

        启动后台同步协程。兼容两种场景：
        - 已经有事件循环（FastAPI 启动后）：直接挂到当前 loop
        - 还没有事件循环（SDK 直接实例化时）：自己起一个 loop
        """
        try:
            # 尝试获取当前正在运行的事件循环（FastAPI 应用内的典型路径）
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有正在运行的事件循环时，新建一个并设为当前循环
            # 典型场景：脚本/SDK 环境，尚未启动 FastAPI/ASGI
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 注册后台任务：它会死循环 + sleep，直到被 cancel
        self._sync_task = loop.create_task(
            self.periodic_sync_in_memory_spend_with_redis(
                default_sync_interval=default_sync_interval
            )
        )

    async def cleanup(self):
        """Cleanup method to be called when shutting down

        优雅关闭：在应用 shutdown 时取消后台同步任务，避免孤儿协程。
        """
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                # 预期中的取消异常，吞掉即可
                pass

    async def _increment_value_list_in_current_window(
        self, increment_list: List[Tuple[str, int]], ttl: int
    ) -> List[float]:
        """
        Increment a list of values in the current window

        批量累加一组 (key, value) 到当前窗口中。
        仅是对 _increment_value_in_current_window 的顺序循环封装，方便一次请求
        里需要同时更新多个计数（例如同时记 RPM 和 TPM）。
        """
        results = []
        for key, value in increment_list:
            result = await self._increment_value_in_current_window(
                key=key, value=value, ttl=ttl
            )
            results.append(result)
        return results

    async def _increment_value_in_current_window(
        self, key: str, value: Union[int, float], ttl: int
    ):
        """
        Increment spend within existing budget window

        Runs once the budget start time exists in Redis Cache (on the 2nd and subsequent requests to the same provider)

        - Increments the spend in memory cache (so spend instantly updated in memory)
        - Queues the increment operation to Redis Pipeline (using batched pipeline to optimize performance. Using Redis for multi instance environment of LiteLLM)

        在"当前预算窗口"内累加指定 key 的值。

        使用场景：某个 provider 的预算/计数窗口已经建立（Redis 里已经有 start_time），
        后续请求走这个路径来持续累加。

        步骤：
        1. 立刻在 in-memory 缓存累加（同步返回结果 → 热路径无 Redis 延迟）
        2. 把这次累加以一条 RedisPipelineIncrementOperation 追加到队列，
           由后台协程批量 flush 到 Redis
        3. 记录该 key，稍后 sync 时从 Redis 拉最新值回内存做合并
        """
        # Step 1: 内存缓存立刻累加，返回累加后的结果给调用方使用（亚毫秒级）
        result = await self.dual_cache.in_memory_cache.async_increment(
            key=key,
            value=value,
            ttl=ttl,
        )

        # Step 2: 构造一个"延迟的 Redis 累加操作"，进队列等待批量 flush
        increment_op = RedisPipelineIncrementOperation(
            key=key,
            increment_value=value,
            ttl=ttl,
        )

        # 把操作塞进队列，不立刻发给 Redis，避免每次请求都一次网络往返
        self.redis_increment_operation_queue.append(increment_op)

        # Step 3: 标记这个 key，稍后 sync 时要从 Redis 拉最新值回内存合并
        self.add_to_in_memory_keys_to_update(key=key)
        return result

    async def periodic_sync_in_memory_spend_with_redis(
        self, default_sync_interval: Optional[Union[int, float]]
    ):
        """
        Handler that triggers sync_in_memory_spend_with_redis every DEFAULT_REDIS_SYNC_INTERVAL seconds

        Required for multi-instance environment usage of provider budgets

        后台常驻协程：
        每隔 default_sync_interval 秒触发一次 _sync_in_memory_spend_with_redis。

        用途：多实例部署下，让各实例通过 Redis 达成最终一致的 spend/RPM/TPM 视图。
        只要任何一次 sync 抛异常，都会被吞掉并继续循环，保证协程永不退出。
        """
        default_sync_interval = default_sync_interval or DEFAULT_REDIS_SYNC_INTERVAL
        while True:
            try:
                # 核心：一次完整的"写入增量 + 拉回合并"流程
                await self._sync_in_memory_spend_with_redis()
                await asyncio.sleep(
                    default_sync_interval
                )  # Wait for DEFAULT_REDIS_SYNC_INTERVAL seconds before next sync
            except Exception as e:
                # 异常场景（例如 Redis 临时抖动）下不能让后台协程挂掉，
                # 打日志 + sleep 后继续下一轮
                verbose_router_logger.error(f"Error in periodic sync task: {str(e)}")
                await asyncio.sleep(
                    default_sync_interval
                )  # Still wait DEFAULT_REDIS_SYNC_INTERVAL seconds on error before retrying

    async def _push_in_memory_increments_to_redis(self):
        """
        How this works:
        - async_log_success_event collects all provider spend increments in `redis_increment_operation_queue`
        - This function compresses multiple increments for the same key into a single operation
        - Then pushes all increments to Redis in a batched pipeline to optimize performance

        Only runs if Redis is initialized

        把 `redis_increment_operation_queue` 里累积的"待写 Redis 操作"批量推到 Redis。

        优化点：
        1. **合并同 key 操作**：若同一个 key 有多条增量，合并为一条（value 相加），减少 pipeline 命令数
        2. **Pipeline 批量写**：把 N 条 INCR 一次性打包发给 Redis，省掉 N 次 RTT
        3. **仅在 Redis 可用时运行**：纯内存模式直接跳过

        返回：
            Dict[key, redis_increment_result] —— 每个被合并写入后 Redis 侧的最新值。
            用于 `_sync_in_memory_spend_with_redis` 里和内存做对账合并。
        """
        try:
            # Redis 没初始化（纯内存模式）直接返回，什么都不做
            if not self.dual_cache.redis_cache:
                return  # Redis is not initialized

            if len(self.redis_increment_operation_queue) > 0:
                # ---- Step 1: 合并同 key 的多条增量 ----
                # 例如队列里有 [("spend:gpt-4", 0.01), ("spend:gpt-4", 0.02), ("spend:claude", 0.03)]
                # 压缩后变成 [("spend:gpt-4", 0.03), ("spend:claude", 0.03)]
                compressed_ops: Dict[str, RedisPipelineIncrementOperation] = {}
                ops_to_remove = []
                for idx, op in enumerate(self.redis_increment_operation_queue):
                    if op["key"] in compressed_ops:
                        # 同一个 key 已经出现过 → 把 value 累加到已有的 op 上
                        compressed_ops[op["key"]]["increment_value"] += op[
                            "increment_value"
                        ]
                    else:
                        # 第一次出现该 key → 直接放入 dict
                        compressed_ops[op["key"]] = op

                    # 记录所有被处理过的索引，后续用来从原队列里移除
                    ops_to_remove.append(idx)

                # 合并后的操作列表（每个 key 只出现一次）
                compressed_queue = list(compressed_ops.values())

                # ---- Step 2: 通过 Redis pipeline 一次性批量 INCR ----
                increment_result = (
                    await self.dual_cache.redis_cache.async_increment_pipeline(
                        increment_list=compressed_queue,
                    )
                )

                # ---- Step 3: 清理已经写入的条目，只保留处理期间新塞进来的 ----
                # 注意：这里的过滤是按"处理时的索引"来的，在 async 期间可能又有新的 op 入队，
                # 那些新 op 的索引一定 >= 之前记录的 max(ops_to_remove)，不会被误删
                self.redis_increment_operation_queue = [
                    op
                    for idx, op in enumerate(self.redis_increment_operation_queue)
                    if idx not in ops_to_remove
                ]

                # ---- Step 4: 把返回结果组装成 { key: redis_最新值 } 方便上层使用 ----
                if increment_result is not None:
                    return_result = {
                        key["key"]: op
                        for key, op in zip(compressed_queue, increment_result)
                    }
                else:
                    return_result = {}
                return return_result

        except Exception as e:
            # 写 Redis 失败时丢弃整个队列，避免无限堆积引发 OOM。
            # 此时本次增量已经写入了内存，短暂的不一致会在下一次 sync 被修正。
            verbose_router_logger.error(
                f"Error syncing in-memory cache with Redis: {str(e)}"
            )
            self.redis_increment_operation_queue = []

    def add_to_in_memory_keys_to_update(self, key: str):
        """记录一个需要在下次 sync 时"从 Redis 拉最新值回内存"的 key。"""
        self.in_memory_keys_to_update.add(key)

    def get_key_pattern_to_sync(self) -> Optional[str]:
        """
        Get the key pattern to sync

        子类可覆盖：返回一个 Redis key 的 pattern（如 "provider_spend:*"），
        用于支持 scan_iter 的 Redis 实现时按模式扫描待同步的 key。
        默认返回 None，即走"记录型"路径（靠 in_memory_keys_to_update 集合）。
        """
        return None

    def get_in_memory_keys_to_update(self) -> Set[str]:
        """获取待同步的 key 集合（非原子）。"""
        return self.in_memory_keys_to_update

    def get_and_reset_in_memory_keys_to_update(self) -> Set[str]:
        """Atomic get and reset in-memory keys to update

        原子地取出并清空待同步 key 集合。
        适合在某一次 sync 开始时调用，避免读写竞态（后续新进来的 key 会进新集合，
        下一轮 sync 再处理）。
        """
        keys = self.in_memory_keys_to_update
        self.in_memory_keys_to_update = set()
        return keys

    def reset_in_memory_keys_to_update(self):
        """清空待同步 key 集合。"""
        self.in_memory_keys_to_update = set()

    async def _sync_in_memory_spend_with_redis(self):
        """
        Ensures in-memory cache is updated with latest Redis values for all provider spends.

        Why Do we need this?
        - Optimization to hit sub 100ms latency. Performance was impacted when redis was used for read/write per request
        - Use provider budgets in multi-instance environment, we use Redis to sync spend across all instances

        What this does:
        1. Push all provider spend increments to Redis
        2. Fetch all current provider spend from Redis to update in-memory cache

        【核心】一次完整的"本地 ↔ Redis"双向同步。

        目的：
        - 性能：热路径只读/写内存，达到亚 100ms 的延迟；Redis 操作都塞到后台批量做
        - 一致性：多实例环境下通过 Redis 汇总所有实例的增量，再广播回内存

        流程（对应下方代码的 4 个步骤）：
        1. 记录 sync 开始前的内存快照（in_memory_before_dict）
        2. 把本实例累积的增量推到 Redis（_push_in_memory_increments_to_redis）
        3. 读取同步期间内存又产生的增量（sync 过程并不阻塞请求，所以 after 可能比 before 更大）
        4. 合并策略：
           - Redis 侧反映了"全量值"（含所有实例的贡献）
           - 本实例在 sync 期间又新增了 delta = after - before
           - 最终写回内存的值 = redis_val + delta
           - 但如果 after > redis_val（内存超前于 Redis，通常说明 Redis 还没拿到本实例的最新 push），
             暂时跳过这个 key，等下一轮 sync 再合并，防止把本地的较大值覆盖成较小值
        """

        try:
            # 纯内存模式直接跳过，不需要做跨实例同步
            # No need to sync if Redis cache is not initialized
            if self.dual_cache.redis_cache is None:
                return

            # ---- 步骤 1：确定这轮要同步哪些 key ----
            # 2. Fetch all current provider spend from Redis to update in-memory cache
            cache_keys = (
                self.get_in_memory_keys_to_update()
            )  # if no pattern OR redis cache does not support scan_iter, use in-memory keys

            cache_keys_list = list(cache_keys)

            # ---- 步骤 2：对这些 key 拍个"sync 开始前"的内存快照 ----
            # 1. Snapshot in-memory before
            in_memory_before_dict = {}
            in_memory_before = (
                await self.dual_cache.in_memory_cache.async_batch_get_cache(
                    keys=cache_keys_list
                )
            )
            for k, v in zip(cache_keys_list, in_memory_before):
                # 缺失的 key 视作 0，避免后续 float(None) 报错
                in_memory_before_dict[k] = float(v or 0)

            # ---- 步骤 3：把队列里累积的增量批量推到 Redis，并拿回 Redis 侧的当前值 ----
            # 1. Push all provider spend increments to Redis
            redis_values = await self._push_in_memory_increments_to_redis()
            if redis_values is None:
                # 没有任何操作被推送 / Redis 不可用 → 本轮直接结束
                return

            # ---- 步骤 4：把 Redis 最新值和"sync 期间新增的 delta"合并写回内存 ----
            # 4. Merge
            for key in cache_keys_list:
                # Redis 汇总值（包含其他实例的贡献）
                redis_val = float(redis_values.get(key, 0) or 0)
                # sync 开始前内存里该 key 的值
                before = float(in_memory_before_dict.get(key, 0) or 0)
                # sync 结束时内存里该 key 的值（期间可能又有请求累加）
                after = float(
                    await self.dual_cache.in_memory_cache.async_get_cache(key=key) or 0
                )
                # 本实例在 sync 期间新增的增量
                delta = after - before

                # 合并策略：
                # - 正常情况：after <= redis_val（Redis 已经包含了本实例 push 的所有增量，
                #   且可能还包含其他实例的贡献）→ 最终内存值 = redis_val + 期间新增 delta
                # - 异常情况：after > redis_val（Redis 暂时落后于本地，可能是 Redis pipeline
                #   还没来得及 apply）→ 本轮先跳过，等下轮 sync 再修正，避免把本地的较大值
                #   "回退"成较小的 Redis 值
                if after <= redis_val:
                    merged = redis_val + delta
                else:
                    continue

                # 下面这段被注释掉的代码是早期 debug 版本：一旦出现 Redis 落后的情况直接 _exit，
                # 仅供排查多实例不一致 bug 时参考，生产中保留"continue 跳过"的温和处理即可
                # elif "rpm" in key:  # redis is behind in-memory cache
                #     # shut down the proxy
                #     print(f"self.redis_increment_operation_queue: {self.redis_increment_operation_queue}")
                #     print(f"Redis_val={redis_val} is behind in-memory cache_val={after} for key: {key}. This should not happen, since we should be updating redis with in-memory cache.")
                #     import os
                #     os._exit(1)
                #     raise Exception(f"Redis is behind in-memory cache for key: {key}. This should not happen, since we should be updating redis with in-memory cache.")

                # 把合并后的最终值写回内存，让后续请求看到最新视图
                await self.dual_cache.in_memory_cache.async_set_cache(
                    key=key, value=merged
                )

        except Exception as e:
            # 任何阶段异常都吞掉并打 exception 级日志（带堆栈），
            # 绝对不能让后台协程因为一次 sync 失败而整个挂掉
            verbose_router_logger.exception(
                f"Error syncing in-memory cache with Redis: {str(e)}"
            )

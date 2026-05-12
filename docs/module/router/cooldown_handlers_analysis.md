# Router Cooldown 冷却策略详解

> 源码位置：`litellm/router_utils/cooldown_handlers.py`
>
> 作用：当某个 deployment（模型实例 / 上游 endpoint）连续失败或触发限流时，将其从可选池中"冷却"一段时间，避免请求反复打到坏节点，配合 Fallback 机制形成完整的"熔断—降级"链路。

---

## 1. 设计目标

- **熔断坏节点**：识别失败率异常的 deployment，暂时下线。
- **保护单实例模型组**：避免因冷却唯一节点导致整组服务不可用。
- **双策略兼容**：同时支持 v2（错误率）和 v1（固定次数）两套冷却策略。
- **可观测**：触发冷却时异步上报事件，接入 Prometheus / OTel / Slack 等。

---

## 2. 判定流水线（三层过滤）

失败发生后，`_set_cooldown_deployments`（`cooldown_handlers.py:260`）按顺序执行三层检查，**全部通过**才真正进入冷却。

```
异常发生
  ↓
_should_run_cooldown_logic   (:98)    —— "能不能跑冷却逻辑"
  ↓
_is_cooldown_required        (:40)    —— "这个状态码要不要冷却"
  ↓
_should_cooldown_deployment  (:166)   —— "综合失败率 / 策略要不要冷却"
  ↓
cooldown_cache.add_deployment_to_cooldown
  + asyncio.create_task(router_cooldown_event_callback)
```

---

### 2.1 第一层：`_should_run_cooldown_logic`（:98–163）

下列任一条件成立 → **直接跳过冷却**：

| 条件 | 说明 |
|---|---|
| `deployment is None` 或 `get_model_group(id=deployment) is None` | 无法归属模型组 |
| `time_to_cooldown ≈ 0`（`math.isclose(abs_tol=1e-9)`） | 调用方显式传 0 秒冷却 |
| `router.disable_cooldowns == True` | 全局关闭冷却 |
| `_is_cooldown_required` 返回 False | 状态码不该冷却 |
| `deployment in provider_default_deployment_ids` | 供应商默认兜底节点，禁止剔除 |

---

### 2.2 第二层：`_is_cooldown_required`（:40–95）—— 按 HTTP 状态码判断

| 异常 / 状态码 | 行为 | 原因 |
|---|---|---|
| 异常字符串含 `APIConnectionError` | **不冷却** | 客户端侧连接异常，多半是本地问题 |
| `429` | **冷却** | Rate Limit |
| `401` | **冷却** | 鉴权错误，key 多半失效 |
| `408` | **冷却** | 超时 |
| `404` | **冷却** | 模型 / endpoint 不存在 |
| 其他 `4xx` | **不冷却** | 客户端请求问题（比如 400 参数错），冷却节点无意义 |
| `5xx` 或非标准状态 | **冷却** | 服务端错误 |
| 状态解析失败 | **冷却**（兜底） | 未知情况按保守策略处理 |

---

### 2.3 第三层：`_should_cooldown_deployment`（:166–257）—— v2 错误率策略

根据 `allowed_fails_policy` / `allowed_fails` 是否设置，走不同路径。

#### 路径 A — v2 主流程（默认）

条件：`allowed_fails_policy is None` 且 `_is_allowed_fails_set_on_router` 返回 False。

步骤：

1. 从 `router_callbacks.track_deployment_metrics` 获取**本分钟**计数：
   - `num_successes_this_minute`
   - `num_fails_this_minute`
2. 计算 `percent_fails = fails / (successes + fails)`。
3. 按优先级判定：

| 序号 | 条件 | 结果 |
|---|---|---|
| 1 | `status == 429` 且 **非**单实例模型组 | 冷却 |
| 2 | `percent_fails == 1.0` 且 `total >= SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD` | 冷却（全挂且有流量） |
| 3 | `percent_fails > DEFAULT_FAILURE_THRESHOLD_PERCENT` 且 `total >= DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS` 且**非**单实例组 | 冷却 |
| 4 | `litellm._should_retry(status) is False` | 冷却（不应重试的错统一冷却） |
| 5 | 以上都不成立 | 不冷却 |

核心阈值常量（`litellm/constants.py`）：

- `DEFAULT_COOLDOWN_TIME_SECONDS` — 默认冷却时长
- `DEFAULT_FAILURE_THRESHOLD_PERCENT` — 错误率阈值
- `DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS` — 最小样本量
- `SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD` — 100% 失败时要求的最小流量

#### 路径 B — v1 Legacy（按失败次数）

条件：显式设置了 `allowed_fails_policy` 或非默认 `allowed_fails`。

进入 `should_cooldown_based_on_allowed_fails_policy`（:398–430）：

```python
allowed_fails = router.get_allowed_fails_from_policy(exception) or router.allowed_fails
cooldown_time = router.cooldown_time or DEFAULT_COOLDOWN_TIME_SECONDS
current_fails = router.failed_calls.get_cache(deployment) or 0
updated_fails = current_fails + 1

if updated_fails > allowed_fails:
    return True  # 触发冷却
else:
    router.failed_calls.set_cache(deployment, updated_fails, ttl=cooldown_time)
    return False
```

---

## 3. 单实例模型组保护

`is_single_deployment_model_group`（:193）贯穿策略判断：

- `429` 限流 → **不冷却**单实例组
- 错误率超阈值 → **不冷却**单实例组
- 但"100% 失败 + 有流量"仍然冷却（节点确实已挂）

**动机**：若某个 `model_name` 只挂一个实例，把它冷却将导致整组请求无处可去。

---

## 4. 冷却落地与恢复

### 4.1 写入

```python
litellm_router_instance.cooldown_cache.add_deployment_to_cooldown(
    model_id=deployment,
    original_exception=original_exception,
    exception_status=exception_status_int,
    cooldown_time=time_to_cooldown,
)
```

- `CooldownCache` 支持本地内存 / Redis 两种后端（见 `cooldown_cache.py`）。
- TTL 到期自动恢复。

### 4.2 事件上报

```python
asyncio.create_task(
    router_cooldown_event_callback(
        litellm_router_instance=router,
        deployment_id=deployment,
        exception_status=exception_status,
        cooldown_time=time_to_cooldown,
    )
)
```

非阻塞触发，用于对接 Prometheus / OTel / Slack 告警（见 `cooldown_callbacks.py`）。

### 4.3 查询接口

| 函数 | 说明 |
|---|---|
| `_get_cooldown_deployments`（:369） | 同步返回冷却中的 `model_id` 列表 |
| `_async_get_cooldown_deployments`（:323） | 异步版本 |
| `_async_get_cooldown_deployments_with_debug_info`（:351） | 返回 `(model_id, status, ...)` 元组，供 debug / UI |

Router 在挑选下一个 deployment 前先调用这些接口剔除冷却节点。

---

## 5. 可配置参数

| 参数 | 作用 | 默认值 |
|---|---|---|
| `router.disable_cooldowns` | 总开关 | False |
| `router.cooldown_time` | 冷却时长（秒） | `DEFAULT_COOLDOWN_TIME_SECONDS` |
| `router.allowed_fails` | v1 允许失败次数 | None |
| `router.allowed_fails_policy` | v1 按异常类型分别配允许失败次数 | None |
| 调用侧 `time_to_cooldown` | 单次覆盖冷却时长（0 = 跳过） | None |
| `router.provider_default_deployment_ids` | 永不冷却的节点 ID 集合 | `[]` |

---

## 6. 与 Fallback 的配合

```
[请求进入]
  ↓
Router.async_function_with_fallbacks  (router.py:5573)
  ↓
挑选 deployment（自动剔除 cooldown 列表）
  ↓
调用失败 → 异常
  ↓
_set_cooldown_deployments  (根据三层漏斗决定是否熔断该节点)
  ↓
Retry 本模型组其他 deployment
  ↓
整组用尽 → fallbacks / context_window_fallbacks / content_policy_fallbacks
  ↓
触发上层降级链
```

冷却与 Fallback 各司其职：

- **Cooldown**：在**同一模型组内**剔除坏节点。
- **Fallback**：在**模型组用尽后**切换到备用模型组。

---

## 7. 关键函数速查

| 函数 | 行号 | 作用 |
|---|---|---|
| `_is_cooldown_required` | :40 | 按状态码判断是否需要冷却 |
| `_should_run_cooldown_logic` | :98 | 前置门控（全局开关 / 默认节点 / 归属检查） |
| `_should_cooldown_deployment` | :166 | v2 错误率 / 路径选择 |
| `_set_cooldown_deployments` | :260 | 主入口，编排三层判断 + 写缓存 + 发事件 |
| `_async_get_cooldown_deployments` | :323 | 异步取冷却列表 |
| `_async_get_cooldown_deployments_with_debug_info` | :351 | 取冷却列表（带调试信息） |
| `_get_cooldown_deployments` | :369 | 同步取冷却列表 |
| `should_cooldown_based_on_allowed_fails_policy` | :398 | v1 按次数策略 |
| `_is_allowed_fails_set_on_router` | :433 | 判断是否启用 v1 策略 |
| `cast_exception_status_to_int` | :450 | 状态码统一转 int（异常默认 500） |

---

## 8. 小结

`cooldown_handlers.py` 是一套 **三层漏斗 + 双策略** 的熔断设计：

1. **能跑吗？**（上下文 / 开关 / 默认节点）
2. **该冷吗？**（状态码语义）
3. **到阈值了吗？**（v2 错误率 / v1 次数）

配合**单实例模型组保护**避免熔断导致服务完全不可用，并通过 callback 与 fallback 形成可观测、可降级的闭环。

相比简单的"N 次失败就拉黑 T 秒"，它更关注**失败比例**和**流量量级**，避免低流量下被个别抖动误熔断；同时保留 v1 路径，便于从旧配置平滑迁移。

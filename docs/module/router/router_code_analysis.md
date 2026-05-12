# LiteLLM 模型路由系统代码分析

## 1. 概述

LiteLLM 路由系统为 100+ LLM 提供商提供统一的负载均衡、故障转移和智能路由能力。核心入口位于 `litellm/router.py` 中的 `Router` 类，配合 `litellm/router_strategy/`（路由策略）和 `litellm/router_utils/`（路由工具）两个子模块协同工作。

### 核心架构图

```
用户请求
  │
  ▼
Router.acompletion() / Router.completion()
  │
  ├── _update_kwargs_before_fallbacks()    # 请求预处理
  │
  ▼
async_function_with_fallbacks()            # 重试与回退包装器
  │
  ├── _acompletion() (原始调用函数)
  │     │
  │     ▼
  │   get_available_deployment()            # 核心路由决策
  │     │
  │     ├── _common_checks_available_deployment()  # 部署过滤
  │     ├── _filter_health_check_unhealthy_deployments()  # 健康检查过滤
  │     ├── _filter_cooldown_deployments()  # 冷却期过滤
  │     ├── _pre_call_checks()             # 预调用检查（上下文窗口、速率限制等）
  │     ├── _get_order_filtered_deployments()  # 顺序过滤
  │     │
  │     ▼
  │   [路由策略选择] ──► simple_shuffle / least_busy / lowest_latency / ...
  │     │
  │     ▼
  │   选中的 deployment
  │
  ├── 成功 → 返回响应
  │
  └── 失败 → async_function_with_fallbacks_common_utils()
        │
        ├── 重试逻辑（基于 retry_policy）
        ├── 顺序回退（order-based fallbacks）
        ├── 模型组回退（model group fallbacks）
        ├── 上下文窗口回退（context_window_fallbacks）
        └── 内容策略回退（content_policy_fallbacks）
```

## 2. 核心文件结构

### 2.1 主路由器 — `litellm/router.py`

| 关键方法 | 行号 | 功能 |
|---------|------|------|
| `Router.__init__()` | ~225 | 初始化缓存、调度器、回退配置、路由策略 |
| `routing_strategy_init()` | ~799 | 根据配置初始化对应的路由策略处理器 |
| `acompletion()` | ~1690 | 异步补全入口，包装 fallback 逻辑 |
| `async_function_with_fallbacks()` | ~5573 | 核心重试和回退逻辑 |
| `async_function_with_fallbacks_common_utils()` | ~5318 | 回退公共处理（顺序回退、模型组回退） |
| `get_available_deployment()` | ~9650 | **核心路由决策方法**：过滤 + 策略选择 |
| `deployment_callback_on_success()` | — | 成功回调，更新部署指标 |
| `deployment_callback_on_failure()` | — | 失败回调，触发冷却逻辑 |

### 2.2 路由策略 — `litellm/router_strategy/`

| 文件 | 类名 | 策略 | 原理 |
|------|------|------|------|
| `simple_shuffle.py` | `simple_shuffle()` | `simple-shuffle` (默认) | 随机选取；支持按 `weight`/`rpm`/`tpm` 加权随机 |
| `least_busy.py` | `LeastBusyLoggingHandler` | `least-busy` | 追踪各部署在途请求数，选最少的 |
| `lowest_latency.py` | `LowestLatencyLoggingHandler` | `latency-based-routing` | 记录响应延迟（流式为首 token 时间），选最低延迟部署 |
| `lowest_cost.py` | `LowestCostLoggingHandler` | `cost-based-routing` | 追踪每个部署的 token 消耗成本，选最低成本 |
| `lowest_tpm_rpm.py` | `LowestTPMLoggingHandler` | `usage-based-routing` | 追踪 TPM/RPM 使用量，选离限额最远的部署 |
| `lowest_tpm_rpm_v2.py` | `LowestTPMLoggingHandler_v2` | `usage-based-routing-v2` | v2 版本，继承 `BaseRoutingStrategy`，支持批量 Redis 写入 |
| `budget_limiter.py` | `RouterBudgetLimiting` | (过滤器) | 提供商级别预算限制，过滤超预算部署 |
| `tag_based_routing.py` | `get_deployments_for_tag()` | (过滤器) | 基于标签的团队路由，匹配请求标签和部署标签 |
| `base_routing_strategy.py` | `BaseRoutingStrategy` | (基类) | 抽象基类，封装 Redis 批量写入和内存/Redis 同步逻辑 |
| `auto_router/auto_router.py` | `AutoRouter` | (语义路由) | 基于 SemanticRouter 的自动路由，使用语义匹配将请求分配到不同模型 |
| `complexity_router/` | `ComplexityRouter` | (复杂度路由) | 基于规则的复杂度评分路由，<1ms 本地评分，无外部 API 调用 |

### 2.3 路由工具 — `litellm/router_utils/`

| 文件 | 功能 |
|------|------|
| `cooldown_cache.py` | `CooldownCache` — 管理部署冷却状态的缓存封装 |
| `cooldown_handlers.py` | 冷却逻辑核心：判断是否需要冷却、设置/获取冷却部署列表 |
| `cooldown_callbacks.py` | 冷却事件回调（Prometheus 指标上报等） |
| `fallback_event_handlers.py` | 回退逻辑：`run_async_fallback()`、`get_fallback_model_group()` |
| `handle_error.py` | 错误处理：无可用部署时的异常抛出和告警 |
| `health_state_cache.py` | `DeploymentHealthCache` — 后台健康检查状态缓存 |
| `pattern_match_deployments.py` | `PatternMatchRouter` — 通配符/正则模型名匹配 |
| `client_initalization_utils.py` | `InitalizeCachedClient` — HTTP/SDK 客户端缓存初始化 |
| `clientside_credential_handler.py` | 客户端凭证处理（动态 API key 等） |
| `common_utils.py` | 通用过滤函数（团队模型过滤、web search 过滤等） |
| `batch_utils.py` | 批处理工具（JSONL 模型替换等） |
| `get_retry_from_policy.py` | 根据重试策略获取重试次数 |
| `add_retry_fallback_headers.py` | 在响应头中添加重试/回退信息 |
| `response_headers.py` | 响应头处理 |
| `prompt_caching_cache.py` | Prompt 缓存索引（用于缓存亲和性路由） |
| `search_api_router.py` | 搜索 API 路由 |

#### 预调用检查 — `litellm/router_utils/pre_call_checks/`

| 文件 | 功能 |
|------|------|
| `deployment_affinity_check.py` | 部署亲和性检查：API key → 部署映射、Responses API 连续性 |
| `model_rate_limit_check.py` | 模型级别速率限制检查 |
| `prompt_caching_deployment_check.py` | Prompt 缓存部署检查：路由到之前缓存过的部署 |
| `responses_api_deployment_check.py` | Responses API 部署检查 |
| `encrypted_content_affinity_check.py` | 加密内容亲和性检查 |

## 3. 路由决策流程详解

### 3.1 部署选择 (`get_available_deployment`)

这是路由系统的核心方法，位于 `router.py:9650`。流程如下：

```
1. _common_checks_available_deployment()
   ├── 模型名解析（别名、通配符、access group）
   ├── 获取该模型组的所有部署
   └── 标签过滤（tag_based_routing）

2. _filter_health_check_unhealthy_deployments()
   └── 基于后台健康检查结果过滤不健康部署

3. _filter_cooldown_deployments()
   └── 过滤处于冷却期的部署
   └── 若全部冷却 + 启用了健康检查路由 → 旁路冷却过滤

4. _pre_call_checks() （可选）
   ├── 上下文窗口检查
   ├── 速率限制检查
   ├── 预算限制检查
   ├── Prompt 缓存亲和性
   └── 部署亲和性（session stickiness）

5. _get_order_filtered_deployments()
   └── 按部署 order 字段过滤，优先选低 order 值

6. 路由策略选择（基于 self.routing_strategy）
   ├── "simple-shuffle"         → simple_shuffle()
   ├── "least-busy"             → LeastBusyLoggingHandler
   ├── "latency-based-routing"  → LowestLatencyLoggingHandler
   ├── "usage-based-routing"    → LowestTPMLoggingHandler
   ├── "usage-based-routing-v2" → LowestTPMLoggingHandler_v2
   └── "cost-based-routing"     → LowestCostLoggingHandler

7. 无可用部署 → 抛出 RouterRateLimitError
```

### 3.2 冷却机制 (`cooldown_handlers.py`)

冷却机制防止向持续出错的部署发送请求：

- **触发条件**：部署在 1 分钟内失败次数超过 `allowed_fails` 阈值
- **冷却判定**：`_is_cooldown_required()` 根据 HTTP 状态码决定
  - 429 (Rate Limit)、401 (Auth)、408 (Timeout)、404 (Not Found) → 触发冷却
  - 其他 4xx → 不触发冷却
  - 5xx 及其他错误 → 触发冷却
- **冷却时间**：默认 `DEFAULT_COOLDOWN_TIME_SECONDS`，支持动态设置
- **存储**：`CooldownCache` 使用 `DualCache`（内存 + Redis），支持多实例环境

### 3.3 回退机制 (`fallback_event_handlers.py`)

回退在 `async_function_with_fallbacks_common_utils()` 中协调，支持多种回退类型：

| 回退类型 | 配置参数 | 触发条件 |
|---------|---------|---------|
| 顺序回退 | 部署的 `order` 字段 | 当前 order 级别全部失败 |
| 模型组回退 | `fallbacks` | 指定模型组 → 备用模型组 |
| 默认回退 | `default_fallbacks` | 所有模型组的通用回退 |
| 上下文窗口回退 | `context_window_fallbacks` | `ContextWindowExceededError` |
| 内容策略回退 | `content_policy_fallbacks` | `ContentPolicyViolationError` |
| 通配符回退 | `fallbacks: [{"*": [...]}]` | 任何未匹配的模型组 |

回退深度由 `max_fallbacks`（默认 5）控制，避免无限级联。

### 3.4 中流回退 (Mid-Stream Fallback)

对于流式响应，路由器支持中流回退（`_acompletion_streaming_iterator`）：
- 捕获 `MidStreamFallbackError`
- 收集已生成的内容
- 使用回退模型继续生成，注入 continuation prompt
- 合并 usage 统计

## 4. 路由策略详解

### 4.1 Simple Shuffle（默认，推荐生产使用）

**文件**：`router_strategy/simple_shuffle.py`

最简单且性能最优的策略，从健康部署列表中随机选取。

**工作原理**：
- 若部署配置了 `weight`、`rpm` 或 `tpm`，按权重加权随机选取（使用 `random.choices` 加权抽样）
- 否则纯随机选取（`random.choice`）
- 权重优先级：`weight` > `tpm` > `rpm`

**配置参数**：

| 参数 | 位置 | 说明 |
|------|------|------|
| `weight` | `model_info` | 直接权重值，如 weight=9 表示 90% 概率被选中 |
| `rpm` | `litellm_params` | 每分钟请求数，作为加权因子 |
| `tpm` | `litellm_params` | 每分钟 token 数，作为加权因子 |

**使用建议**：适合大多数生产场景。无 Redis 依赖，延迟开销最小（仅一次随机操作），结合冷却机制即可实现基本的故障转移。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 实现极简（一次随机调用）；零 Redis 依赖、零状态；多实例天然一致；冷启动无负担；与冷却机制天然兼容 |
| ❌ 缺点 | 不感知部署的实时负载/延迟/成本；权重静态配置后无法自适应；流量倾斜需运维手动调整 weight |
| 🎯 适用 | 通用生产首选；多实例部署且无强追踪需求；只需 weight 控制配额比例的简单场景 |
| 🚫 不适用 | 部署间真实承载能力差异大；需要按实时延迟/用量动态分配 |

### 4.2 Least Busy（最少繁忙）

**文件**：`router_strategy/least_busy.py`
**类名**：`LeastBusyLoggingHandler`（继承 `CustomLogger`）

通过回调机制追踪每个部署的在途请求数（in-flight requests）。

**工作原理**：
- `log_pre_api_call()`: 请求开始时，对应部署的计数 +1
- `log_success_event()` / `log_failure_event()`: 请求完成（无论成功失败）时计数 -1
- 选择时，从候选部署中选择当前并发计数最小的

**数据存储**：使用 `router_cache`（DualCache）存储每个部署的在途请求计数，key 格式为 `{model_id}:in_flight`。

**使用建议**：适合请求处理时间差异大的场景（如长文本生成 vs 短问答混合）。无需额外配置参数，但需要注意在多实例部署时计数可能不精确（除非配置 Redis）。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 反映真实并发占用；适合长短请求混合；动态自适应，无需人工调权重 |
| ❌ 缺点 | 多实例下若不接 Redis，计数仅本地准确；遇到部署中途 hang 死时计数无法回收（依赖请求最终触发回调）；冷启动时所有部署计数为 0，退化为随机 |
| 🎯 适用 | 单实例 / 共享 Redis 的多实例；请求耗时方差大（chat + 长文生成混部）；后端容量异质 |
| 🚫 不适用 | 短请求为主、并发数差异不显著；不愿引入 Redis 又需多实例一致性 |

### 4.3 Lowest Latency（最低延迟）

**文件**：`router_strategy/lowest_latency.py`
**类名**：`LowestLatencyLoggingHandler`（继承 `CustomLogger`）

追踪每个部署的响应时间，路由到历史延迟最低的部署。

**工作原理**：
- 非流式请求：记录完整响应时间
- 流式请求：记录 TTFT（Time To First Token，首 token 时间）
- 超时请求：记录 1000 秒的惩罚延迟
- 延迟按 `completion_tokens` 归一化（秒/token），消除输出长度对延迟统计的干扰
- 维护每个部署最近 N 次延迟的滑动窗口

**配置参数**（通过 `routing_strategy_args` 传入）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ttl` | 3600 | 延迟数据的缓存时间窗口（秒），超过该时间的样本会被淘汰 |
| `lowest_latency_buffer` | 0 | 延迟缓冲区百分比（如 0.5 表示 50%），所有延迟在最低值 × (1+buffer) 范围内的部署都视为候选，随机选取 |
| `max_latency_list_size` | 10 | 每个部署保留的延迟样本数 |

**使用建议**：适合对响应速度敏感的场景。建议设置 `lowest_latency_buffer` 为 0.3~0.5，避免流量全部涌向最快的单个部署导致其过载。冷启动时（无延迟数据），回退到随机选择。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 流式请求用 TTFT、非流式用整请求耗时，指标更贴近用户体验；按 token 归一化避免输出长度干扰；`lowest_latency_buffer` 可避免单点过载 |
| ❌ 缺点 | 滑动窗口需要持续样本，冷启动期效果差；超时惩罚（1000s）对偶发抖动惩罚过重；无 buffer 时容易把所有流量打到当前最快节点造成新热点 |
| 🎯 适用 | C 端在线服务、首 token 时间敏感的 chat / 流式场景；同一模型组下后端实例延迟差异大（跨地域、跨厂商） |
| 🚫 不适用 | 离线批处理（不在乎个体延迟）；部署数过少（统计无意义）；不能容忍冷启动随机分布 |

### 4.4 Usage-Based Routing（基于用量）

**文件**：`router_strategy/lowest_tpm_rpm.py` (v1) / `lowest_tpm_rpm_v2.py` (v2)
**类名**：`LowestTPMLoggingHandler` (v1) / `LowestTPMLoggingHandler_v2` (v2)

追踪每个部署每分钟的 TPM 和 RPM 使用量，路由到使用率最低的部署。

**工作原理**：
- 记录每个部署当前分钟已消耗的 TPM 和 RPM
- 计算使用率：`usage_ratio = current_usage / configured_limit`
- 选择使用率最低的部署
- 若某部署已超限（使用率 ≥ 1），自动从候选集中剔除

**v1 vs v2 的区别**：

| 维度 | v1 | v2 |
|------|----|----|
| 同步方式 | 同步 Redis 调用 | 异步 Redis 调用（`redis.incr` + `redis.mget`） |
| 基类 | `CustomLogger` | `BaseRoutingStrategy` |
| 批量写入 | 不支持 | 支持 Redis pipeline 批量写入 |
| 缓存粒度 | model_group 级别 | 单个 model（deployment）级别 |
| 预检查 | 不支持 | 支持 `pre_call_check`（请求发起前检查 RPM） |
| 适用场景 | 单实例 | 多实例生产环境 |

**配置参数**：

| 参数 | 位置 | 说明 |
|------|------|------|
| `tpm` | `litellm_params` | 部署的 TPM 上限 |
| `rpm` | `litellm_params` | 部署的 RPM 上限 |
| `routing_strategy_args.ttl` | `router_settings` | 缓存 TTL，默认 60 秒 |
| `enable_pre_call_checks` | `router_settings` | 启用预调用检查（v2） |

**使用建议**：当模型部署有明确的 TPM/RPM 限额时使用。v2 适合多实例部署，但 Redis 操作会增加延迟。文档建议在不需要跨实例精确追踪时优先使用 `simple-shuffle` + weight。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 显式遵守上游配额，避免触发 429；超限自动剔除；v2 支持 pre-call check 提前拦截；多实例下精度高（v2 + Redis） |
| ❌ 缺点 | 必须为每个 deployment 配置准确的 `tpm`/`rpm`，配错则失效；v1 同步 Redis 调用阻塞热路径；额外 Redis 网络延迟（v2 异步缓解但仍存在） |
| 🎯 适用 | Azure / OpenAI 等有官方 RPM/TPM 配额的部署；同一模型组下多个 key/region 配额拼接；需要 SLA 级别避免 429 的生产环境 |
| 🚫 不适用 | 部署没有可量化的 RPM/TPM；纯路由能力优先于配额管理 → 用 simple-shuffle 即可 |

### 4.5 Cost-Based Routing（基于成本）

**文件**：`router_strategy/lowest_cost.py`
**类名**：`LowestCostLoggingHandler`（继承 `CustomLogger`）

从候选部署中选择 token 单价最低的部署。

**工作原理**：
1. 获取所有健康部署
2. 过滤掉超出 rpm/tpm 限制的部署
3. 查询 `litellm_model_cost_map`（全局模型成本映射表）获取每个部署的 input/output token 单价
4. 选择加权成本最低的部署
5. 未在成本表中的模型默认 $1（最高优先级最低）

**配置参数**：

| 参数 | 位置 | 说明 |
|------|------|------|
| `input_cost_per_token` | `litellm_params` | 自定义输入 token 单价（覆盖默认成本表） |
| `output_cost_per_token` | `litellm_params` | 自定义输出 token 单价 |

**使用建议**：适合有多个功能等价但价格不同的部署时（如 Azure vs OpenAI 的同款模型）。确保 `model_prices_and_context_window.json` 中有对应模型的定价数据，或通过 `litellm_params` 手动指定。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 直接优化账单；无 Redis 依赖；多实例天然一致（成本表是只读全局数据） |
| ❌ 缺点 | 只看单价不看延迟/可用性，可能把流量全压到便宜但慢/不稳的部署；未在成本表中的模型按 $1 兜底，可能偏离真实价格；成本表升级未必及时 |
| 🎯 适用 | 同一能力级别的模型有多个价位（OpenAI vs Azure vs Bedrock 同款）；离线/批量任务对延迟不敏感；与 `provider_budget_config` 配合做硬上限 |
| 🚫 不适用 | 在线 C 端、对延迟稳定性敏感的场景；部署间能力不对等（便宜模型质量差） |

### 4.6 Tag-Based Routing（基于标签过滤器）

**文件**：`router_strategy/tag_based_routing.py`
**函数**：`get_deployments_for_tag()`

作为过滤器工作（在主路由策略之前执行，可与任何策略组合）。

**工作原理**：
- 请求通过 `metadata.tags` 携带标签
- 部署通过 `model_info.tags` 配置标签
- 匹配逻辑：请求标签必须是部署标签的子集
- 支持两种匹配模式：
  - `match_any`：请求的任一 tag 匹配即可
  - `match_all`：请求的所有 tag 都必须匹配
- 支持正则表达式标签匹配

**配置**：
```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4
    model_info:
      tags: ["production", "team-a"]

router_settings:
  enable_tag_filtering: true
```

**使用建议**：适合多团队共享 proxy 时，按团队/环境/业务线隔离流量。没有标签的请求会回退到未配置标签的默认部署。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 与任意主路由策略正交组合；声明式配置清晰；支持正则匹配灵活分组；天然实现多租户隔离 |
| ❌ 缺点 | 只是过滤器不做选择，仍需搭配主策略；标签维护成本随团队/环境数量上升；标签错配可能让请求被静默路由到默认部署 |
| 🎯 适用 | 多团队 / 多环境（dev/staging/prod）共享同一 proxy；按业务线隔离配额或机房；金丝雀放量（按 tag 引流到新部署） |
| 🚫 不适用 | 单租户、单业务线场景，强行加 tag 反而徒增复杂度 |

### 4.7 Budget Limiter（Provider 预算限制过滤器）

**文件**：`router_strategy/budget_limiter.py`
**类名**：`RouterBudgetLimiting`（继承 `BaseRoutingStrategy`）

作为过滤器工作（可与其他路由策略组合），为每个 LLM Provider 设置预算上限。

**工作原理**：
1. 通过回调追踪每个 Provider（如 openai、anthropic、azure）的累计花费
2. 在路由决策时，过滤掉已超预算 Provider 对应的所有部署
3. 使用双层缓存（内存 + Redis）+ 批量异步同步架构保证性能

**配置参数**（通过 `provider_budget_config`）：

| 参数 | 说明 |
|------|------|
| `budget_limit` | 预算金额（美元） |
| `time_period` | 预算周期，支持 `1d`（天）、`7d`（周）、`1mo`（月） |

**配置示例**：
```yaml
router_settings:
  provider_budget_config:
    openai:
      budget_limit: 100
      time_period: 1d
    anthropic:
      budget_limit: 200
      time_period: 7d
```

**使用建议**：适合需要控制各 Provider 花费总额的场景。超预算后，该 Provider 的所有部署自动从候选集中移除，请求会路由到其他未超预算的 Provider。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 硬性预算护栏，防止账单失控；与所有主策略正交；支持日/周/月周期；内存 + Redis 双层缓存延迟低 |
| ❌ 缺点 | 粒度是 Provider 级，不能细到 deployment / team；预算耗尽后请求可能整体失败（若没有其他 provider 兜底）；依赖 callback 累积花费，回调延迟会导致超支几秒 |
| 🎯 适用 | 多 Provider 部署且各 Provider 有独立账单上限；防止某个昂贵 Provider 失控烧钱；和 fallbacks 链组合实现「便宜耗尽 → 切贵的」 |
| 🚫 不适用 | 单一 Provider；需要 team/key 维度预算 → 用 LiteLLM 的 budget 体系而非这里 |

### 4.8 Auto Router（语义路由）

**文件**：`router_strategy/auto_router/auto_router.py`
**类名**：`AutoRouter`

基于 `semantic-router` 库的语义路由，通过请求内容的语义相似度将请求分配到不同模型。

**工作原理**：
- 预先配置每个模型的 utterances（示例话语），描述该模型擅长处理的请求类型
- 使用嵌入模型计算请求与各 utterance 的语义相似度
- 将请求路由到语义最匹配的模型

**配置方式**：deployment 的 model 以 `auto_router/` 前缀标识，路由规则通过 JSON 配置文件指定。

**依赖**：需要安装 `semantic_router` 库。

**使用建议**：适合需要根据请求内容智能分流的场景（如简单问答走便宜模型，复杂推理走高端模型）。但引入了嵌入模型调用的额外延迟。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 真正按请求语义匹配模型，分流效果好；utterances 配置直观；可同时实现路由和能力分级 |
| ❌ 缺点 | 每次路由要做一次嵌入计算，引入额外延迟（百毫秒级）和成本；utterances 配置质量直接决定路由质量；依赖外部 `semantic_router` 库 |
| 🎯 适用 | 不同请求类型差异显著（代码 / 翻译 / 问答 / 长文）的统一入口；多模型矩阵（专长模型 + 通用模型）智能分发 |
| 🚫 不适用 | 对路由本身的延迟敏感（OLTP 场景）；模型间能力同质，分流意义不大 |

### 4.9 Complexity Router（复杂度路由）

**文件**：`router_strategy/complexity_router/complexity_router.py` + `config.py`
**类名**：`ComplexityRouter`

基于规则的本地复杂度评分系统，将请求按复杂度分级后路由到对应模型。

**评分维度与权重**：

| 维度 | 权重 | 评分逻辑 |
|------|------|----------|
| tokenCount | 10% | 短文本=简单，长文本=复杂 |
| codePresence | 30% | 检测代码相关关键词（function、class、import 等） |
| reasoningMarkers | 25% | 检测推理标记词（analyze、prove、compare 等） |
| technicalTerms | 25% | 检测技术术语（algorithm、architecture 等） |
| simpleIndicators | 5% | 检测简单指标（hi、hello、thanks 等），负权重 |
| multiStepPatterns | 3% | 检测多步骤模式（first...then、step by step 等） |
| questionComplexity | 2% | 问题结构复杂度 |

**复杂度分级**：

| 级别 | 分数范围 | 路由目标 |
|------|----------|----------|
| SIMPLE | 0 ~ 0.25 | 轻量级模型（如 GPT-3.5） |
| MEDIUM | 0.25 ~ 0.5 | 中等模型（如 GPT-4-mini） |
| COMPLEX | 0.5 ~ 0.75 | 高端模型（如 GPT-4） |
| REASONING | 0.75 ~ 1.0 | 推理模型（如 o1） |

**配置方式**：deployment 的 model 以 `auto_router/complexity_router` 前缀标识。

**使用建议**：适合混合负载场景，自动将简单请求路由到便宜模型、复杂请求路由到高端模型。完全本地计算 <1ms，无外部 API 调用，无额外依赖。灵感来自 ClawRouter 项目。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 纯本地评分 <1ms，无外部依赖、无额外费用；规则透明可调；天然实现「便宜模型兜底 + 复杂任务上高端模型」的成本优化 |
| ❌ 缺点 | 基于关键词/启发式规则，对中文 / 多语言场景效果未必好；规则需根据业务调参；对不在评分维度的复杂度（如领域专业度）无感 |
| 🎯 适用 | 简单问答 + 复杂推理混合的统一入口；对延迟敏感、不想引入嵌入调用的成本优化场景；轻量级请求分级 |
| 🚫 不适用 | 请求语义需要细粒度判断（用 Auto Router）；非英语主导且未调过参的语料 |

### 4.10 Custom Routing Strategy（自定义路由）

**基类**：`litellm/types/router.py` 中的 `CustomRoutingStrategyBase`

允许用户完全自定义路由逻辑。

**实现方式**：
```python
from litellm.types.router import CustomRoutingStrategyBase

class MyRoutingStrategy(CustomRoutingStrategyBase):
    async def async_get_available_deployment(self, model, healthy_deployments, messages, ...):
        # 自定义选择逻辑
        return chosen_deployment

    def get_available_deployment(self, model, healthy_deployments, messages, ...):
        # 同步版本
        return chosen_deployment

# 注入到 Router
router.set_custom_routing_strategy(MyRoutingStrategy())
```

**使用建议**：当内置策略无法满足需求时使用，如需要基于请求内容、用户身份、时间段等自定义维度做路由。

**优缺点与场景**：

| 维度 | 说明 |
|------|------|
| ✅ 优点 | 完全可控，可融合任意业务维度（用户分层、时间段、灰度比例、A/B 实验）；同步/异步双签名 |
| ❌ 缺点 | 维护成本高，需自行处理冷却、健康过滤、并发安全；逻辑 bug 直接影响线上路由稳定性；不享受内置策略的指标收集与可观测性 |
| 🎯 适用 | 内置策略组合无法表达的特殊业务（如 VIP 优先 / 时间段切流 / 实验分桶）；需要把外部系统决策结果带入路由 |
| 🚫 不适用 | 内置策略组合已能覆盖的场景，避免重复造轮子 |

## 5. 缓存架构

路由系统使用多级缓存：

```
DualCache
├── InMemoryCache    ← 热路径读写（<1ms）
└── RedisCache       ← 多实例同步（可选）
    └── RedisPipelineIncrementOperation  ← 批量写入优化
```

**缓存用途**：
- **冷却状态**：`CooldownCache` — 部署冷却期信息
- **健康状态**：`DeploymentHealthCache` — 后台健康检查结果
- **路由指标**：TPM/RPM 计数、延迟统计、请求计数
- **Prompt 缓存**：`PromptCachingCache` — prompt 到部署的映射
- **部署亲和性**：API key → deployment 映射
- **HTTP 客户端**：`LLMClientCache` — SDK/HTTP 客户端复用

**Redis 同步机制** (`BaseRoutingStrategy`):
1. 请求热路径中只写内存缓存
2. 增量操作入队到 `redis_increment_operation_queue`
3. 定期合并压缩后批量推送到 Redis
4. 从 Redis 拉取最新值更新内存缓存

## 6. 模式匹配路由

**文件**：`router_utils/pattern_match_deployments.py`

`PatternMatchRouter` 支持通配符和正则模型名匹配：

```python
# 配置示例
model_list:
  - model_name: "openai/*"        # 匹配所有 openai/ 前缀
    litellm_params:
      model: "openai/*"
  - model_name: "anthropic/claude-*"  # 匹配 claude 系列
    litellm_params:
      model: "anthropic/claude-*"
```

- 将通配符模式转换为正则表达式
- 按模式特异性排序（更具体的模式优先匹配）
- 支持团队级别的模式路由器 (`team_pattern_routers`)

## 7. 预调用检查系统

通过 `optional_pre_call_checks` 配置启用，在路由决策前进一步过滤部署：

| 检查 | 功能 |
|------|------|
| `deployment_affinity` | API key 亲和性，将同一 key 的请求路由到同一部署（利用隐式 prompt 缓存） |
| `prompt_caching` | 将包含大量 prompt 的请求路由到之前处理过相同 prompt 的部署 |
| `responses_api_deployment_check` | Responses API 连续性，根据 `previous_response_id` 路由 |
| `model_rate_limit_check` | 模型级别速率限制检查 |
| `router_budget_limiting` | 提供商预算限制 |

## 8. 配置示例

```yaml
# proxy_config.yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-deployment-1
      api_base: https://xxx.openai.azure.com/
      api_key: sk-xxx
    model_info:
      id: "deployment-1"

  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-deployment-2
      api_base: https://yyy.openai.azure.com/
      api_key: sk-yyy
    model_info:
      id: "deployment-2"

  - model_name: claude-3
    litellm_params:
      model: anthropic/claude-3-sonnet
      api_key: sk-zzz

router_settings:
  routing_strategy: "latency-based-routing"  # 路由策略
  num_retries: 3                             # 重试次数
  timeout: 30                                # 超时时间（秒）
  allowed_fails: 3                           # 冷却前允许的失败次数
  cooldown_time: 60                          # 冷却时间（秒）
  enable_pre_call_checks: true               # 启用预调用检查
  enable_tag_filtering: true                 # 启用标签过滤
  retry_policy:                              # 自定义重试策略
    RateLimitErrorRetries: 5
    TimeoutErrorRetries: 3
  fallbacks:                                 # 模型组回退
    - gpt-4: [claude-3]
  context_window_fallbacks:                  # 上下文窗口回退
    - gpt-4: [gpt-4-32k]
  provider_budget_config:                    # 提供商预算
    openai:
      budget_limit: 100
      time_period: 1d
```

## 9. 策略选择指南

### 9.1 策略对比总览

| 策略 | 类别 | 延迟开销 | Redis 依赖 | 多实例支持 | 冷启动表现 | 配置复杂度 | 主要优势 | 主要劣势 | 典型场景 |
|------|------|----------|-----------|-----------|-----------|-----------|----------|----------|---------|
| simple-shuffle | 主策略 | 极低（~0ms） | 无 | 天然支持 | 良好（直接随机）| 低（仅 weight）| 简单稳定、零状态 | 不感知实时负载 | 通用默认、多实例首选 |
| least-busy | 主策略 | 低 | 可选 | 需要 Redis | 退化为随机 | 低 | 反映真实并发占用 | 多实例不接 Redis 不准 | 长短请求混合 |
| latency-based | 主策略 | 低 | 可选 | 需要 Redis | 退化为随机 | 中（需 buffer 调优）| 优化用户感知延迟 | 需持续样本积累 | C 端在线 / 流式 chat |
| usage-based v1 | 主策略 | 中 | 可选 | 不精确 | 退化为随机 | 中（需配 TPM/RPM）| 显式遵守配额 | 同步 Redis 阻塞热路径 | 单实例、有明确配额 |
| usage-based v2 | 主策略 | 中~高 | 推荐 | 精确 | 退化为随机 | 中~高 | 多实例精确、支持预检查 | Redis 网络延迟 | 多实例 + 严格配额 |
| cost-based | 主策略 | 低 | 无 | 天然支持 | 直接生效（成本表静态）| 低 | 直接降低账单 | 不看延迟/质量 | 离线批处理、价位差异大 |
| auto-router | 主策略 | 高（嵌入调用）| 无 | 天然支持 | 立即可用 | 高（utterances 配置）| 按语义智能分流 | 引入嵌入延迟与成本 | 多专长模型矩阵 |
| complexity-router | 主策略 | 极低（<1ms 本地）| 无 | 天然支持 | 立即可用 | 中（规则调参）| 本地零成本分级 | 启发式规则有局限 | 简单+复杂混合负载 |
| custom | 主策略 | 取决于实现 | 取决于实现 | 取决于实现 | 取决于实现 | 高（需编码）| 完全可控 | 自行保障稳定性 | 业务自定义维度路由 |
| tag-based | 过滤器 | 极低 | 无 | 天然支持 | 立即可用 | 中（标签维护）| 多租户隔离 | 需正确维护标签 | 多团队 / 多环境共享 |
| budget-limiter | 过滤器 | 低 | 推荐 | 精确（带 Redis）| 立即可用 | 中 | 硬性预算护栏 | 仅 Provider 粒度 | 防止 Provider 烧钱 |

### 9.2 选择决策树

```
需要智能路由吗？
├── 否 → simple-shuffle（加权）
└── 是
    ├── 关注什么？
    │   ├── 响应速度 → latency-based-routing
    │   ├── 成本控制 → cost-based-routing + provider-budget
    │   ├── 负载均衡 → least-busy 或 usage-based
    │   └── 内容分流 → complexity-router 或 auto-router
    │
    └── 多实例部署？
        ├── 否 → 任何策略均可
        └── 是 → 需要 Redis，推荐 usage-based-v2 或 simple-shuffle
```

### 9.3 组合推荐

| 场景 | 推荐组合 |
|------|----------|
| 小规模、单实例 | `simple-shuffle` + cooldown + fallbacks |
| 多团队共享 | `simple-shuffle` + tag-filtering + provider-budget |
| 延迟敏感 + 多实例 | `latency-based` + prompt-caching-affinity + Redis |
| 成本优化 | `cost-based` + provider-budget + context-window-fallbacks |
| 混合负载（简单+复杂） | `complexity-router`（自动分流到不同价位模型） |
| 高可用生产 | `simple-shuffle` + cooldown + retry + fallbacks + health-check |

### 9.4 反模式（不推荐组合）

| 反模式 | 问题 |
|--------|------|
| 多实例 + `usage-based v1`（无 Redis）| 配额计数仅本地准确，跨实例可能整体超限触发 429 |
| `latency-based` + 部署数 < 3 | 样本不足，统计意义弱，不如 simple-shuffle |
| `cost-based` 用于在线 C 端 | 全部流量打向最便宜部署，延迟/可用性失控 |
| `auto-router` 放在路由热路径而无缓存 | 嵌入调用延迟叠加，TP99 显著退化 |
| 仅 `tag-filtering` 无主策略兜底 | 标签未匹配时行为不确定，可能落到默认部署 |
| 自定义策略中阻塞 I/O | 阻塞事件循环，整 Router 吞吐下降 |

## 10. 关键设计决策

1. **策略-过滤器分离**：路由策略（shuffle/latency/cost）和过滤器（tag/budget/affinity）正交组合，过滤器先运行缩小候选集，策略再从候选集中选择。

2. **回调驱动的指标收集**：所有路由策略通过 `CustomLogger` 回调收集指标，与请求处理热路径解耦。

3. **内存优先 + Redis 同步**：热路径只操作内存缓存（保证 <100ms），通过定期批量同步保持多实例一致性。

4. **渐进式降级**：健康检查 → 冷却过滤 → 预调用检查 → 路由策略 → 回退，每层都有独立的降级逻辑。

5. **客户端不关闭**：HTTP/SDK 客户端在缓存淘汰时不关闭（避免影响在途请求），仅在进程退出时统一清理。

6. **冷启动友好**：所有基于统计的策略（latency/usage/cost）在无历史数据时均回退到随机选择，避免冷启动死锁。

7. **可观测性内建**：冷却事件、回退事件、路由决策均可通过 Prometheus 回调或日志系统追踪，便于生产环境排障。

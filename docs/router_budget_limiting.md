# RouterBudgetLimiting — 路由级预算管控

## 概述

`RouterBudgetLimiting` 是 LiteLLM Router 内置的预算管控机制，作为路由过滤器运行。当某个 Provider 或 Deployment 的花费超过预算上限时，自动将其从路由候选中移除，请求会 fallback 到其他可用的 deployment。

核心特性：
- 作为路由过滤器工作，可与任何路由策略（`simple-shuffle`、`least-busy`、`latency-based-routing` 等）组合使用
- 支持三个预算层级：**Provider**、**Deployment**、**Tag**
- 花费通过 DualCache（内存 + Redis）跟踪，支持多实例部署
- 预算窗口过期后自动重置
- 支持 Prometheus 监控剩余预算

## 架构

```
                        ┌─────────────────────────────────┐
                        │         Router 路由请求          │
                        └──────────────┬──────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────┐
                        │  async_filter_deployments()      │
                        │  (RouterBudgetLimiting)          │
                        │                                  │
                        │  1. 检查 Provider 预算            │
                        │  2. 检查 Deployment 预算          │
                        │  3. 检查 Tag 预算                 │
                        │                                  │
                        │  → 过滤掉超预算的 deployment      │
                        └──────────────┬──────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────┐
                        │  路由策略选择最终 deployment       │
                        │  (simple-shuffle / latency 等)   │
                        └──────────────┬──────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────┐
                        │  async_log_success_event()       │
                        │  请求完成后累加花费到 DualCache     │
                        └─────────────────────────────────┘
```

## 关键源码文件

| 文件 | 职责 |
|------|------|
| `litellm/router_strategy/budget_limiter.py` | `RouterBudgetLimiting` 核心实现：过滤、花费跟踪、缓存同步 |
| `litellm/router.py` | Router 初始化和生命周期管理，动态注册/移除 deployment 预算 |
| `litellm/proxy/hooks/model_max_budget_limiter.py` | Virtual Key + Model 级别预算（Proxy 专用） |
| `litellm/types/utils.py` | `BudgetConfig`、`GenericBudgetConfigType` 类型定义 |
| `litellm/litellm_core_utils/duration_parser.py` | 时间周期解析（`1d`、`7d`、`1mo` 等） |
| `litellm/proxy/spend_tracking/spend_management_endpoints.py` | `GET /provider/budgets` API 端点 |

---

## 预算层级详解

### 1. Provider 级别预算

按 LLM 供应商（openai、anthropic、azure 等）设定预算。同一 provider 下的所有 deployment 共享预算额度。

**运行时行为**：每次路由时，动态获取 deployment 的 provider → 查 `provider_budget_config` → 比对花费。即使 deployment 是后续从数据库动态加载的，只要 provider 匹配就会被管控。

**配置示例**：

```yaml
router_settings:
  provider_budget_config:
    openai:
      budget_limit: 100      # $100
      time_period: 1d         # 每天重置
    anthropic:
      budget_limit: 500
      time_period: 7d         # 每7天重置
    azure:
      budget_limit: 200
      time_period: 30d        # 每30天重置
```

**Cache Key 格式**：`provider_spend:{provider}:{budget_duration}`
例如：`provider_spend:openai:1d`

### 2. Deployment 级别预算

按具体 deployment（model_id 粒度）设定独立预算。适用于需要对同一 provider 下的不同 deployment 分别管控的场景。

**配置示例**：

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
      max_budget: 50          # $50
      budget_duration: 1d     # 每天重置
    model_info:
      id: "gpt4-deployment-1"

  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY_2
      max_budget: 100         # $100
      budget_duration: 30d    # 每30天重置
    model_info:
      id: "gpt4-deployment-2"
```

**Cache Key 格式**：`deployment_spend:{model_id}:{budget_duration}`
例如：`deployment_spend:gpt4-deployment-1:1d`

### 3. Tag 级别预算（Enterprise 功能）

按请求 Tag 设定预算，适用于按业务线/项目维度管控花费。

**配置示例**：

```python
import litellm

litellm.tag_budget_config = {
    "project-alpha": {
        "budget_limit": 200,
        "time_period": "1d",
    },
    "project-beta": {
        "budget_limit": 500,
        "time_period": "7d",
    },
}
```

**Cache Key 格式**：`tag_spend:{tag}:{budget_duration}`

### 4. Virtual Key + Model 级别预算（Proxy 功能）

在 API Key 上设置每个模型的预算上限，由 `_PROXY_VirtualKeyModelMaxBudgetLimiter` 处理。超预算时抛出 `BudgetExceededError`（HTTP 429）。

**配置方式**：通过 Proxy API 在 key 上设置 `model_max_budget`：

```json
{
  "model_max_budget": {
    "gpt-4": {
      "budget_limit": 50,
      "time_period": "1d"
    }
  }
}
```

---

## 自动开启条件

`RouterBudgetLimiting` 支持两种初始化方式：

### 方式一：Router 初始化时自动启用

满足以下任一条件，Router 在构造时即创建 `RouterBudgetLimiting`：

| 条件 | 来源 |
|------|------|
| `provider_budget_config` 不为 None | YAML `router_settings.provider_budget_config` |
| `litellm.tag_budget_config` 不为 None | Python 代码设置 |
| model_list 中任一 deployment 设置了 `max_budget` 或 `budget_duration` | YAML `model_list[].litellm_params` |

判断逻辑（`budget_limiter.py` `should_init_router_budget_limiter()`）：

```python
@staticmethod
def should_init_router_budget_limiter(provider_budget_config, model_list):
    if provider_budget_config is not None:
        return True
    if litellm.tag_budget_config is not None:
        return True
    if model_list:
        for _model in model_list:
            _litellm_params = _model.get("litellm_params", {})
            if _litellm_params.get("max_budget") or _litellm_params.get("budget_duration") is not None:
                return True
    return False
```

### 方式二：懒初始化（Lazy Init）

当 Router 初始化时未满足上述条件（例如 YAML 中无模型、无 `provider_budget_config`），但后续通过数据库动态加载的 deployment 包含 `max_budget` + `budget_duration` 时，Router 会在 `add_deployment()` 中**自动懒初始化** `RouterBudgetLimiting`（`router.py` `_lazy_init_router_budget_limiter()`）。

这意味着：**即使 YAML 中没有任何预算相关配置，只要数据库模型设置了 `max_budget`/`budget_duration`，`RouterBudgetLimiting` 也会被自动创建并生效。**

---

## 数据库模型支持

### 问题背景

`RouterBudgetLimiting` 的 `_init_deployment_budgets()` 仅在 `__init__()` 时执行一次，读取初始化时的 `model_list`。后续从数据库动态加载的模型通过 `Router.upsert_deployment()` → `add_deployment()` 添加到 `model_list`，但不会自动注册到 `deployment_budget_config` 中。

### 解决方案

通过在 Router 的 `add_deployment()` / `delete_deployment()` / `upsert_deployment()` 中增加动态注册/移除逻辑解决：

**新增方法**（`budget_limiter.py`）：

```python
def register_deployment_budget(self, model_id, max_budget, budget_duration):
    """动态注册 deployment 预算到 deployment_budget_config"""

def remove_deployment_budget(self, model_id):
    """从 deployment_budget_config 移除 deployment 预算"""
```

**调用链路**：

```
DB 模型加载
  → proxy_server._add_deployment()
    → llm_router.upsert_deployment()
      → router.add_deployment()
        → router._register_deployment_budget()
          → router_budget_logger 为 None?
            → 是: _lazy_init_router_budget_limiter() 创建实例并注册到 callbacks
            → 否: 直接使用已有实例
          → router_budget_logger.register_deployment_budget()
            → deployment_budget_config[model_id] = config  ✅
```

```
模型删除
  → router.delete_deployment()
    → router_budget_logger.remove_deployment_budget()  ← 新增
      → deployment_budget_config.pop(model_id)  ✅
```

```
模型更新（upsert 替换）
  → router.upsert_deployment()
    → remove old: router_budget_logger.remove_deployment_budget()  ← 新增
    → add new:   router.add_deployment()  → _register_deployment_budget()
```

### 配置方式

**YAML 配置（可选）**

如果需要 Provider 级别预算，在 YAML 中配置 `provider_budget_config`。如果只需要 Deployment 级别预算，YAML 中无需任何预算相关配置，`RouterBudgetLimiting` 会在 DB 模型加载时自动懒初始化。

```yaml
# config.yaml — 可选的 provider 级别预算
router_settings:
  provider_budget_config:       # 可选，不配也不影响 deployment 级别预算
    openai:
      budget_limit: 1000
      time_period: 30d
```

**数据库模型设置预算**

通过 Proxy API 创建/更新模型时，在 `litellm_params` 中传入 `max_budget` 和 `budget_duration`：

```bash
curl -X POST http://localhost:4000/model/new \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-master-key" \
  -d '{
    "model_name": "gpt-4",
    "litellm_params": {
      "model": "openai/gpt-4",
      "api_key": "sk-...",
      "max_budget": 50,
      "budget_duration": "1d"
    }
  }'
```

模型加载到 Router 后，`max_budget` 和 `budget_duration` 会自动注册到 `RouterBudgetLimiting.deployment_budget_config`。

### 各层级对数据库模型的支持矩阵

| 预算层级 | YAML 模型 | DB 模型（首次初始化） | DB 模型（动态加载） |
|----------|----------|---------------------|-------------------|
| Provider | ✅ | ✅ | ✅ 运行时动态匹配 |
| Deployment | ✅ | ✅ | ✅ 动态注册 |
| Tag | ✅ | ✅ | ✅ 运行时动态匹配 |
| Virtual Key + Model | ✅ | ✅ | ✅ 按 key 级别检查 |

---

## 时间周期格式

由 `litellm/litellm_core_utils/duration_parser.py` 解析，支持以下格式：

| 格式 | 含义 | 示例 |
|------|------|------|
| `<n>s` | 秒 | `30s` = 30 秒 |
| `<n>m` | 分钟 | `5m` = 5 分钟 |
| `<n>h` | 小时 | `2h` = 2 小时 |
| `<n>d` | 天 | `1d` = 1 天, `30d` = 30 天 |
| `<n>w` | 周 | `1w` = 1 周 |
| `<n>mo` | 月 | `1mo` = 1 个月 |

---

## 超预算行为

当所有候选 deployment 都超预算时，`async_filter_deployments()` 抛出 `ValueError`：

```
No deployments available - crossed budget:
Exceeded budget for provider openai: 100.5 >= 100
Exceeded budget for deployment model_name: gpt-4, ..., model_id: xxx: 50.2 >= 50
```

客户端收到此错误后应进行 fallback 处理或等待预算窗口重置。

---

## 多实例部署（Redis 同步）

在多实例 Proxy 部署环境下，各实例的花费数据通过 Redis 同步：

1. **写入路径**：请求完成后，花费先增量写入内存缓存（即时生效），同时入队 `redis_increment_operation_queue`
2. **批量推送**：`periodic_sync_in_memory_spend_with_redis()` 每秒执行一次，将队列中的增量通过 Redis Pipeline 批量推送
3. **读取同步**：同一周期内从 Redis 批量拉取所有 provider/deployment 的最新花费，更新本地内存缓存

```
Instance A ──increment──→ 内存缓存 A ──pipeline──→ Redis ──batch_get──→ 内存缓存 B ← Instance B
```

**配置 Redis**：

```yaml
router_settings:
  redis_host: localhost
  redis_port: 6379
  redis_password: your_password
  provider_budget_config:
    openai:
      budget_limit: 100
      time_period: 1d
```

同步间隔默认 1 秒（`DEFAULT_REDIS_SYNC_INTERVAL = 1`）。

---

## 监控

### Prometheus 指标

当配置了 Prometheus callback 时，`RouterBudgetLimiting` 自动上报 provider 剩余预算指标：

```python
_track_provider_remaining_budget_prometheus(provider, spend, budget_limit)
```

### API 端点

**GET /provider/budgets** — 查看当前 provider 预算状态：

```bash
curl -X GET http://localhost:4000/provider/budgets \
  -H "Authorization: Bearer sk-master-key"
```

响应示例：

```json
{
  "providers": {
    "openai": {
      "budget_limit": 100.0,
      "time_period": "1d",
      "spend": 42.5,
      "budget_reset_at": "2026-03-15T00:00:00+00:00"
    },
    "anthropic": {
      "budget_limit": 500.0,
      "time_period": "7d",
      "spend": 128.3,
      "budget_reset_at": "2026-03-21T00:00:00+00:00"
    }
  }
}
```

---

## 完整配置示例

### 场景：YAML 开启 + 数据库模型独立预算

```yaml
# config.yaml

model_list:
  # YAML 中定义的模型（可选，也可全部来自 DB）
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY
      max_budget: 20
      budget_duration: 1d

router_settings:
  # Provider 级别预算 — 对所有模型（含 DB）生效
  provider_budget_config:
    openai:
      budget_limit: 200
      time_period: 1d
    anthropic:
      budget_limit: 300
      time_period: 7d

  # 可选：配合其他路由策略
  routing_strategy: least-busy

general_settings:
  master_key: sk-master-key
  database_url: os.environ/DATABASE_URL
```

数据库中添加模型（通过 API）：

```bash
# 添加一个有独立预算的 DB 模型
curl -X POST http://localhost:4000/model/new \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-master-key" \
  -d '{
    "model_name": "gpt-4",
    "litellm_params": {
      "model": "openai/gpt-4",
      "api_key": "sk-...",
      "max_budget": 100,
      "budget_duration": "1d"
    }
  }'

# 添加一个没有独立预算的 DB 模型（仍受 provider 级别预算管控）
curl -X POST http://localhost:4000/model/new \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-master-key" \
  -d '{
    "model_name": "claude-3-opus",
    "litellm_params": {
      "model": "anthropic/claude-3-opus-20240229",
      "api_key": "sk-ant-..."
    }
  }'
```

效果：
- `gpt-3.5-turbo`（YAML）：deployment 预算 $20/天 + provider 预算 $200/天（openai 共享）
- `gpt-4`（DB）：deployment 预算 $100/天 + provider 预算 $200/天（openai 共享）
- `claude-3-opus`（DB）：无 deployment 预算，但受 provider 预算 $300/7天（anthropic 共享）管控

### 场景：Python SDK 直接使用

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "sk-...",
                "max_budget": 50,
                "budget_duration": "1d",
            },
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "azure/gpt-4",
                "api_key": "...",
                "api_base": "...",
                "max_budget": 80,
                "budget_duration": "1d",
            },
        },
    ],
    provider_budget_config={
        "openai": {"budget_limit": 100, "time_period": "1d"},
        "azure": {"budget_limit": 200, "time_period": "1d"},
    },
)

# 动态添加 deployment（模拟 DB 加载）
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

router.add_deployment(
    Deployment(
        model_name="gpt-4",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4",
            api_key="sk-...",
            max_budget=30,
            budget_duration="1d",
        ),
        model_info=ModelInfo(id="gpt4-dynamic-1"),
    )
)
# gpt4-dynamic-1 的 deployment 预算自动注册到 RouterBudgetLimiting
```

---

## 预算检查优先级

在 `_filter_out_deployments_above_budget()` 中，对每个 deployment 按以下顺序检查，任一层级超预算即被排除：

```
1. Provider 预算检查
   └── provider_spend:{provider}:{duration} >= provider max_budget ?
       └── 是 → 排除，跳到下一个 deployment

2. Deployment 预算检查
   └── deployment_spend:{model_id}:{duration} >= deployment max_budget ?
       └── 是 → 排除，跳到下一个 deployment

3. Tag 预算检查（如有 request tag）
   └── tag_spend:{tag}:{duration} >= tag max_budget ?
       └── 是 → 排除，跳到下一个 deployment

4. 全部通过 → 加入候选列表
```

---

## 注意事项

1. **预算单位**：`max_budget` / `budget_limit` 的单位是美元（$），基于 LiteLLM 的 response_cost 计算
2. **时间窗口**：预算在时间窗口到期后自动重置（通过 cache TTL 实现），无需手动清理
3. **即时生效**：花费增量即时写入内存缓存，同一实例内的后续请求立即感知预算变化
4. **多实例延迟**：跨实例的花费同步有最多 1 秒延迟（`DEFAULT_REDIS_SYNC_INTERVAL`），极端并发下可能短暂超支
5. **无持久化**：花费数据存储在缓存中（内存/Redis），进程重启后重置。如需持久化预算跟踪，需结合数据库 spend 记录
6. **Tag 预算**：Tag 级别预算是 Enterprise 功能，需要 premium_user 授权

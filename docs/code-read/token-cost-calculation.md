# LiteLLM Token 费用计算流程详解

## 概述

本文档详细分析 LiteLLM 中 **Token 费用计算** 的完整流程，从 LLM 响应返回到最终费用数字的每个环节，包括核心函数调用链、数据结构、定价策略和自定义扩展点。

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM Provider API                              │
│              (OpenAI / Anthropic / ...)                        │
│                  │                                        │
│                  ▼                                        │
│            ModelResponse                                     │
│          (包含 usage + _hidden_params)                      │
│                  │                                        │
│                  ▼                                        │
│       update_response_metadata()                                │
│     文件: litellm_core_utils/llm_response_utils/response_metadata.py      │
│     功能: 设置 hidden_params, 计算并存储 response_cost           │
│                  │                                        │
│                  ▼                                        │
│       _response_cost_calculator()                               │
│     文件: litellm/litellm_core_utils/litellm_logging.py:1429   │
│     功能: 从 result + logging_obj 一致地计算费用                   │
│                  │                                        │
│                  ▼                                        │
│       response_cost_calculator()                                 │
│     文件: litellm/cost_calculator.py:1645                     │
│     功能: 核心费用计算入口, 调用 completion_cost()             │
│                  │                                        │
│                  ▼                                        │
│       completion_cost()                                         │
│     文件: litellm/cost_calculator.py:1011                     │
│     功能: 提取 token 数量, 选择定价策略, 返回总费用           │
│                  │                                        │
│                  ▼                                        │
│       generic_cost_per_token()                                  │
│     文件: litellm/llm_core_utils/llm_cost_calc/utils.py:620    │
│     功能: 通用按 token 计算函数, 处理各种 token 类型         │
│                  │                                        │
│                  ▼                                        │
│       calculate_cost_component()                                │
│     文件: litellm/llm_core_utils/llm_cost_calc/utils.py:324    │
│     功能: 单项费用计算 (input/output/reasoning/image/audio)      │
│                  │                                        │
│                  ▼                                        │
│       _get_token_base_cost()                                   │
│     文件: litellm/llm/llm_core_utils/llm_cost_calc/utils.py:695    │
│     功能: 从 model_info 获取基础单价 (支持 service tier)        │
│                  │                                        │
│                  ▼                                        │
│       get_model_info()                                          │
│     文件: litellm/utils.py                                       │
│     功能: 从 model_prices_and_context_window.json 获取模型定价信息         │
│                  │                                        │
│                  ▼                                        │
│       model_prices_and_context_window.json                             │
│     文件: litellm/model_prices_and_context_window.json                 │
│     功能: 所有模型的定价数据 (input/output cost per token)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 完整调用链

### 2.1 入口：Proxy 后处理阶段

**触发位置**: `proxy/common_request_processing.py` 的 `base_process_llm_request()` 方法返回后

```python
# 文件: litellm/proxy/common_request_processing.py:1022
# 在路由和护栏钩子并行执行后获取响应

response = responses[1]  # 取路由任务结果

hidden_params = getattr(response, "_hidden_params", {})
model_id = self._get_model_id_from_response(hidden_params, self.data)
cache_key, api_base, response_cost = (
    hidden_params.get("cache_key", None) or "",
    hidden_params.get("api_base", None) or "",
    hidden_params.get("response_cost", None) or "",
)
```

### 2.2 费用归因：`update_response_metadata()`

**文件**: `litellm/litellm_core_utils/llm_response_utils/response_metadata.py`

**核心功能**: 将费用信息写入 response 的 `_hidden_params`

```python
# 文件: litellm/litellm_core_utils/llm_response_utils/response_metadata.py:38-56

class ResponseMetadata:
    def set_hidden_params(self, logging_obj, model, kwargs) -> None:
        """Set hidden parameters on the response"""
        new_params = {
            "litellm_call_id": ...,
            "api_base": get_api_base(model=model),        # API Base URL
            "model_id": model_info.get("id"),                # 部署 ID
            "response_cost": logging_obj._response_cost_calculator(   # ⭐ 核心费用计算
                result=self.result,
                litellm_model_name=model,
                router_model_id=router_model_id,
            ),
            "additional_headers": ...,
            "litellm_model_name": model,
        }
        self._update_hidden_params(new_params)
```

**设置的隐藏参数**:

| 参数 | 说明 | 来源 |
|------|------|------|
| `response_cost` | **本次请求的费用 (USD)** | `logging_obj._response_cost_calculator()` 计算 |
| `litellm_overhead_time_ms` | LiteLLM 开销时间 (ms) | `total_response_ms - llm_api_duration_ms` |
| `callback_duration_ms` | 回调处理时间 (ms) | `logging_obj.callback_duration_ms` |
| `litellm_call_id` | 本次调用唯一标识 | `logging_obj.litellm_call_id` |
| `model_id` | Router 中的部署 ID | `hidden_params.model_id` 或 `router_model_id` |

---

## 3. 核心费用计算器

### 3.1 `_response_cost_calculator()` — 日志对象方法

**文件**: `litellm/litellm_core_utils/litellm_logging.py:1429-1562`

这是 Proxy 层面的费用计算入口，确保 header 和日志使用**一致**的费用值。

```python
def _response_cost_calculator(self, result, cache_hit=None,
                            litellm_model_name=None, router_model_id=None) -> Optional[float]:

    # 1. 缓存命中 → 费用为 0
    if cache_hit is True:
        return 0.0

    # 2. 优先使用已计算的费用 (Provider 可能自行计算)
    if hasattr(result, "_hidden_params"):
        hp = result._hidden_params
        if "response_cost" in hp and hp["response_cost"] is not None:
            return hp["response_cost"]

    # 3. 否则调用全局 response_cost_calculator()
    response_cost = litellm.response_cost_calculator(
        response_object=result,
        model=litellm_model_name or self.model,
        cache_hit=cache_hit,
        custom_llm_provider=self.model_call_details.get("custom_llm_provider"),
        base_model=_get_base_model_from_metadata(...),
        call_type=self.call_type,
        optional_params=self.optional_params,
        custom_pricing=use_custom_pricing_for_model(...),
        prompt=self.model_call_details.get("input"),
        standard_built_in_tools_params=self.standard_built_in_tools_params,
        router_model_id=router_model_id,
        litellm_logging_obj=self,
        service_tier=self.optional_params.get("service_tier"),
    )
    return response_cost
```

### 3.2 `response_cost_calculator()` — 全局函数

**文件**: `litellm/cost_calculator.py:1645-1730`

**功能**: 统一的费用计算入口，根据 call type 分发到不同计算逻辑。

```python
def response_cost_calculator(response_object, model, custom_llm_provider, call_type,
                          optional_params, cache_hit, base_model, ...) -> float:

    # 1. 缓存命中 → 免费
    if cache_hit is True:
        return 0.0

    # 2. BaseModel 对象 → 提取 hidden_params 中的预计算费用
    if isinstance(response_object, BaseModel):
        if hasattr(response_object, "_hidden_params"):
            hp = response_object._hidden_params
            if "response_cost" in hp and hp["response_cost"] is not None:
                return hp["response_cost"]

    # 3. 调用 completion_cost() 进行实际计算
    return completion_cost(
        completion_response=response_object,
        model=model,
        call_type=call_type,
        custom_llm_provider=custom_llm_provider,
        optional_params=optional_params,
        custom_pricing=custom_pricing,
        base_model=base_model,
        prompt=prompt,
        ...
    )
```

### 3.3 `completion_cost()` — 主计算函数

**文件**: `litellm/cost_calculator.py:1011-1066`

**功能**: 从响应中提取 token 数量，查找模型定价，计算总费用。

#### 3.3.1 Usage 信息提取

```python
def completion_cost(completion_response=None, model=None, ...):

    # 从 ModelResponse 提取 usage
    usage_obj = getattr(completion_response, "usage", None)
    if usage_obj is None:
        usage_obj = getattr(completion_response, "usage", {})  # dict fallback

    # 解析各类 token
    prompt_tokens = usage_obj.get("prompt_tokens", 0)
    completion_tokens = usage_obj.get("completion_tokens", 0)
    cache_creation_input_tokens = usage_obj.get(
        "cache_creation_input_tokens", 0
    )
    cache_read_input_tokens = usage_obj.get("cache_read_input_tokens", 0)

    # 解析详细 token 分解 (Anthropic/OpenAI 格式)
    if "prompt_tokens_details" in usage_obj:
        prompt_tokens_details = _parse_prompt_tokens_details(usage)
        # → text_tokens, cache_hit_tokens, audio_tokens,
        #   cache_creation_tokens, image_tokens, character_count...

    if "completion_tokens_details" in usage_obj:
        completion_tokens_details = _parse_completion_tokens_details(usage)
        # → text_tokens, reasoning_tokens, audio_tokens, image_tokens

    # 从 hidden_params 获取 provider 和 region
    hidden_params = getattr(completion_response, "_hidden_params", {})
    custom_llm_provider = hidden_params.get(
        "custom_llm_provider", custom_llm_provider or None
    )
```

#### 3.3.2 模型名称选择策略 (`_select_model_name_for_cost_calc`)

```python
def _select_model_name_for_cost_calc(model, completion_response,
                                    custom_llm_provider, custom_pricing,
                                    base_model, router_model_id):
    potential_model_names = [
        selected_model,                     # 显式传入的 model
        _get_response_model(completion_response),  # 响应中的实际模型
    ]

    for idx, model in enumerate(potential_model_names):
        try:
            # 尝试计算费用
            cost = cost_per_token(...)
            if cost > 0:
                return model  # 找到第一个有定价的模型名
        except Exception:
            continue
    raise Exception(f"Model not found in cost map: {model}")
```

#### 3.3.3 定价查询优先级

```
模型名称匹配顺序 (cost_per_token 查找):
1. "provider/model" 格式 → 如 "openai/gpt-4o" ✅
2. 纯 model 名 → 如 "gpt-4o" ✅
3. 去掉 provider 前缀后 → 如 "gpt-4o" ✅
4. Bedrock 特殊格式 → 如 "bedrock/anthropic.claude-..." ✅
```

---

## 4. 通用按 Token 计算函数

### 4.1 `generic_cost_per_token()` — 核心计算逻辑

**文件**: `litellm/llm_core_utils/llm_cost_calc/utils.py:620-781`

**公式**:

```
总费用 = 输入费用 + 输出费用

输入费用 = text_tokens × input_cost_per_token
         + cache_hit_tokens × cache_read_cost
         + cache_creation_tokens × cache_creation_cost
         + audio_tokens × input_cost_per_audio_token
         + image_tokens × input_cost_per_image_token
         + character_count × input_cost_per_character
         + video_length_seconds × input_cost_per_video_second
         + image_count × input_cost_per_image

输出费用 = text_tokens × output_cost_per_token
         + reasoning_tokens × output_cost_per_reasoning_token
         + audio_tokens × output_cost_per_audio_token
         + image_tokens × output_cost_per_image_token
```

### 4.2 `_get_token_base_cost()` — 获取基础单价

**文件**: `litellm/llm/llm_core_utils/llm_cost_calc/utils.py:695-697`

```python
def _get_token_base_cost(model_info, usage, service_tier=None):
    """
    返回 (prompt_base_cost, completion_base_cost, cache_creation_cost,
           cache_read_cost, cache_creation_cost_above_1hr)
    """
    # 支持 OpenAI Service Tier 定价 (如 batch/tier)
    return _get_cost_per_unit(model_info, "input_cost_per_token") * 1.0, \
           _get_cost_per_unit(model_info, "output_cost_per_token") * 1.0, \
           ..., service_tier
```

### 4.3 `_get_cost_per_unit()` — 单价查询

**文件**: `litellm/llm/llm_core_utils/llm_cost_calc/utils.py:349-388`

```python
def _get_cost_per_unit(model_info, cost_key, default_value=0.0):
    cost_per_unit = model_info.get(cost_key)

    # 支持字符串类型的定价 (如 "3e-7")
    if isinstance(cost_per_unit, str):
        try:
            return float(cost_per_unit)
        except ValueError:
            pass

    # Service Tier 后缀匹配 (如 "_default", "_batch")
    for service_tier in ServiceTier:
        suffix = f"_{service_tier.value}"
        if suffix in cost_key:
            base_key = cost_key.replace(suffix, "")
            fallback_cost = model_info.get(base_key)
            if isinstance(fallback_cost, float): return fallback_cost

    return default_value
```

---

## 5. 各 Call Type 的特殊计算逻辑

### 5.1 Chat Completion（默认）

直接使用 `generic_cost_per_token()` 计算：

```python
# 输入: prompt_tokens × input_cost_per_token
# 输出: completion_tokens × output_cost_per_token
prompt_cost, completion_cost = generic_cost_per_token(
    model=model, usage=usage, custom_llm_provider="openai"
)
return prompt_cost + completion_cost
```

### 5.2 Embedding

```python
# 直接使用 OpenAI 兼容的 cost_per_token
prompt_cost, completion_cost = openai_cost_per_token(
    model=model, usage=usage, service_tier=service_tier
)
return prompt_cost + completion_cost
```

### 5.3 Image Generation

**两种路径**：

**路径 A — 有 usage metadata**（推荐）：
```python
# 先尝试从 ImageResponse.usage 获取费用
cost = calculate_image_response_cost_from_usage(
    model=model, image_response=response, custom_llm_provider="openai"
)
if cost is not None:
    return cost  # 成功返回
```

**路径 B — 按 image 数量计费**：
```python
# 使用 flat pricing (每张图片固定价格)
image_cost = default_image_cost_calculator(
    model=model, completion_response=response, n=n, size=size, quality=quality
)
```

### 5.4 Audio Transcription

```python
# 按时长计费 (TTS)
if transcription_file_duration > 0:
    return openai_cost_per_second(
        model=model, duration=audio_transcription_file_duration
    )

# 或按 token 计费
return openai_cost_per_token(model, usage=usage)
```

### 5.5 Rerank

```python
return rerank_cost(
    model=model, custom_llm_provider=provider,
    billed_units=rerank_billed_units
)
```

### 5.6 Search

```python
# 按查询次数计费 (非 per-token)
return search_provider_cost_per_query(
    model=model, custom_llm_provider=provider,
    number_of_queries=number_of_queries,
    optional_params=response._hidden_params
)
```

### 5.7 OCR

```python
return ocr_cost(model=model, custom_llm_provider=provider, response=response)
```

### 5.8 Batch (批量)

```python
return batch_cost_calculator(usage=usage, model=model, ...)
```

---

## 6. 定价数据来源与结构

### 6.1 `model_prices_and_context_window.json` 数据结构

```json
{
  "dashscope/qwen-turbo": {
    "litellm_provider": "dashscope",
    "mode": "chat",
    "max_input_tokens": 129024,
    "max_output_tokens": 16384,
    "input_cost_per_token": 5e-08,
    "output_cost_per_token": 2e-07,
    "output_cost_per_reasoning_token": 5e-07,
    "supports_function_calling": true,
    "supports_reasoning": true,
    "supports_tool_choice": true
  },

  "openai/gpt-4o": {
    "litellm_provider": "openai",
    "mode": "chat",
    "max_input_tokens": 128000,
    "max_output_tokens": 16384,
    "input_cost_per_token": 2.5e-06,
    "output_cost_per_token": 1e-05,
    "tiered_pricing": [
      { "range": [0, 256000], "input_cost_per_token": 5e-08, "output_cost_per_token": 4e-07 },
      { "range": [256000, 1000000], "input_cost_per_token": 2.5e-07, "output_cost_per_token": 2e-06 }
    ]
  }
}
```

### 6.2 Service Tier 定价 (OpenAI)

支持 `default`, `batch`, `tier1`, `tier2`, `tier3` 等级：

```python
# tier3 定价示例 (GPT-4o)
"input_cost_per_token_tier3": 2.5e-06  # 比 default 贵 50%
"output_cost_per_token_tier3": 1e-05
```

### 6.3 Context Caching 定价 (Anthropic)

```python
# Cache Read: 读取缓存时付费
"cache_read_input_token_cost": 0.000125,  # 每 token $0.000125

# Cache Creation: 写入缓存时付费
"cache_creation_input_token_cost": 0.000375,  # 每 token $0.000375
"cache_creation_cost_above_1hr": 0.15  # 超过 1h 的缓存创建费用
```

---

## 7. 自定义定价 (Custom Pricing)

### 7.1 配置方式

**方式1: 环境变量**
```bash
export CUSTOM_PRICING="[{\"model\": \"custom-model\", \"input_cost_per_token\": 0.001, \"output_cost_per_token\": 0.002}]"
```

**方式2: config.yaml**
```yaml
model_list:
  - model_name: my-custom-model
    litellm_params:
      model: my-base-model
      custom_pricing:
        input_cost_per_token: 0.001
        output_cost_per_token: 0.002
```

**方式3: 代码中传入**
```python
response.cost = litellm.completion_cost(
    completion_response=response,
    model="my-model",
    custom_cost_per_token=CostPerToken(
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    ),
)
```

### 7.2 自定义 Cost Calculator

可以实现 `cost_per_token` 函数来完全控制计算逻辑：

```python
from litellm import cost_calculator

@cost_calculator
def my_custom_cost_calculator(
    model: str,
    usage,  # Usage object
    custom_llm_provider: str,
    **kwargs
) -> float:
    """我的自定义费用计算 - 按字符数收费"""
    input_text = kwargs["prompt"] or ""
    output_text = ""

    # 自定义: 每个字符 $0.00001
    return len(input_text.encode('utf-8')) * 0.00001 + len(output_text.encode('utf-8')) * 0.00002

# 注册
litellm.register_custom_cost_calculator(
    "my-custom-model",
    my_custom_cost_calculator
)
```

---

## 8. 费用在 Proxy 中的流转

```
用户请求 → chat_completion()
    ↓
base_process_llm_request()
    ↓
common_processing_pre_call_logic()  ← 预检/限流/预算
    ↓
route_request()  ← 路由到具体 Provider
    ↓
acompletion()  ← SDK 调用 LLM
    ↓
ModelResponse (含 usage + _hidden_params)
    ↓
update_response_metadata()  ← 写入 response_cost 到 hidden_params
    ↓
_response_cost_calculator()  ← 从 hidden_params 读取或重新计算
    ↓
select_data_generator()  ← 选择输出格式
    ↓
HTTP Response
    ├─ Headers:
    │   x-litellm-response-cost: 0.00452
    └─ Body: JSON ModelResponse

异步日志 (async_success_handler):
    → DBSpendUpdateWriter.update_database() → 内存队列/Redis → PostgreSQL
```

---

## 9. 计费数据存储流程

费用计算完成后，数据需要**持久化**到数据库中，用于预算管理、用量统计和审计。这个过程是**异步**的，不会阻塞 HTTP 响应。

### 9.1 整体存储架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        计费数据存储流程                                       │
│                                                                              │
│  async_success_handler()                                                     │
│      ↓                                                                       │
│  _PROXY_track_cost_callback()                                                │
│  文件: litellm/proxy/hooks/proxy_track_cost_callback.py                      │
│      ↓                                                                       │
│  DBSpendUpdateWriter.update_database()                                       │
│  文件: litellm/proxy/db/db_spend_update_writer.py                            │
│      │                                                                       │
│      ├─→ ① _insert_spend_log_to_db()          SpendLogs 明细入队             │
│      │      → prisma_client.spend_log_transactions (内存队列)                 │
│      │                                                                       │
│      └─→ ② _batch_database_updates() (asyncio.create_task)                   │
│            │                                                                 │
│            ├─ _update_user_db()      → SpendUpdateQueue (User)               │
│            ├─ _update_key_db()       → SpendUpdateQueue (Key)                │
│            ├─ _update_team_db()      → SpendUpdateQueue (Team + TeamMember)  │
│            ├─ _update_org_db()       → SpendUpdateQueue (Organization)       │
│            ├─ _update_tag_db()       → SpendUpdateQueue (Tag)                │
│            ├─ _update_agent_db()     → SpendUpdateQueue (Agent)              │
│            │                                                                 │
│            ├─ add_spend_log_transaction_to_daily_user_transaction()           │
│            ├─ add_spend_log_transaction_to_daily_team_transaction()           │
│            ├─ add_spend_log_transaction_to_daily_org_transaction()            │
│            ├─ add_spend_log_transaction_to_daily_end_user_transaction()       │
│            ├─ add_spend_log_transaction_to_daily_agent_transaction()          │
│            └─ add_spend_log_transaction_to_daily_tag_transaction()            │
│                  → DailySpendUpdateQueue (各维度)                             │
│                                                                              │
│  ────────────────── 定时任务 (每分钟) ──────────────────                      │
│                                                                              │
│  update_spend()                                                              │
│  文件: litellm/proxy/utils.py                                                │
│      ↓                                                                       │
│  db_update_spend_transaction_handler()                                        │
│      ├─ 无 Redis: 直接 flush 内存队列 → Prisma batch → PostgreSQL            │
│      └─ 有 Redis: 内存队列 → Redis → Leader Pod 获取锁 → Prisma → PostgreSQL │
│                                                                              │
│  update_spend_logs_job()                                                     │
│      → prisma_client.spend_log_transactions → Prisma batch → SpendLogs 表    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 触发入口：`_PROXY_track_cost_callback`

**文件**: `litellm/proxy/hooks/proxy_track_cost_callback.py`

该回调由 `async_success_handler` 触发，是计费数据写入存储的**入口**。

```python
async def _PROXY_track_cost_callback(self, kwargs, completion_response, start_time, end_time):
    # 1. 从 standard_logging_object 获取 response_cost
    sl_object = kwargs.get("standard_logging_object", None)
    response_cost = sl_object.get("response_cost", None)

    # 2. 缓存命中时费用归零
    if kwargs.get("cache_hit", False) is True:
        response_cost = 0.0

    # 3. 调用 DBSpendUpdateWriter 写入数据库
    await proxy_logging_obj.db_spend_update_writer.update_database(
        token=user_api_key,
        response_cost=response_cost,
        user_id=user_id, team_id=team_id, org_id=org_id,
        end_user_id=end_user_id,
        kwargs=kwargs, completion_response=completion_response,
        start_time=start_time, end_time=end_time,
    )

    # 4. 原子更新内存+Redis计数器 (用于跨 Pod 预算强制)
    await increment_spend_counters(token=user_api_key, team_id=team_id, ...)

    # 5. 更新缓存 (fire-and-forget)
    asyncio.create_task(update_cache(token=user_api_key, ...))
```

### 9.3 数据库更新核心：`update_database()`

**文件**: `litellm/proxy/db/db_spend_update_writer.py`

该方法负责将费用数据拆分到不同的队列中：

```python
async def update_database(self, token, user_id, end_user_id, team_id, org_id,
                          kwargs, completion_response, start_time, end_time, response_cost):

    # Step 1: 构建 SpendLogsPayload
    payload = get_logging_payload(kwargs=kwargs, response_obj=completion_response,
                                  start_time=start_time, end_time=end_time)
    payload["spend"] = response_cost or 0.0

    # Step 2: SpendLogs 明细入队 (同步等待，不丢数据)
    if disable_spend_logs is False:
        await self._insert_spend_log_to_db(payload=copy.deepcopy(payload), prisma_client=prisma_client)

    # Step 3: 异步批量更新各维度表 (create_task, 不阻塞当前函数)
    asyncio.create_task(self._batch_database_updates(
        response_cost=response_cost,
        user_id=user_id, hashed_token=hashed_token,
        team_id=team_id, org_id=org_id, end_user_id=end_user_id,
        prisma_client=prisma_client, user_api_key_cache=user_api_key_cache,
        litellm_proxy_budget_name=litellm_proxy_budget_name,
        payload_copy=payload_copy, request_tags=request_tags,
    ))
```

### 9.4 两级队列模型

#### 第一级：SpendUpdateQueue (实体级累加)

**文件**: `litellm/proxy/db/db_transaction_queue/spend_update_queue.py`

每次请求将费用增量写入 `asyncio.Queue`，按实体类型分类：

| 实体类型 | 说明 | 更新目标表 |
|----------|------|------------|
| `KEY` | API Key 费用累加 | `LiteLLM_VerificationToken.spend` |
| `USER` | 用户费用累加 | `LiteLLM_UserTable.spend` |
| `END_USER` | 最终用户费用 (upsert) | `LiteLLM_EndUserTable.spend` |
| `TEAM` | 团队费用累加 | `LiteLLM_TeamTable.spend` |
| `TEAM_MEMBER` | 团队成员费用 | `LiteLLM_TeamMembership.spend` |
| `ORGANIZATION` | 组织费用累加 | `LiteLLM_OrganizationTable.spend` |
| `AGENT` | Agent 费用累加 | `LiteLLM_AgentTable.spend` |
| `TAG` | Tag 费用累加 | (通过 Daily 表) |

```python
# 入队示例
await self.spend_update_queue.add_update(
    update=SpendUpdateQueueItem(
        entity_type=Litellm_EntityType.USER,
        entity_id=user_id,
        response_cost=response_cost,
    )
)
```

#### 第二级：DailySpendUpdateQueue (日维度聚合)

**文件**: `litellm/proxy/db/db_transaction_queue/daily_spend_update_queue.py`

按 `{entity_id}_{date}_{api_key}_{model}_{provider}_{endpoint}` 聚合每日花费：

| 日维度表 | 唯一约束键 |
|----------|------------|
| `LiteLLM_DailyUserSpend` | `user_id + date + api_key + model + provider + tool + endpoint` |
| `LiteLLM_DailyTeamSpend` | `team_id + date + api_key + model + provider + tool + endpoint` |
| `LiteLLM_DailyOrganizationSpend` | `organization_id + date + ...` |
| `LiteLLM_DailyEndUserSpend` | `end_user_id + date + ...` |
| `LiteLLM_DailyAgentSpend` | `agent_id + date + ...` |
| `LiteLLM_DailyTagSpend` | `tag + date + ...` |

### 9.5 定时刷入数据库

**文件**: `litellm/proxy/utils.py`

定时任务 `update_spend()` **每分钟**执行一次，负责将内存队列中的数据批量写入 PostgreSQL：

```python
async def update_spend(prisma_client, db_writer_client, proxy_logging_obj):
    """Batch write updates to db. Triggered every minute."""
    n_retry_times = 3

    # 1. 刷入实体级 spend 增量 + 日维度聚合数据
    await proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler(
        prisma_client=prisma_client,
        n_retry_times=n_retry_times,
        proxy_logging_obj=proxy_logging_obj,
    )

    # 2. 刷入 SpendLogs 明细
    if queue_size > 0:
        await update_spend_logs_job(prisma_client, db_writer_client, proxy_logging_obj)
```

#### 两种刷入模式

**模式 A — 无 Redis (单 Pod)**:

```
SpendUpdateQueue.flush() → 聚合 → Prisma batch → PostgreSQL
DailySpendUpdateQueue.flush() → 聚合 → Prisma execute_raw (UPSERT) → PostgreSQL
```

**模式 B — 有 Redis (多 Pod)**:

```
各 Pod: 内存队列 → Redis (RPUSH)
Leader Pod (获取分布式锁):
    Redis (LRANGE + DEL) → 聚合 → Prisma batch → PostgreSQL
```

> **设计要点**: Redis 模式下只有获取到 `PodLockManager` 锁的 Leader Pod 执行 DB 写入，避免多 Pod 并发写入导致的死锁问题。

### 9.6 SpendLogs 明细写入

**SpendLogsPayload** 由 `get_logging_payload()` 构建（文件: `litellm/proxy/spend_tracking/spend_tracking_utils.py`），包含完整的请求审计信息：

| 字段 | 说明 |
|------|------|
| `request_id` | 请求唯一标识 |
| `call_type` | 调用类型 (completion/embedding/...) |
| `api_key` | 使用的 API Key (hash) |
| `model` | 模型名称 |
| `spend` | 本次请求费用 (USD) |
| `prompt_tokens` | 输入 token 数 |
| `completion_tokens` | 输出 token 数 |
| `total_tokens` | 总 token 数 |
| `user` / `team_id` / `organization_id` | 归属实体 |
| `startTime` / `endTime` | 请求时间范围 |
| `cache_hit` | 是否缓存命中 |
| `metadata` | 包含 usage_object, model_map_information 等详细元信息 |

明细数据先追加到 `prisma_client.spend_log_transactions` 内存列表，由 `update_spend_logs_job` 定时批量写入 `LiteLLM_SpendLogs` 表。

### 9.7 数据流总结

```
请求完成
    ↓
async_success_handler()              ← 异步回调
    ↓
_PROXY_track_cost_callback()          ← 提取 response_cost
    ↓
DBSpendUpdateWriter.update_database()
    ├─ _insert_spend_log_to_db()      ← SpendLogs 明细 → 内存列表
    └─ _batch_database_updates()      ← asyncio.create_task (fire-and-forget)
         ├─ _update_{user,key,team,org,agent,tag}_db()
         │     → SpendUpdateQueue      ← 实体级 spend 增量
         └─ add_spend_log_transaction_to_daily_{user,team,org,...}_transaction()
               → DailySpendUpdateQueue ← 日维度聚合
    ↓
定时任务 update_spend() (每分钟)
    ├─ db_update_spend_transaction_handler()
    │     ├─ flush SpendUpdateQueue → Prisma batch UPDATE → PostgreSQL 实体表
    │     └─ flush DailySpendUpdateQueue → Prisma UPSERT → PostgreSQL 日维度表
    └─ update_spend_logs_job()
          → prisma_client.spend_log_transactions → Prisma batch CREATE → LiteLLM_SpendLogs
```

---

## 10. 关键设计模式

### 10.1 费用计算的幂等性

同一响应对象的费用**只计算一次**：

```python
# response_cost_calculator 内部会检查
if "response_cost" in hidden_params:
    return hidden_params["response_cost"]  # 直接返回已计算结果
```

### 10.2 缓存命中免费

```python
if cache_hit is True:
    return 0.0  # 缓存读取不收费
```

### 10.3 定价回退机制

当模型不在定价表中时抛出明确异常：

```python
raise Exception(
    f"Model not found in cost map: {model}. "
    "Register model via custom pricing or PR - "
    "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
)
```

### 10.4 多 Provider 统一路由

不同 Provider 可能有不同的 `cost_per_token` 实现：

| Provider | 文件 | 特殊逻辑 |
|----------|------|----------|
| OpenAI | `llms/openai/cost_calculation.py` | Service Tier 定价 |
| Anthropic | `llms/anthropic/chat/transformation.py` | Prompt Caching 定价 |
| Azure | `llms/azure/cost_calculation.py` | TTS 按秒计费 |
| Vertex AI | `llm/vertex_ai/...` | 字符数计费 |
| DashScope | `llms/dashscope/chat/transformation.py` | 复用 OpenAI 兼容层 |

---

## 11. 性能优化建议

1. **避免重复计算**: `response_cost_calculator` 会先检查 `hidden_params.response_cost`
2. **模型查找优化**: `potential_model_names` 列表通常只有 2-3 个元素
3. **Usage 解析**: 大部分 Provider 返回的 `usage` 字段格式一致，但细节可能有差异
4. **Tier 定价**: 对于高吞吐场景，合理配置 Service Tier 可以显著降低成本
5. **自定义定价**: 对于内部/私有模型，通过 Custom Pricing 实现精确成本核算


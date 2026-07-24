# Silent 模型对比可观测性故障记录

## 状态

2026-07-24，在同时使用 Primary 模型和后台 `silent_model` 的 LiteLLM 环境中确认该问题

本文只记录已经观察到的故障现象、证据和可观测性缺口，不包含具体修复实现

## 测试链路

每个客户端请求预期产生两次模型调用：

1. Primary 模型调用，其结果返回给客户端
2. 后台 Silent 模型调用，只用于模型效果和性能对比

本次检查的模型为：

- Primary LiteLLM 模型：`bailian/deepseek-v4-flash`
- Silent LiteLLM 模型：`glm-5.2`
- Silent Provider 模型：`zai-org/GLM-5.2-FP8`
- 共同使用的 Key Alias：`llm-shadow-yj-pre-20260723`

## 观察到的现象

### Request Logs 数量看起来不是一比一

LiteLLM Request Logs 页面按具体 Deployment 过滤时显示：

- GLM Deployment：134 条
- DeepSeek Deployment：125 条

在相同时间范围内，直接按 Provider 模型名称查询得到两边各 137 条原始日志。页面上的数量差异由多种日志形态共同造成：

- DeepSeek 的部分失败发生在选出具体 Deployment 之前，因此日志里没有 `model_id`，使用具体 Deployment 筛选时会被排除
- 截图和 API 查询的结束时间有轻微差异
- 7 个 GLM Session ID 分别产生了 3 条 GLM 记录，一共多出 14 条重复侧记录
- Cache Hit 场景下，Primary 和 Silent 有时使用不同的 Session ID

因此，即使两边原始日志总数相等，也不能证明每个 Primary 请求都有且只有一个对应的 Silent 请求

### 无法完整地按 Session 配对

在检查的时间范围内：

| 统计项 | GLM | DeepSeek |
| --- | ---: | ---: |
| 原始日志数 | 137 | 137 |
| 唯一 Session ID 数 | 123 | 137 |
| 成功日志数 | 127 | 111 |
| Cache Hit 数 | 10 | 5 |
| 失败日志数 | 0 | 21 |

只有 111 个 Session ID 能组成干净、成功的 Primary/Silent 配对。另有 12 个 Session 只包含 GLM，26 个 Session 只包含 DeepSeek

DeepSeek 的失败类型包括 `APIConnectionError`、`BudgetExceededError`、`ProxyRateLimitError` 和 `TypeError`。部分失败没有 Deployment ID。Cache Hit 和重复调用也会破坏基于 Session ID 的严格一角色一行关联

### Silent 模型存在明显的延迟长尾

在 111 组干净配对中，两边调用的开始时间通常只相差几毫秒，可以确认属于同一次 Primary/Silent 对比：

| Duration | GLM Silent | DeepSeek Primary |
| --- | ---: | ---: |
| P50 | 15.0 秒 | 3.95 秒 |
| P90 | 80.1 秒 | 9.74 秒 |
| P95 | 112.0 秒 | 11.8 秒 |
| P99 | 154.4 秒 | 21.4 秒 |
| 最大值 | 346.2 秒 | 27.0 秒 |

同 Session 下，GLM 与 DeepSeek 的耗时比值中位数为 3.45 倍，P95 为 22.96 倍，最大值为 62.38 倍

GLM 经常生成更多输出 Token。配对后的输出 Token 比值中位数为 1.62 倍，P90 为 5.69 倍。输出长度可以解释部分延迟差异，但不能解释全部差异。例如，有一组请求中 GLM 输出 422 Token、耗时约 100 秒，而 DeepSeek 输出 181 Token、耗时约 3.7 秒

### 现有 Prometheus 指标无法找到差异最大的配对请求

当前部署的 `litellm_request_total_latency_metric` 包含以下聚合维度：

- `requested_model`
- `model`
- `model_id`
- `api_provider`
- Key、Team、Organization 和 User 等维度

它不包含 `session_id`、稳定的 Shadow Pair ID，也不包含 Primary/Silent 请求角色

因此，Prometheus 可以比较不同模型的整体延迟分布，但无法回答：

- 哪一条 Primary 样本和哪一条 Silent 样本属于同一个请求
- 哪些 Session 的耗时比值或耗时差值最大
- 哪些 Primary 请求成功，但对应的 Silent 请求失败或缺失
- 数量差异来自配对缺失、重试、Cache Hit，还是重复调用

不能简单地把 `session_id` 添加成普通 Prometheus Label。这样会为几乎每个请求创建新的时间序列，造成高基数问题。Histogram 指标只保留桶计数，不保存每次请求的原始观测值，因此采集完成后也无法重新执行逐 Session 关联

### Primary 和 Silent 的 TTFT 不能直接比较

Silent 请求会被强制设置为 `stream=False`，以保证后台响应被完整消费并触发 Callback。Primary 请求则可能使用 Streaming

对于非 Streaming 的 Silent 请求，页面显示的 TTFT 可能等于或接近完整请求耗时，不能直接拿来和 Streaming Primary 请求的 TTFT 比较

### 当前 OTEL 不能自动把 Primary 和 Silent 关联成同一条 Trace

OpenTelemetry 很适合承载这种对比关系。理想情况下，同一个客户端请求对应一条 Trace，Primary 和 Silent 分别是其中的两个模型调用 Span；Span 上记录请求角色、模型、耗时、状态和 Token 数。Grafana 配合 Tempo 等 Trace Backend 后，就可以筛选慢请求并从指标跳转到具体 Trace

但是，仅配置 OTEL Exporter 还不能自动得到上述关系。当前 Silent 实现包含以下逻辑：

```python
def _get_silent_experiment_kwargs(self, **kwargs) -> dict:
    from litellm.litellm_core_utils.core_helpers import safe_deep_copy

    silent_kwargs = safe_deep_copy(kwargs)
    original_metadata = kwargs.get("metadata")
    if original_metadata is not None and silent_kwargs.get("metadata") is original_metadata:
        silent_kwargs["metadata"] = dict(original_metadata)

    if "metadata" not in silent_kwargs:
        silent_kwargs["metadata"] = {}

    silent_kwargs["metadata"].pop("litellm_parent_otel_span", None)
    silent_kwargs["metadata"]["is_silent_experiment"] = True
    silent_kwargs["stream"] = False
    silent_kwargs.pop("litellm_call_id", None)
    silent_kwargs.pop("litellm_logging_obj", None)
    silent_kwargs.pop("standard_logging_object", None)
    return silent_kwargs
```

这里的“主动移除”指的是 LiteLLM 代码明确执行了：

```python
silent_kwargs["metadata"].pop("litellm_parent_otel_span", None)
```

这不是用户配置删除了 Span，也不是 OTEL Exporter 丢失了数据。原因是 Silent 请求在后台线程和新的 Event Loop 中运行，不能安全地把一个仍然存活的 OTel Span 对象直接传到另一个 Event Loop。代码为了避免跨线程或跨 Event Loop 使用 Span 导致竞态和上下文损坏，先删除了这个对象

这个保护逻辑本身有合理目的，但副作用是 Silent 调用失去了原有的父 Span 引用。同时代码还为 Silent 调用创建新的 LiteLLM Call ID 和 Logging Context，并且没有创建指向 Primary Span 的 Span Link。因此当前行为是：

- Primary 模型调用可以出现在原始 HTTP 请求 Trace 中
- Silent 模型调用可能成为另一条 Trace，或者成为无法和原请求稳定关联的 Span
- 两边虽然可能在 Spend Logs 中保留相同 Session ID，但当前 OTEL GenAI Span 没有把通用 LiteLLM Session ID 作为默认的一等对比属性
- Grafana/Tempo 无法仅依靠现有 Trace 结构稳定地列出同一次请求的 Primary/Silent 差异

所以，配置 OTEL 的目标确实可以是让 Grafana 更快发现并定位问题，但当前还缺少跨后台边界的关联信息。需要传播可序列化的 OTel Context，或者显式创建 Span Link，并增加稳定的 Shadow Pair ID 和 `request_role=primary|silent` 属性。不能直接跨线程传递 live Span 对象

## 对 Grafana 的预期能力

完成关联后，Grafana 应提供两层入口：

1. Metrics 面板快速发现哪段时间、哪一对模型发生整体退化，包括配对成功率、失败率、耗时差值和耗时比值分布
2. Trace 面板按 Shadow Pair ID 展示 Primary 和 Silent Span，并快速筛选耗时比值大、耗时差值大、状态不一致或缺失一侧的请求

推荐的使用路径是：

```text
Prometheus 指标发现异常时间段或模型组合
    -> 通过 Exemplar 或 Data Link 跳转到 Grafana Trace
    -> 在同一条 Trace 中对比 Primary 和 Silent Span
    -> 必要时继续跳转到对应 Request Logs
```

OTEL 主要解决单次请求的上下文、时序和下钻定位，Prometheus 主要解决整体趋势和告警。两者结合后，Grafana 才能同时做到快速发现和快速定位

## 用户影响

当前只能从 Grafana 看出某个模型的整体 P95 或 P99 较差，无法马上得到按差异程度排序的配对请求列表

目前的排查方式需要查询或导出 Spend Logs，按 Session ID 关联记录，排除 Cache Hit 和重复调用，再手工计算耗时比值。这会延长性能退化、Silent 请求缺失和 Provider 失败的发现与定位时间

## 复现与验收范围

同时满足以下条件时，可以确认该可观测性问题仍然存在：

1. 请求配置了 `silent_model`
2. Request Logs 中部分 Primary 和 Silent 记录可以通过 Session ID 关联
3. Prometheus 中存在两边模型各自的聚合延迟指标
4. 现有 Metrics 或 Trace 查询无法返回按照耗时比值或耗时差值排序的精确配对请求
5. Cache Hit、失败或重复调用导致 UI 日志数或唯一 Session 数量出现差异

未来的修复需要覆盖：正常成功配对、Primary 单边记录、Silent 单边记录、缺少 Deployment ID 的 Provider 失败、Cache Hit、重试、重复调用、Streaming Primary 请求，以及非 Streaming Silent 请求

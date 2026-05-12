# LiteLLM 监控数据全览

本文档汇总了 LiteLLM 项目中所有的监控数据，包括指标（Metrics）、告警（Alerting）、日志（Logging）、费用追踪（Spend Tracking）和健康检查（Health Check）。

---

## 一、Prometheus 指标（/metrics 端点）

LiteLLM Proxy 通过 `PrometheusLogger`（`litellm/integrations/prometheus.py`）暴露以下 Prometheus 指标：

### 1.1 请求指标


| 指标名                                   | 类型    | 描述                                                 |
| ---------------------------------------- | ------- | ---------------------------------------------------- |
| `litellm_proxy_total_requests_metric`    | Counter | 代理服务器接收的总请求数                             |
| `litellm_proxy_failed_requests_metric`   | Counter | 代理返回失败响应的总数                               |
| `litellm_requests_metric`                | Counter | (已废弃) LLM 调用总数，按 API Key / Team / User 维度 |
| `litellm_llm_api_failed_requests_metric` | Counter | (已废弃) LLM API 失败请求总数                        |

### 1.2 延迟指标


| 指标名                                       | 类型      | 描述                               |
| -------------------------------------------- | --------- | ---------------------------------- |
| `litellm_request_total_latency_metric`       | Histogram | 请求的总端到端延迟（秒）           |
| `litellm_llm_api_latency_metric`             | Histogram | LLM API 调用延迟（秒）             |
| `litellm_llm_api_time_to_first_token_metric` | Histogram | 首 Token 响应时间（TTFT）          |
| `litellm_overhead_latency_metric`            | Histogram | LiteLLM 自身处理的额外延迟（毫秒） |
| `litellm_request_queue_time_seconds`         | Histogram | 请求在队列中的等待时间（秒）       |

### 1.3 Token 与费用指标


| 指标名                         | 类型    | 描述                   |
| ------------------------------ | ------- | ---------------------- |
| `litellm_spend_metric`         | Counter | LLM 请求的总花费       |
| `litellm_total_tokens_metric`  | Counter | 输入 + 输出 Token 总数 |
| `litellm_input_tokens_metric`  | Counter | 输入 Token 总数        |
| `litellm_output_tokens_metric` | Counter | 输出 Token 总数        |

### 1.4 预算指标


| 指标名                                          | 类型  | 描述                                            |
| ----------------------------------------------- | ----- | ----------------------------------------------- |
| `litellm_remaining_team_budget_metric`          | Gauge | Team 剩余预算                                   |
| `litellm_team_max_budget_metric`                | Gauge | Team 最大预算                                   |
| `litellm_team_budget_remaining_hours_metric`    | Gauge | Team 预算重置剩余小时数                         |
| `litellm_remaining_org_budget_metric`           | Gauge | Organization 剩余预算                           |
| `litellm_org_max_budget_metric`                 | Gauge | Organization 最大预算                           |
| `litellm_org_budget_remaining_hours_metric`     | Gauge | Organization 预算重置剩余小时数                 |
| `litellm_remaining_api_key_budget_metric`       | Gauge | API Key 剩余预算                                |
| `litellm_api_key_max_budget_metric`             | Gauge | API Key 最大预算                                |
| `litellm_api_key_budget_remaining_hours_metric` | Gauge | API Key 预算重置剩余小时数                      |
| `litellm_remaining_user_budget_metric`          | Gauge | User 剩余预算                                   |
| `litellm_user_max_budget_metric`                | Gauge | User 最大预算                                   |
| `litellm_user_budget_remaining_hours_metric`    | Gauge | User 预算重置剩余小时数                         |
| `litellm_provider_remaining_budget_metric`      | Gauge | Provider 剩余预算（设置了 Provider 预算限制时） |

### 1.5 速率限制指标


| 指标名                                         | 类型  | 描述                                                      |
| ---------------------------------------------- | ----- | --------------------------------------------------------- |
| `litellm_remaining_requests_metric`            | Gauge | 模型剩余请求数（来自 LLM Provider 响应头）                |
| `litellm_remaining_tokens_metric`              | Gauge | 模型剩余 Token 数（来自 LLM Provider 响应头）             |
| `litellm_remaining_api_key_requests_for_model` | Gauge | API Key 对特定模型的剩余请求数（基于 Key 的 RPM 限制）    |
| `litellm_remaining_api_key_tokens_for_model`   | Gauge | API Key 对特定模型的剩余 Token 数（基于 Key 的 TPM 限制） |

### 1.6 缓存指标


| 指标名                         | 类型    | 描述                    |
| ------------------------------ | ------- | ----------------------- |
| `litellm_cache_hits_metric`    | Counter | 缓存命中总数            |
| `litellm_cache_misses_metric`  | Counter | 缓存未命中总数          |
| `litellm_cached_tokens_metric` | Counter | 从缓存返回的 Token 总数 |

### 1.7 部署（Deployment）分析指标


| 指标名                                        | 类型      | 描述                                     |
| --------------------------------------------- | --------- | ---------------------------------------- |
| `litellm_deployment_state`                    | Gauge     | 部署状态：0=健康, 1=部分故障, 2=完全故障 |
| `litellm_deployment_tpm_limit`                | Gauge     | 部署配置的 TPM 限制                      |
| `litellm_deployment_rpm_limit`                | Gauge     | 部署配置的 RPM 限制                      |
| `litellm_deployment_cooled_down`              | Counter   | 部署被冷却（cooldown）的次数             |
| `litellm_deployment_success_responses`        | Counter   | 部署成功响应总数                         |
| `litellm_deployment_failure_responses`        | Counter   | 部署失败响应总数                         |
| `litellm_deployment_total_requests`           | Counter   | 部署总请求数（成功+失败）                |
| `litellm_deployment_latency_per_output_token` | Histogram | 每个输出 Token 的延迟                    |
| `litellm_deployment_successful_fallbacks`     | Counter   | 成功的 Fallback 次数                     |
| `litellm_deployment_failed_fallbacks`         | Counter   | 失败的 Fallback 次数                     |

### 1.8 Guardrail 指标


| 指标名                              | 类型      | 描述                     |
| ----------------------------------- | --------- | ------------------------ |
| `litellm_guardrail_latency_seconds` | Histogram | Guardrail 执行延迟（秒） |
| `litellm_guardrail_errors_total`    | Counter   | Guardrail 执行错误总数   |
| `litellm_guardrail_requests_total`  | Counter   | Guardrail 调用总数       |

### 1.9 Batch 管理指标


| 指标名                                          | 类型      | 描述                              |
| ----------------------------------------------- | --------- | --------------------------------- |
| `litellm_managed_batch_created_total`           | Counter   | 创建的 Managed Batch 总数         |
| `litellm_managed_file_size_bytes`               | Gauge     | Managed Batch 文件大小（字节）    |
| `litellm_managed_batch_duration_seconds`        | Histogram | Batch 完成时长（秒）              |
| `litellm_managed_file_created_total`            | Counter   | 创建的 Managed File 总数          |
| `litellm_managed_file_deleted_total`            | Counter   | 删除的 Managed File 总数          |
| `litellm_check_batch_cost_jobs_polled`          | Gauge     | 最近一次轮询发现的未处理 Batch 数 |
| `litellm_check_batch_cost_jobs_processed_total` | Counter   | 已处理 Batch 费用统计数           |
| `litellm_check_batch_cost_errors_total`         | Counter   | Batch 费用统计错误数              |
| `litellm_check_batch_cost_last_run_timestamp`   | Gauge     | 最近一次 Batch 费用统计运行时间戳 |

### 1.10 系统统计指标


| 指标名                                     | 类型    | 描述                      |
| ------------------------------------------ | ------- | ------------------------- |
| `litellm_total_users`                      | Gauge   | 系统中的用户总数          |
| `litellm_teams_count`                      | Gauge   | 系统中的团队总数          |
| `litellm_callback_logging_failures_metric` | Counter | Callback 日志记录失败次数 |

---

## 二、服务健康监控（Prometheus Services）

`PrometheusServicesLogger`（`litellm/integrations/prometheus_services.py`）监控 LiteLLM 及其依赖服务的健康状态。

### 监控的服务类型


| 服务                              | 枚举值                               | 描述                 |
| --------------------------------- | ------------------------------------ | -------------------- |
| Redis                             | `redis`                              | Redis 缓存服务       |
| PostgreSQL                        | `postgres`                           | 数据库服务           |
| Batch Write to DB                 | `batch_write_to_db`                  | 批量写入数据库操作   |
| Reset Budget Job                  | `reset_budget_job`                   | 预算重置定时任务     |
| LiteLLM Self                      | `self`                               | LiteLLM 自身服务     |
| Router                            | `router`                             | 路由器服务           |
| Auth                              | `auth`                               | 认证服务             |
| Proxy Pre-Call                    | `proxy_pre_call`                     | 代理预调用处理       |
| Pod Lock Manager                  | `pod_lock_manager`                   | Pod 锁管理           |
| Daily Spend Update Queue (Memory) | `in_memory_daily_spend_update_queue` | 内存中日花费更新队列 |
| Daily Spend Update Queue (Redis)  | `redis_daily_spend_update_queue`     | Redis 日花费更新队列 |

### 每个服务的指标

- **Histogram**: `litellm_{service}_latency` — 服务请求延迟
- **Counter**: `litellm_{service}_failed_requests` — 服务失败请求数
- **Counter**: `litellm_{service}_total_requests` — 服务总请求数
- **Gauge**: `litellm_{service}_size` — 服务队列大小（部分服务）

---

## 三、告警系统

### 3.1 Slack 告警

通过 `SlackAlerting`（`litellm/integrations/SlackAlerting/slack_alerting.py`）实现。

#### 告警类型


| 告警类型                | 描述             |
| ----------------------- | ---------------- |
| `llm_exceptions`        | LLM 调用异常     |
| `llm_too_slow`          | LLM 响应过慢     |
| `llm_requests_hanging`  | LLM 请求挂起     |
| `budget_alerts`         | 预算告警         |
| `spend_reports`         | 花费报告         |
| `failed_tracking_spend` | 花费追踪失败     |
| `db_exceptions`         | 数据库异常       |
| `daily_reports`         | 每日报告         |
| `cooldown_deployment`   | 部署冷却告警     |
| `new_model_added`       | 新模型添加通知   |
| `outage_alerts`         | 服务中断告警     |
| `region_outage_alerts`  | 区域服务中断告警 |
| `fallback_reports`      | Fallback 报告    |

#### 管理事件告警


| 事件类型                    | 描述           |
| --------------------------- | -------------- |
| `new_virtual_key_created`   | 创建新虚拟密钥 |
| `virtual_key_updated`       | 更新虚拟密钥   |
| `virtual_key_deleted`       | 删除虚拟密钥   |
| `new_team_created`          | 创建新团队     |
| `team_updated`              | 更新团队       |
| `team_deleted`              | 删除团队       |
| `new_internal_user_created` | 创建新内部用户 |
| `internal_user_updated`     | 更新内部用户   |
| `internal_user_deleted`     | 删除内部用户   |

#### 预算告警类别


| 告警类别                   | 描述           |
| -------------------------- | -------------- |
| `proxy_budget`             | 代理整体预算   |
| `soft_budget`              | 软预算超限     |
| `user_budget`              | 用户预算       |
| `team_budget`              | 团队预算       |
| `organization_budget`      | 组织预算       |
| `token_budget`             | API Key 预算   |
| `project_budget`           | 项目预算       |
| `projected_limit_exceeded` | 预测将超出限额 |

### 3.2 邮件告警

通过 `email_alerting.py` 支持邮件通知，支持 Resend、SendGrid、SMTP 三种邮件服务。

### 3.3 PagerDuty

通过 `pagerduty` 回调集成，用于严重告警升级。

### 3.4 Webhook

通过 `generic_api` 回调支持自定义 Webhook 告警。

---

## 四、日志与可观测性集成

### 4.1 支持的集成平台


| 集成                | 回调名                      | 描述                        |
| ------------------- | --------------------------- | --------------------------- |
| **Prometheus**      | `prometheus`                | 指标采集与暴露              |
| **OpenTelemetry**   | `otel`                      | 分布式链路追踪 + 指标       |
| **Datadog**         | `datadog`                   | 日志推送                    |
| **Datadog Metrics** | `datadog_metrics`           | Datadog 指标                |
| **Datadog LLM Obs** | `datadog_llm_observability` | Datadog LLM 可观测性        |
| **Langfuse**        | `langfuse`                  | LLM 可观测性平台            |
| **Langfuse OTEL**   | `langfuse_otel`             | 通过 OTEL 协议对接 Langfuse |
| **LangSmith**       | `langsmith`                 | LangChain 生态可观测性      |
| **Arize / Phoenix** | `arize` / `arize_phoenix`   | ML 可观测性平台             |
| **Braintrust**      | `braintrust`                | AI 产品开发平台             |
| **MLflow**          | `mlflow`                    | ML 实验跟踪                 |
| **Opik**            | `opik`                      | LLM 评估与监控              |
| **Galileo**         | `galileo`                   | AI 质量平台                 |
| **LangTrace**       | `langtrace`                 | LLM 追踪                    |
| **Logfire**         | `logfire`                   | Pydantic 团队可观测性       |
| **Literal AI**      | `literalai`                 | 对话 AI 可观测性            |
| **Humanloop**       | `humanloop`                 | LLM 评估平台                |
| **AgentOps**        | `agentops`                  | Agent 可观测性              |
| **Weave**           | `weave_otel`                | Weights & Biases Weave      |
| **PostHog**         | `posthog`                   | 产品分析                    |
| **DeepEval**        | `deepeval`                  | LLM 评测框架                |
| **Lunary**          | —                          | LLM 监控平台                |
| **Helicone**        | —                          | LLM 代理 + 可观测性         |
| **Traceloop**       | —                          | OpenTelemetry for LLMs      |
| **PromptLayer**     | —                          | Prompt 管理与日志           |

### 4.2 日志存储集成


| 集成                   | 回调名           | 描述                          |
| ---------------------- | ---------------- | ----------------------------- |
| **S3**                 | `s3_v2`          | AWS S3 日志存储               |
| **GCS Bucket**         | `gcs_bucket`     | Google Cloud Storage 日志存储 |
| **GCS Pub/Sub**        | `gcs_pubsub`     | Google Cloud Pub/Sub 日志推送 |
| **Azure Blob Storage** | `azure_storage`  | Azure Blob 日志存储           |
| **Azure Sentinel**     | `azure_sentinel` | Azure 安全信息与事件管理      |
| **AWS SQS**            | `aws_sqs`        | AWS SQS 消息队列              |
| **DynamoDB**           | —               | AWS DynamoDB 日志存储         |
| **Supabase**           | —               | Supabase 日志存储             |
| **Bitbucket**          | `bitbucket`      | Bitbucket Pipeline 日志       |
| **GitLab**             | `gitlab`         | GitLab 集成                   |

### 4.3 费用管理集成


| 集成           | 回调名      | 描述                |
| -------------- | ----------- | ------------------- |
| **Lago**       | `lago`      | 用量计费平台        |
| **OpenMeter**  | `openmeter` | 用量计量            |
| **CloudZero**  | `cloudzero` | 云成本优化          |
| **Vantage**    | `vantage`   | 云成本可视化        |
| **Focus**      | `focus`     | FinOps 费用数据标准 |
| **GreenScale** | —          | AI 碳排放追踪       |

---

## 五、StandardLoggingPayload（统一日志格式）

所有回调都接收标准化的 `StandardLoggingPayload`，包含以下核心字段：


| 字段                    | 类型          | 描述                                    |
| ----------------------- | ------------- | --------------------------------------- |
| `id`                    | str           | 请求唯一 ID                             |
| `trace_id`              | str           | 追踪 ID（关联同一请求的重试/fallback）  |
| `call_type`             | str           | 调用类型（completion, embedding 等）    |
| `stream`                | bool          | 是否流式请求                            |
| `response_cost`         | float         | 请求费用                                |
| `cost_breakdown`        | dict          | 费用明细                                |
| `status`                | str           | 请求状态                                |
| `total_tokens`          | int           | 总 Token 数                             |
| `prompt_tokens`         | int           | 输入 Token 数                           |
| `completion_tokens`     | int           | 输出 Token 数                           |
| `startTime`             | float         | 请求开始时间                            |
| `endTime`               | float         | 请求结束时间                            |
| `completionStartTime`   | float         | 首 Token 时间                           |
| `response_time`         | float         | 响应时间                                |
| `model`                 | str           | 使用的模型                              |
| `model_id`              | str           | 模型部署 ID                             |
| `model_group`           | str           | 模型组                                  |
| `custom_llm_provider`   | str           | LLM 提供商                              |
| `api_base`              | str           | API 地址                                |
| `cache_hit`             | bool          | 是否缓存命中                            |
| `cache_key`             | str           | 缓存 Key                                |
| `saved_cache_cost`      | float         | 缓存节省的费用                          |
| `request_tags`          | list          | 请求标签                                |
| `end_user`              | str           | 终端用户 ID                             |
| `requester_ip_address`  | str           | 请求者 IP                               |
| `messages`              | str/list/dict | 请求消息内容                            |
| `response`              | str/list/dict | 响应内容                                |
| `error_str`             | str           | 错误信息                                |
| `metadata`              | dict          | 元数据（包含 API Key、Team、User 信息） |
| `guardrail_information` | list          | Guardrail 执行信息                      |

---

## 六、费用追踪（Spend Tracking）

### 6.1 数据库表


| 表名                             | 描述                     |
| -------------------------------- | ------------------------ |
| `LiteLLM_SpendLogs`              | 每次请求的详细费用日志   |
| `LiteLLM_ErrorLogs`              | 错误请求日志             |
| `LiteLLM_DailyUserSpend`         | 按用户的每日花费聚合     |
| `LiteLLM_DailyTeamSpend`         | 按团队的每日花费聚合     |
| `LiteLLM_DailyOrganizationSpend` | 按组织的每日花费聚合     |
| `LiteLLM_DailyEndUserSpend`      | 按终端用户的每日花费聚合 |
| `LiteLLM_DailyAgentSpend`        | 按 Agent 的每日花费聚合  |
| `LiteLLM_DailyTagSpend`          | 按标签的每日花费聚合     |

### 6.2 每日花费聚合字段

所有 `DailyXxxSpend` 表都包含：

- `date` — 日期
- `api_key` — API Key
- `model` / `model_group` — 模型
- `custom_llm_provider` — LLM 提供商
- `mcp_namespaced_tool_name` — MCP 工具名
- `endpoint` — 端点
- `prompt_tokens` — 输入 Token 数
- `completion_tokens` — 输出 Token 数
- `cache_read_input_tokens` — 缓存读取 Token 数
- `cache_creation_input_tokens` — 缓存创建 Token 数
- `spend` — 花费金额
- `api_requests` — API 请求数
- `successful_requests` — 成功请求数
- `failed_requests` — 失败请求数

### 6.3 预算管理

`LiteLLM_BudgetTable` 支持为以下实体设置预算：

- Organization
- Team
- Project
- User
- API Key (Verification Token)
- End User
- Tag

每个预算包含：`max_budget`、`soft_budget`、`tpm_limit`、`rpm_limit`、`budget_duration`、`budget_reset_at`。

---

## 七、审计日志（Audit Log）

`LiteLLM_AuditLog` 表记录所有管理操作：


| 字段                 | 描述                                 |
| -------------------- | ------------------------------------ |
| `changed_by`         | 操作人                               |
| `changed_by_api_key` | 操作使用的 API Key（哈希）           |
| `action`             | 操作类型（create / update / delete） |
| `table_name`         | 操作的表                             |
| `object_id`          | 操作对象 ID                          |
| `before_value`       | 变更前的值                           |
| `updated_values`     | 变更后的值                           |

---

## 八、健康检查端点


| 端点                        | 描述                                  |
| --------------------------- | ------------------------------------- |
| `GET /health`               | 模型健康检查（发送测试请求）          |
| `GET /health/liveliness`    | 存活探针（K8s liveness probe）        |
| `GET /health/liveness`      | 同上（K8s 标准命名）                  |
| `GET /health/readiness`     | 就绪探针（检查 DB + 缓存连接）        |
| `GET /health/backlog`       | 请求积压状态                          |
| `GET /health/services`      | 检查特定服务健康（如 Datadog, Slack） |
| `GET /health/history`       | 健康检查历史                          |
| `GET /health/latest`        | 最新健康检查结果                      |
| `GET /health/shared-status` | 共享健康状态                          |
| `GET /health/license`       | 许可证状态                            |

---

## 九、回调配置方式

```python
import litellm

# 成功回调
litellm.success_callback = ["langfuse", "prometheus", "datadog"]

# 失败回调
litellm.failure_callback = ["langfuse", "prometheus"]

# 服务监控回调
litellm.service_callback = ["prometheus_system"]

# 审计日志回调
litellm.audit_log_callbacks = ["generic_api"]

# 通用回调（同时覆盖成功和失败）
litellm.callbacks = ["langfuse", "prometheus"]
```

YAML 配置（Proxy）：

```yaml
litellm_settings:
  success_callback: ["langfuse", "prometheus"]
  failure_callback: ["langfuse"]
  service_callback: ["prometheus_system"]

general_settings:
  alerting: ["slack"]
  alert_types: ["llm_exceptions", "budget_alerts", "outage_alerts"]
```

---

## 十、自定义回调开发

通过继承 `CustomLogger`（`litellm/integrations/custom_logger.py`）可自定义监控回调：

```python
from litellm.integrations.custom_logger import CustomLogger

class MyLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # 接收 StandardLoggingPayload
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass
```

关键生命周期钩子：

- `log_pre_api_call` — API 调用前
- `log_success_event` / `async_log_success_event` — 成功后
- `log_failure_event` / `async_log_failure_event` — 失败后
- `log_stream_event` / `async_log_stream_event` — 流式事件

---

## 十一、Router 监控数据

Router（`litellm/router.py` + `litellm/router_utils/`）提供以下监控能力：

### Cooldown 机制

- 部署失败后自动冷却，防止流量继续打到故障节点
- 通过 `cooldown_cache.py` 追踪冷却状态
- Prometheus 指标 `litellm_deployment_cooled_down` 记录冷却事件

### 健康状态缓存

- `health_state_cache.py` 缓存部署健康状态
- 状态值：0（健康）、1（部分故障）、2（完全故障）

### Fallback 追踪

- 记录成功和失败的 Fallback 请求
- 支持 Fallback 报告告警

---

## 十二、数据库性能监控

`litellm/proxy/db/log_db_metrics.py` 记录数据库操作指标，通过 Prometheus 暴露 DB 相关的延迟和错误数据。

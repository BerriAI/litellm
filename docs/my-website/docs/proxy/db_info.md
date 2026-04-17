# 数据库中存储了什么

LiteLLM Proxy 使用 PostgreSQL 数据库存储各种信息。数据库主要用于以下功能：
- 虚拟密钥、组织、团队、用户、预算等的管理
- 按请求的用量追踪

## 数据库 Schema 链接

你可以在[这里](https://github.com/BerriAI/litellm/blob/main/schema.prisma)查看完整的数据库 Schema。

## 数据库表

### 组织、团队、用户、终端用户

| 表名 | 描述 | 行插入频率 |
|------------|-------------|---------------------|
| LiteLLM_OrganizationTable | 管理组织级别的配置。追踪组织花费、模型访问权限和元数据。关联预算配置和团队。 | 低 |
| LiteLLM_TeamTable | 处理组织内的团队级别设置。管理团队成员、管理员及其角色。控制团队专属的预算、速率限制和模型访问权限。 | 低 |
| LiteLLM_UserTable | 存储用户信息及其设置。追踪个人用户的花费、模型访问权限和速率限制。管理用户角色和团队成员关系。 | 低 |
| LiteLLM_EndUserTable | 管理终端用户配置。控制模型访问权限和区域要求。追踪终端用户花费。 | 低 |
| LiteLLM_TeamMembership | 追踪用户在团队中的参与关系。管理团队内用户的预算和花费。 | 低 |
| LiteLLM_OrganizationMembership | 管理用户在组织中的角色。追踪组织级别的用户权限和花费。 | 低 |
| LiteLLM_InvitationLink | 处理用户邀请。管理邀请状态和过期时间。追踪邀请的创建者和接受者。 | 低 |
| LiteLLM_UserNotifications | 处理模型访问请求。追踪用户的模型访问申请。管理审批状态。 | 低 |

### 认证

| 表名 | 描述 | 行插入频率 |
|------------|-------------|---------------------|
| LiteLLM_VerificationToken | 管理虚拟密钥及其权限。控制密钥专属的预算、速率限制和模型访问权限。追踪密钥的花费和元数据。 | **中** - 存储所有虚拟密钥 |

### 模型（LLM）管理

| 表名 | 描述 | 行插入频率 |
|------------|-------------|---------------------|
| LiteLLM_ProxyModelTable | 存储模型配置。定义可用模型及其参数。包含模型特定的信息和设置。 | 低 - 仅配置使用 |

### 预算管理

| 表名 | 描述 | 行插入频率 |
|------------|-------------|---------------------|
| LiteLLM_BudgetTable | 存储组织、密钥和终端用户的预算及速率限制配置。追踪最大预算、软预算、TPM/RPM 限制和模型级别的预算。处理预算周期和重置时间。 | 低 - 仅配置使用 |


### 追踪与日志

| 表名 | 描述 | 行插入频率 |
|------------|-------------|---------------------|
| LiteLLM_SpendLogs | 所有 API 请求的详细日志。记录 Token 用量、花费和耗时信息。追踪使用了哪些模型和密钥。 | **中 - 这是一个按时间间隔运行的批处理过程。** |
| LiteLLM_AuditLog | 追踪系统配置的变更。记录谁进行了修改以及修改了什么内容。维护团队、用户和模型的更新历史。 | **默认关闭**，**高 - 每次实体变更时都会运行** |

## 禁用 `LiteLLM_SpendLogs`

你可以通过在 proxy_config.yaml 文件的 `general_settings` 部分将 `disable_spend_logs` 和 `disable_error_logs` 设置为 `True` 来禁用花费日志和错误日志。

```yaml
general_settings:
  disable_spend_logs: True   # 禁止将花费日志写入数据库
  disable_error_logs: True   # 仅禁止将错误日志写入数据库，常规花费日志仍会写入，除非设置了 `disable_spend_logs: True`
```

### 禁用这些日志有什么影响？

禁用花费日志（`disable_spend_logs: True`）时：
- 你**将无法**在 LiteLLM UI 上查看用量
- 你**仍然可以**在 S3、Prometheus、Langfuse（以及你使用的任何其他日志集成）上看到成本指标

禁用错误日志（`disable_error_logs: True`）时：
- 你**将无法**在 LiteLLM UI 上查看错误
- 你**仍然可以**在应用日志和其他日志集成中看到错误日志


## 数据库迁移

如果你需要迁移数据库，以下表应被复制以确保服务的连续性和零停机时间：


| 表名 | 描述 |
|------------|-------------|
| LiteLLM_VerificationToken | **必须** 确保现有虚拟密钥继续工作 |
| LiteLLM_UserTable | **必须** 确保现有虚拟密钥继续工作 |
| LiteLLM_TeamTable | **必须** 确保团队数据被迁移 |
| LiteLLM_TeamMembership | **必须** 确保团队成员预算被迁移 |
| LiteLLM_BudgetTable | **必须** 迁移现有预算设置 |
| LiteLLM_OrganizationTable | **可选** 仅在数据库中使用了组织功能时才需要迁移 |
| LiteLLM_OrganizationMembership | **可选** 仅在数据库中使用了组织功能时才需要迁移 |
| LiteLLM_ProxyModelTable | **可选** 仅在将 LLM 存储在数据库中时才需要迁移（即设置了 `STORE_MODEL_IN_DB=True`） |
| LiteLLM_SpendLogs | **可选** 仅在需要 LiteLLM UI 上的历史数据时才需要迁移 |
| LiteLLM_ErrorLogs | **可选** 仅在需要 LiteLLM UI 上的历史数据时才需要迁移 |



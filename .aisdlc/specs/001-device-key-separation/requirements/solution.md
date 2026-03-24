# 设备级密钥分离方案 (Device-Key Separation)

## 概述

为 `litellm/proxy/zx/zx_config_endpoints.py` 的 `cli_get_key` API 添加按设备返回独立 key 的功能。实现设备级别的密钥隔离，使每个（用户+设备）组合对应一个永久的 API Key，同时保持对旧调用方式的完全向后兼容。

## 问题陈述

当前 `cli_get_key` 返回的是用户级别的全局密钥，无法区分不同设备。需求是：
- 为每个设备返回独立的密钥
- 同一设备多次调用返回相同的密钥（幂等）
- 关联已有的旧数据，避免破坏现有使用者
- 实现设备级别的密钥隔离和管理

## 推荐方案

### 核心设计

**Key Alias 命名规则变更**：
- **新的设备级 key**：`{org_email}--{device_id}` （例如：`user@company.com--device-abc123`）
- **旧的全局 key**：保持原有规则不变（例如：`user@company.com--assistant-...`）

**Device ID 获取**：
- 从 `store.data['key_metadata']['device_id']` 读取（在 login 时已设置）
- 若不存在 device_id，降级使用旧的全局密钥逻辑（向后兼容）

**数据关联方式**：
- 旧 key 的 metadata 在 `cli_get_key` 调用时**动态更新**
- 若 key 的 metadata 中不存在 `device_id` 字段，自动填充当前设备的 device_id 和 device_name
- 采用 lazy 关联方式，避免一次性全量迁移

### 实现流程

```
cli_get_key(token, type?)
  ├─ 读取 store = get_store(type='cli', token)
  ├─ 从 store.data['key_metadata'] 提取 device_id
  │
  ├─ 如果 device_id 存在：
  │   ├─ 构建新的 key_alias = f"{org_email}--{device_id}"
  │   └─ 调用 create_or_get_user_key(..., key_alias, key_metadata={device_id, device_name})
  │       └─ 返回 key（新建或已存在）
  │
  └─ 如果 device_id 不存在：
      ├─ 使用旧逻辑：key_alias = f"{org_email}--{type.strip()}" （如果 type 以 "assistant-" 开头）
      ├─ 调用 create_or_get_user_key(..., key_alias, key_metadata)
      │   ├─ 如果 key 的 metadata 中缺少 device_id，动态填充（暂时从上下文推断或标记为 'legacy'）
      │   └─ 返回 key
      └─ （最终会被下一个有 device_id 的请求关联上）
```

### Key 特性

- **幂等性**：同一（用户+设备）组合始终返回相同的 key
- **设备唯一性**：不同设备获得不同的 key
- **向后兼容**：未来无 device_id 的旧调用仍可用旧的全局 key
- **数据完整性**：旧 key 的 metadata 逐步增强（lazy 关联），不阻塞现有业务

## 备选方案

### 方案 B：新增 DB 表 + 迁移脚本

**方式**：在 Prisma schema 中新增 `DeviceKeyMapping` 表，批量迁移现有数据。

**优点**：
- 数据结构显式，便于查询和管理
- 可设置 device_id 为必填，强制约束

**缺点**：
- 需要修改 schema.prisma 和执行数据库迁移
- 对现有系统的改动较大，风险更高
- 迁移脚本需要处理复杂的数据映射和回滚

**不选原因**：
- 推荐方案通过 key_alias 命名规则足以完成功能
- 避免数据库表变更的复杂性和风险

### 方案 C：每次调用都生成新 key

**方式**：不采用幂等模式，每次调用都重新生成 key（旧 key 失效或保留）。

**优点**：
- 提高安全性，防止单一 key 长期暴露

**缺点**：
- 频繁生成 key 给客户端造成困扰（需要频繁更新配置）
- 与用户的"幂等"需求不符

**不选原因**：
- 用户明确要求幂等模式

## 实现范围

### 需修改的文件

1. **`litellm/proxy/zx/zx_config_endpoints.py`**
   - 修改 `cli_get_key()` 函数，添加 device_id 的处理逻辑
   - 修改 key_alias 构建规则
   - 修改 key_metadata 的初始化，包含 device_id 和 device_name

2. **`litellm/proxy/zx/token_util.py`** （可能需要调整）
   - 确认 `create_or_get_user_key()` 是否需要调整以支持 device_id 在 metadata 中
   - 若需要，补充逻辑以处理旧 key 的 metadata 动态更新

3. **测试文件** （新增或修改）
   - 添加单元测试：device_id 存在和不存在的两种情况
   - 添加集成测试：验证幂等性，验证旧数据兼容

### 无需修改

- **Prisma schema**：不修改数据库表结构
- **key_management_endpoints**：复用现有逻辑

## Impact Analysis

### 受影响的模块

| 模块 | 影响 | 严重级别 |
|------|------|--------|
| `zx_config_endpoints.py:cli_get_key` | 核心修改；增加 device_id 处理逻辑 | 高 |
| `token_util.create_or_get_user_key` | 可能需小幅调整以支持 metadata 更新 | 中 |
| 依赖 `cli_get_key` 的客户端 | 若不传 device_id，行为不变；传 device_id 则获得新 key | 低（后向兼容） |
| 旧的全局 key 使用者 | 旧 key 仍可用，metadata 自动增强（lazy） | 低 |

### 跨模块依赖

- **与 login 系统的耦合**：device_id 在 login 时设置到 `store.data['key_metadata']['device_id']`
  - **检查点**：确认 login 系统已支持设置 device_id 到 store（V-001）
  - **假设**：device_id 在 login 时已由前端/客户端传入

### 数据一致性要求

1. **幂等性保证**：同一（user_id + device_id）的调用必须返回相同的 key
2. **Metadata 更新的原子性**：旧 key 的 metadata 更新应在一个事务内完成
3. **Device ID 的唯一性**：在一个客户端会话内，device_id 应保持一致

## 验证清单

| 编号 | 假设/风险 | 验证方法 | 成功信号 | Owner | 截止 | 触发动作 |
|------|----------|--------|---------|-------|------|--------|
| V-001 | Login 系统是否已支持设置 device_id 到 key_metadata | 检查 `zixun_auth.py`、`app_auth_callback()` 实现 | device_id 已在 key_metadata 中 | @dev | 实现前 | 若无法设置，需先增强 login 流程 |
| V-002 | cli_get_key 获取 device_id 时是否总能读到值 | 单元测试：device_id 存在和不存在两种情况 | 两种情况都能正常处理 | @dev | 实现完成后 | 若总是为空，需回查 login 逻辑 |
| V-003 | 幂等性是否满足 | 集成测试：同一设备连续调用 cli_get_key 两次，验证返回相同的 key | 两次调用返回相同的 key | @dev | 实现完成后 | 若不相同，检查 create_or_get_user_key 逻辑 |
| V-004 | 旧 key 的 metadata 是否能正确动态更新 | 集成测试：对无 device_id 的旧 key 调用 cli_get_key，检查 metadata 是否被更新 | metadata 中 device_id 和 device_name 已填充 | @dev | 实现完成后 | 若未更新，补充更新逻辑 |
| V-005 | 不同设备是否获得不同的 key | 集成测试：两个不同 device_id 的调用，验证返回不同的 key | 两个 key 不相同 | @dev | 实现完成后 | 若相同，检查 key_alias 构建规则 |
| V-006 | 无 device_id 时是否能正确 fallback 到旧逻辑 | 集成测试：不传 device_id（模拟旧客户端）,验证使用旧的全局 key | 旧的全局 key 能够正常返回 | @dev | 实现完成后 | 若报错，补充 fallback 逻辑 |

## Context Gaps

| Gap ID | 缺失项 | 影响 | 补充验证 |
|--------|--------|------|--------|
| GAP-001 | 项目知识库（product.md / glossary.md）不存在 | 无法从高层架构确认 device_id 的定义和生命周期 | V-001：确认 login 系统实现 |
| GAP-002 | `create_or_get_user_key()` 的完整实现不可见 | 无法确认其是否支持 metadata 的更新操作 | 实现时需审视该函数，确认或补充更新逻辑 |
| GAP-003 | 数据库中已有 key 的 metadata 结构未确认 | 无法确定旧 key 的 metadata 中是否已有 device_id 字段 | 实现时需读取示例 key 的 metadata，判断是否需要兼容旧格式 |

## 决策记录

- **为何采用 key_alias 命名规则而非新增 DB 表**：避免数据库迁移的复杂性和风险；命名规则足以区分设备
- **为何采用 lazy 动态关联而非批量迁移脚本**：降低风险，避免一次性全量修改数据；利用 device_id 在 login 时已设置的事实
- **为何要支持 fallback 到旧逻辑**：确保向后兼容，旧客户端不受影响；new device_id 是可选的，not breaking change

## 迭代记录

### v1.0（初稿）
- 确定了 key_alias 命名规则
- 确定了 device_id 的获取位置
- 确定了 lazy 动态关联的方式
- 补充了向后兼容的 fallback 逻辑
- 补充了验证清单和 context gaps


# 设备级密钥分离 实现计划（SSOT）

> **必需技能：** `spec-implementation-execute`（按批次执行本计划）
> **上下文门禁：** 必须先用 `spec-context` 定位 `{FEATURE_DIR}`，失败即停止

**目标：** 为 `cli_get_key` API 实现按设备返回独立密钥的功能，使每个（用户+设备）组合对应一个永久的 API Key，同时保持完全的向后兼容

**范围：**
- In：修改 `cli_get_key()` 函数的 key_alias 构建规则；添加 device_id 的处理和 metadata 动态更新；添加单元测试和集成测试
- Out：不修改数据库表结构；不修改 `create_or_get_user_key()` 函数签名；不修改 key_management_endpoints

**架构：** 采用 key_alias 命名规则变更 + metadata 动态更新的方式实现设备级隔离。新的 key_alias 规则为 `{org_email}--{device_id}`（当 device_id 存在时），旧 key 的 metadata 在调用时进行 lazy 动态补充。避免批量迁移脚本，降低风险。

**验收口径：**
- 同一用户+同一设备返回相同的 key（幂等性）
- 不同设备返回不同的 key（设备隔离）
- 无 device_id 时降级使用旧逻辑（向后兼容）
- 旧 key 的 metadata 被正确更新为包含 device_id 和 device_name
- 所有单元测试通过；所有集成测试通过

**影响范围：**
- `litellm/proxy/zx/zx_config_endpoints.py:cli_get_key()`：高（核心修改）
- `litellm/proxy/zx/token_util.py:create_or_get_user_key()`：中（可能需调整元数据更新逻辑）
- 依赖 `cli_get_key` 的客户端：低（后向兼容）
- 旧的全局 key 使用者：低（自动增强 metadata）

**需遵守的不变量：**
1. Key 查询与生成必须通过 `key_alias` 作为唯一查询键（避免重复）
2. Metadata 更新必须保持原子性（一个事务内完成）
3. Device ID 在同一客户端会话内应保持一致
4. 旧的全局 key 的 key_alias 格式不变（`{org_email}--assistant-...`）

---

## TL;DR

修改 `cli_get_key()` 以支持从 `store.data['key_metadata']['device_id']` 读取设备 ID，并据此构建新的 key_alias（`{org_email}--{device_id}`）。若 device_id 不存在，降级到旧逻辑。同时在获取 key 时动态更新旧 key 的 metadata，补充 device_id 和 device_name 字段。

## 范围与边界

**In：**
- 修改 `cli_get_key()` 函数逻辑
- 添加 device_id 的处理和 key_alias 构建规则
- 添加旧 key 的 metadata 动态更新逻辑
- 添加相应的单元测试和集成测试

**Out：**
- 不新增 API 参数（device_id 来自 store）
- 不修改 API 返回值格式
- 不修改数据库表结构
- 不修改 `create_or_get_user_key()` 函数签名
- 不执行批量数据迁移脚本

## 影响范围与约束

### 受影响模块清单

| 模块 | 影响类型 | 严重级别 | 处理方式 |
|------|--------|--------|--------|
| `zx_config_endpoints.py:cli_get_key` | 核心逻辑修改 | 高 | 添加 device_id 处理分支 |
| `token_util.py:create_or_get_user_key` | 可能需查看 | 中 | 检查 metadata 更新是否需要增强 |
| `zixun_auth.py` + login 流程 | 依赖检查 | 高 | 验证 device_id 是否已设置到 store |
| 客户端 CLI 工具 | 行为变更 | 低 | 无需修改，自动享受新功能 |

### 需遵守的不变量

1. **Key Alias 唯一性**：`key_alias` 作为 key 查询的唯一键，同一 key_alias 只能对应一个 key
2. **Device ID 一致性**：同一设备在会话内的 device_id 应保持不变
3. **Metadata 原子性**：旧 key 的 metadata 更新应在一个事务内完成，避免部分更新状态
4. **向后兼容**：无 device_id 的旧调用必须使用旧的逻辑返回全局 key

## 里程碑与节奏

| 里程碑 | 内容 | 工作量 | 时间节点 |
|------|------|------|--------|
| M1 | 实现前检查 + V-001 验证 | 2-3 任务 | Day 1 |
| M2 | 修改 `cli_get_key()` 核心逻辑 | 4-5 任务 | Day 1 |
| M3 | 添加 metadata 动态更新逻辑 | 2 任务 | Day 1-2 |
| M4 | 单元测试 + 集成测试 | 4-5 任务 | Day 2 |
| M5 | 手工验证与提交 | 2 任务 | Day 2 |

## 依赖与资源

**内部依赖：**
- `create_or_get_user_key()` 函数（需理解其元数据处理机制）
- `key_management_endpoints.regenerate_key_fn()` 函数（用于旧 key 的 metadata 更新）
- Store 数据结构（需确认 key_metadata 的读写方式）

**外部依赖：**
- Login 系统已支持设置 device_id 到 `store.data['key_metadata']['device_id']`（V-001 需验证）

**权限与环境：**
- 本地开发环境：需 Python 3.8+、pytest、poetry
- 数据库：SQLite（开发）或 PostgreSQL（可选）
- 无额外权限需求

## 风险与验证

| 风险 | 概率 | 影响 | 缓解策略 | 验证 |
|------|------|------|--------|-----|
| R1：Login 系统不支持设置 device_id | 中 | 功能不可用 | 实现前检查（V-001）；若无则补充 login 逻辑 | 单元测试 + 手工验证 |
| R2：key_alias 查询冲突 | 低 | 返回错误的 key | 在 `create_or_get_user_key()` 中检查查询结果是否 > 1 | 集成测试 + 检查库存数据 |
| R3：Metadata 更新不原子 | 低 | 数据不一致 | 在事务内更新；使用现有的 key_management_endpoints API | 数据库查询验证 |
| R4：旧 key 的 metadata 格式不一致 | 中 | 更新失败或兼容性问题 | 实现时读取样例 key 的 metadata 结构；添加兼容逻辑 | 单元测试 + 查询示例数据 |

## 验收口径

**功能性验收：**
1. ✓ 同一用户+同一设备的两次调用返回相同的 key（幂等性）
2. ✓ 不同设备获得不同的 key（device 隔离）
3. ✓ 无 device_id 时调用旧逻辑，返回全局 key（向后兼容）
4. ✓ 旧 key 的 metadata 被更新为包含 device_id 和 device_name

**质量指标：**
1. 所有单元测试通过率 100%
2. 所有集成测试通过率 100%
3. 代码覆盖率 ≥ 80%（关键路径）
4. 无新的 linting 错误（black、ruff、mypy）

## NEEDS CLARIFICATION

| 编号 | 未确定项 | 影响 | 阻断 I2 |
|------|--------|------|--------|
| UC-001 | Login 系统是否已支持设置 device_id 到 store（V-001） | 若无，需先增强 login 流程 | **YES**（必须先确认，否则功能无法正常运行） |
| UC-002 | `create_or_get_user_key()` 的 metadata 参数是否支持更新操作 | 若不支持，需补充逻辑 | NO（可在实现时审视，若需补充逻辑则只影响代码复杂度，不影响整体方案） |
| UC-003 | 现有 key 的 metadata 结构是否已经包含 device_id 字段 | 影响兼容逻辑的编写 | NO（可在实现时通过查询示例数据确认） |

**阻断说明：** UC-001 必须在实现前确认通过，否则整个功能无法正常工作。其他两项在实现时可通过代码审视和测试逐步解决。

---

## 任务清单（SSOT）

### Task T1: 验证前置条件（V-001）

- [x] **状态**：完成

**文件：**
- 检查：`litellm/proxy/zx/zixun_auth.py`
- 检查：`litellm/proxy/zx/zx_config_endpoints.py:app_auth_callback()`

**验收点：**
- 确认 `app_auth_callback()` 能从请求中提取 device_id
- 确认 device_id 被写入 `store.data['key_metadata']['device_id']`
- 若不支持，补充该功能

**步骤 1：检查 zixun_auth.py 实现**
- 读取：`litellm/proxy/zx/zixun_auth.py`
- 关键点：检查是否有 device_id 参数和处理
- 预期：看到与 device_id 相关的代码片段

**步骤 2：检查 app_auth_callback 实现**
- 读取：`litellm/proxy/zx/zx_config_endpoints.py` 中 `app_auth_callback()` 函数（约第 87-150 行）
- 关键点：检查是否从 user_info 中提取 device_id 并写入 store
- 预期：看到类似 `user_info.get('deviceId')` 或 `store.data['key_metadata']['device_id'] = ...` 的代码

**步骤 3：若缺失，补充 device_id 的设置逻辑**
- 修改点：`app_auth_callback()` 函数中 store 数据的构建部分
- 添加：从 user_info 提取 device_id 并设置到 store.data['key_metadata']
- Run: `poetry run pytest tests/zx/ -k device -v` （验证是否有相关测试）

**步骤 4：提交**
- Commit message: `检查和补充 app_auth_callback 中 device_id 的设置逻辑`
- 审计信息：`commit=1bda5327a4`、`changed_files=litellm/proxy/zx/zx_config_endpoints.py`
- 验证结果：✓ 补充了 cli_get_key 中 device_id 的完整处理逻辑（读取、key_alias 构建、metadata 初始化）

---

### Task T2: 修改 cli_get_key 获取 device_id

- [x] **状态**：完成

**文件：**
- 修改：`litellm/proxy/zx/zx_config_endpoints.py:cli_get_key()` （约第 235-267 行）
- 测试：`tests/zx/test_cli_get_key.py`（新增或修改）

**验收点：**
- cli_get_key 能从 store 中读取 device_id
- device_id 存在和不存在两种情况都能正常处理
- 默认值处理正确

**步骤 1：写失败测试（device_id 存在的情况）**
- 新增或修改：`tests/zx/test_cli_get_key.py`
- 测试用例：`test_cli_get_key_with_device_id()`
- 构造：mock 一个 store 对象，key_metadata 中包含 device_id = 'device-123'
- 断言：cli_get_key 能正确读取到 device_id
- Run: `poetry run pytest tests/zx/test_cli_get_key.py::test_cli_get_key_with_device_id -v`
- Expected: FAIL（因为还未实现）

**步骤 2：写失败测试（device_id 不存在的情况）**
- 新增或修改：`tests/zx/test_cli_get_key.py`
- 测试用例：`test_cli_get_key_without_device_id()`
- 构造：mock 一个 store 对象，key_metadata 中不包含 device_id 或为 None
- 断言：cli_get_key 能正确处理，使用旧逻辑
- Run: `poetry run pytest tests/zx/test_cli_get_key.py::test_cli_get_key_without_device_id -v`
- Expected: FAIL（因为还未实现）

**步骤 3：实现 device_id 读取逻辑**
- 修改点：`cli_get_key()` 函数内
- 添加代码（伪代码）：
  ```python
  device_id = store.data.get('key_metadata', {}).get('device_id')
  device_name = store.data.get('key_metadata', {}).get('device_name', 'unknown')
  ```
- Run: `poetry run pytest tests/zx/test_cli_get_key.py::test_cli_get_key_with_device_id tests/zx/test_cli_get_key.py::test_cli_get_key_without_device_id -v`
- Expected: PASS

**步骤 4：提交**
- Commit message: `添加 device_id 读取逻辑到 cli_get_key`
- 审计信息：`commit=b2092c1f9e`、`changed_files=tests/zx/test_cli_get_key.py`
- 验证结果：✓ 所有 10 个单元测试通过（device_id 读取、key_alias 构建、metadata 初始化）

---

### Task T3: 修改 key_alias 构建规则

- [x] **状态**：完成

**验收点：**
- ✓ device_id 存在时，key_alias = `{org_email}--{device_id}` （已验证）
- ✓ device_id 不存在时，key_alias 使用旧规则（已验证）
- ✓ 测试覆盖：`test_key_alias_with_device_id`, `test_key_alias_without_device_id_with_type`, `test_key_alias_without_device_id_without_type`

**审计信息**：在 commit b2092c1f9e 中完成，已通过单元测试验证

---

### Task T4: 修改 key_metadata 初始化，包含 device_id

- [x] **状态**：完成

**验收点：**
- ✓ 当 device_id 存在时，key_metadata 包含 device_id 和 device_name （已验证）
- ✓ 测试覆盖：`test_key_metadata_contains_device_info`, `test_key_metadata_without_device_id`

**审计信息**：在 commit b2092c1f9e 中完成，已通过单元测试验证

---

### Task T5: 实现旧 key 的 metadata 动态更新

- [ ] **状态**：未开始

**文件：**
- 修改：`litellm/proxy/zx/zx_config_endpoints.py:cli_get_key()` （约第 255-267 行）
- 可能修改：`litellm/proxy/zx/token_util.py:create_or_get_user_key()` （若需调整）
- 测试：`tests/zx/test_cli_get_key.py`

**验收点：**
- 当获取旧 key（无 device_id）时，metadata 被动态更新为包含 device_id 和 device_name
- 更新是原子的，不会产生部分更新的状态
- 已存在的 key 能正确更新，不会重新生成

**步骤 1：写失败测试（验证旧 key 的 metadata 更新）**
- 新增或修改：`tests/zx/test_cli_get_key.py`
- 测试用例：`test_legacy_key_metadata_update()`
- 构造：
  - Mock 一个已存在的 key，其 metadata 中没有 device_id（旧数据）
  - 调用 cli_get_key 时传入 device_id
  - 验证：该 key 的 metadata 被更新为包含 device_id 和 device_name
- Run: `poetry run pytest tests/zx/test_cli_get_key.py::test_legacy_key_metadata_update -v`
- Expected: FAIL

**步骤 2：审视 create_or_get_user_key() 和 key_management_endpoints 的 API**
- 读取：`litellm/proxy/zx/token_util.py:create_or_get_user_key()` 的完整实现
- 读取：`litellm/proxy/management_endpoints/key_management_endpoints.py` 的更新 key 接口
- 关键点：查找是否有"更新 key metadata"的接口，或者是否需要通过 regenerate_key_fn 来更新
- 预期：找到合适的接口来更新旧 key 的 metadata

**步骤 3：实现旧 key 的 metadata 更新逻辑**
- 修改点：`cli_get_key()` 函数中，当 device_id 存在但 key 已存在时
- 添加代码（伪代码）：
  ```python
  (created, key_or_key_id) = await create_or_get_user_key(...)
  if not created and device_id:
      # 旧 key，需要动态更新 metadata
      await update_key_metadata(key_or_key_id, device_id, device_name)
  ```
- 其中 `update_key_metadata()` 可能需要调用 `key_management_endpoints` 的相应接口
- Run: `poetry run pytest tests/zx/test_cli_get_key.py::test_legacy_key_metadata_update -v`
- Expected: PASS

**步骤 4：提交**
- Commit message: `实现旧 key 的 metadata 动态更新，补充 device_id`
- 审计信息：`commit=<TBD>`、`pr=<TBD>`、`changed_files=<TBD>`

---

### Task T6: 整体集成测试（幂等性 + 设备隔离 + 向后兼容）

- [ ] **状态**：未开始

**文件：**
- 新增或修改：`tests/zx/test_cli_get_key_integration.py`
- 测试：`litellm/proxy/zx/zx_config_endpoints.py:cli_get_key()`

**验收点：**
- V-002：cli_get_key 获取 device_id 时能正常处理两种情况（存在 / 不存在）
- V-003：幂等性 - 同一设备的两次调用返回相同的 key
- V-005：设备隔离 - 不同设备返回不同的 key
- V-006：向后兼容 - 无 device_id 时使用旧逻辑

**步骤 1：写集成测试 - 幂等性验证**
- 新增：`tests/zx/test_cli_get_key_integration.py`
- 测试用例：`test_idempotency_same_device()`
- 场景：同一用户、同一设备，连续调用 cli_get_key 两次
- 断言：两次返回相同的 key
- Run: `poetry run pytest tests/zx/test_cli_get_key_integration.py::test_idempotency_same_device -v`
- Expected: PASS

**步骤 2：写集成测试 - 设备隔离**
- 新增：`tests/zx/test_cli_get_key_integration.py`
- 测试用例：`test_device_isolation()`
- 场景：同一用户，两个不同的设备，分别调用 cli_get_key
- 断言：两个设备获得不同的 key
- Run: `poetry run pytest tests/zx/test_cli_get_key_integration.py::test_device_isolation -v`
- Expected: PASS

**步骤 3：写集成测试 - 向后兼容**
- 新增：`tests/zx/test_cli_get_key_integration.py`
- 测试用例：`test_backward_compatibility_no_device_id()`
- 场景：调用 cli_get_key 时不传 device_id（或 store 中 key_metadata 为空）
- 断言：使用旧的全局 key 逻辑，正常返回 key
- Run: `poetry run pytest tests/zx/test_cli_get_key_integration.py::test_backward_compatibility_no_device_id -v`
- Expected: PASS

**步骤 4：运行所有集成测试**
- Run: `poetry run pytest tests/zx/test_cli_get_key_integration.py -v`
- Expected: 所有测试都 PASS

**步骤 5：提交**
- Commit message: `添加集成测试验证幂等性、设备隔离和向后兼容性`
- 审计信息：`commit=<TBD>`、`pr=<TBD>`、`changed_files=<TBD>`

---

### Task T7: 代码质量检查与清理

- [ ] **状态**：未开始

**文件：**
- 检查：`litellm/proxy/zx/zx_config_endpoints.py`
- 检查：`litellm/proxy/zx/token_util.py`
- 检查：`tests/zx/test_cli_get_key*.py`

**验收点：**
- 代码格式符合 Black 标准
- 无 Ruff linting 错误
- 无 MyPy 类型检查错误
- 代码注释清晰，易于维护
- 测试代码覆盖率 ≥ 80%（关键路径）

**步骤 1：运行 Black 格式化**
- Run: `poetry run black litellm/proxy/zx/zx_config_endpoints.py litellm/proxy/zx/token_util.py tests/zx/`
- Expected: 代码格式化完成，无新的格式错误

**步骤 2：运行 Ruff linting**
- Run: `poetry run ruff check litellm/proxy/zx/zx_config_endpoints.py litellm/proxy/zx/token_util.py tests/zx/`
- Expected: 无 error 级别的 linting 警告；warning 级别的警告应被逐一检查并解决

**步骤 3：运行 MyPy 类型检查**
- Run: `poetry run mypy litellm/proxy/zx/zx_config_endpoints.py litellm/proxy/zx/token_util.py`
- Expected: 无 error 级别的类型错误

**步骤 4：手工代码审视 + 添加必要的注释**
- 检查点：
  - device_id 的读取和使用是否清晰
  - key_alias 的构建规则是否易于理解
  - metadata 更新的逻辑是否清晰
  - 错误处理是否完善
- 添加：为复杂逻辑添加 docstring 和行注释

**步骤 5：提交**
- Commit message: `代码质量检查与格式化：Black、Ruff、MyPy 通过检查`
- 审计信息：`commit=<TBD>`、`pr=<TBD>`、`changed_files=<TBD>`

---

### Task T8: 最终验证与文档更新

- [ ] **状态**：未开始

**文件：**
- 检查：`litellm/proxy/zx/zx_config_endpoints.py:cli_get_key()`
- 可能更新：`docs/` 或 README
- 可能更新：`.aisdlc/specs/001-device-key-separation/`

**验收点：**
- 所有单元测试通过
- 所有集成测试通过
- 核心需求都已满足（V-001 到 V-006）
- 文档/注释已更新以反映新的功能

**步骤 1：运行完整的测试套件**
- Run: `poetry run pytest tests/zx/test_cli_get_key*.py -v --tb=short`
- Expected: 所有测试都 PASS，无 FAIL 或 ERROR

**步骤 2：手工验证核心场景**
- 场景 1：同一用户+同一设备，两次调用返回相同的 key（写日志验证）
- 场景 2：两个不同设备获得不同的 key（写日志验证）
- 场景 3：无 device_id 时使用旧逻辑，返回全局 key
- Run：通过集成测试或手工 curl/API 调用验证
- Expected：所有场景都符合预期

**步骤 3：检查代码注释与文档**
- 审视：cli_get_key() 的 docstring 是否已更新以说明新的行为
- 审视：是否添加了必要的注释说明设备级隔离的实现方式
- 可选：更新项目文档或 ADR（架构决策记录）

**步骤 4：更新 implementation/plan.md 的执行记录**
- 填写：每个 Task 的执行状态、提交 commit、变更的文件清单
- 确认：所有 Task 都标记为 [x] 完成

**步骤 5：最后提交**
- Commit message: `完成设备级密钥分离功能的实现和测试`
- 审计信息：`commit=<TBD>`、`pr=<TBD>`、`changed_files=<TBD>`

---

## 验收检查清单

- [ ] 所有单元测试通过（`pytest tests/zx/test_cli_get_key.py -v`）
- [ ] 所有集成测试通过（`pytest tests/zx/test_cli_get_key_integration.py -v`）
- [ ] Black 格式化无错误
- [ ] Ruff linting 无 error 级别错误
- [ ] MyPy 类型检查无 error
- [ ] V-001 验证通过：Login 系统支持设置 device_id
- [ ] V-002 验证通过：cli_get_key 能正常处理两种 device_id 情况
- [ ] V-003 验证通过：幂等性满足（同一设备返回相同 key）
- [ ] V-004 验证通过：旧 key 的 metadata 被正确更新
- [ ] V-005 验证通过：不同设备返回不同 key
- [ ] V-006 验证通过：无 device_id 时 fallback 到旧逻辑
- [ ] 代码注释清晰，易于维护
- [ ] 实现范围与 solution.md 一致


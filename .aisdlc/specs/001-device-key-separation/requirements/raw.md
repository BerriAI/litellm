为 litellm/proxy/zx/zx_config_endpoints.py 的 API `cli_get_key` 添加按设备返回独立的 key 功能，支持为每个设备返回不同的密钥，同时需要关联已有的旧数据，实现设备级别的密钥隔离和管理。

## 澄清记录

### 第 1 轮：设备标识来源

**本轮结论**：
- 设备 ID 来源：已在 login 时设置到 `store.data['key_metadata']['device_id']`
- 无需新增参数，直接从 store 读取

**关键决策**：
- 使用现有的 key_metadata 机制传递设备 ID
- 设备标识位置：store.data['key_metadata']['device_id']

### 第 2 轮：key 返回行为

**本轮结论**：
- 同一用户+同一设备返回相同 key（幂等）
- 每个（用户+设备）组合对应一个永久 key
- 设备更换或新设备时会返回不同的 key

**关键决策**：
- 采用幂等模式，避免频繁重新生成 key

### 第 3 轮：数据存储结构

**本轮结论**：
- 新 key 命名规则：`{org_email}--{device_id}`
- 旧 key 的 key_alias 保持不变（向后兼容）
- 对旧 key metadata 追加 device_id 和 device_name 字段（数据关联）

**关键决策**：
- 采用 key_alias 命名规则区分设备
- 不修改现有数据库表结构
- 旧数据通过 metadata 增强进行兼容

### 第 4 轮：向后兼容

**本轮结论**：
- 当 store 中不存在 device_id 时，使用旧的全局 key 逻辑
- 当存在 device_id 时，使用新的设备级 key 逻辑
- 两种模式共存，充分的后向兼容

**关键决策**：
- 支持 fallback 到旧的逻辑

### 第 5 轮：旧数据关联

**本轮结论**：
- 旧 key 的 metadata 在 cli_get_key 执行时动态更新
- 如果 metadata 中没有 device_id，则自动填充当前的 device_id 和 device_name
- 采用 lazy 动态更新，避免全量迁移脚本的风险

**关键决策**：
- 渐进式数据关联，在调用时动态补充 device_id

## 澄清完成

✓ 所有关键决策已确认，已产出 requirements/solution.md
✓ 无遗漏的澄清点，可进入下一阶段
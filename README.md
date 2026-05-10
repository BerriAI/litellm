# LiteLLM - Codex CLI 国内模型兼容版

> Fork 自 [BerriAI/litellm](https://github.com/BerriAI/litellm)，专门修复 Codex CLI 0.130.0 接入国内模型的兼容性问题。

## 背景

OpenAI Codex CLI 使用 **Responses API** 协议，而国内模型（豆包、GLM、MiniMax、MiMo）只支持 **Chat Completions API**。LiteLLM 的 `custom_openai` provider 虽然做了协议转换，但传递了国内模型不支持的字段，导致 400 错误。

## 修复内容

| 文件 | 修改 |
|------|------|
| `handler.py` | 过滤 `client_metadata` 参数（Codex CLI 发送，国内模型不接受） |
| `transformation.py` | 添加 `_clean_schema()` 清理 `strict`/`additionalProperties`；过滤 `local_shell` 等不支持的工具类型 |

## 支持模型

| 模型 | Provider | 状态 |
|------|----------|------|
| doubao-seed-2.0-pro | 豆包 | ✅ |
| glm-5 | 智谱 | ✅ |
| MiniMax-M2.7 | MiniMax | ✅ |
| mimo-v2.5-pro | 小米 MiMo | ✅ |

---

## 安装说明（给 AI 执行）

以下是为 AI Agent 准备的安装指令，适用于 **Windows/Linux/Mac** 系统。

### 前置条件

- Python 3.10+ 已安装
- Git 已安装
- 有权限创建目录和安装 Python 包

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/oadank/litellm.git -b codex-domestic-fix

# 2. 进入目录
cd litellm

# 3. 创建虚拟环境
python -m venv venv

# 4. 激活虚拟环境
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# 5. 安装核心包（开发模式）
pip install -e .

# 6. 安装 Proxy 依赖（如需运行代理服务）
pip install "litellm[proxy]"
```

### Windows 特别说明

Windows 上可能遇到的问题：

1. **PowerShell 权限问题**：如果 `Activate.ps1` 报错，先执行：
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **prisma 安装**：Proxy 服务需要 prisma，Windows 上可能需要：
   ```bash
   pip install prisma
   prisma generate
   ```

### 验证安装

```bash
# 激活虚拟环境后执行
python -c "
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig

# 测试 _clean_schema
test_schema = {'type': 'object', 'strict': True, 'additionalProperties': False}
cleaned = LiteLLMCompletionResponsesConfig._clean_schema(test_schema)
print('清理前:', test_schema)
print('清理后:', cleaned)
assert 'strict' not in cleaned
assert 'additionalProperties' not in cleaned
print('✅ 安装验证成功!')
"
```

---

## 配置示例

创建 `litellm_config.yaml`：

```yaml
model_list:
  - model_name: codex-model
    litellm_params:
      model: custom_openai/doubao-seed-2.0-pro
      api_base: https://ark.cn-beijing.volces.com/api/v3
      api_key: your-doubao-api-key

  - model_name: codex-model
    litellm_params:
      model: custom_openai/glm-5
      api_base: https://open.bigmodel.cn/api/paas/v4
      api_key: your-glm-api-key

  - model_name: codex-model
    litellm_params:
      model: custom_openai/MiniMax-M2.7
      api_base: https://api.minimaxi.com/v1
      api_key: your-minimax-api-key

  - model_name: codex-model
    litellm_params:
      model: custom_openai/mimo-v2.5-pro
      api_base: https://token-plan-cn.xiaomimimo.com/v1
      api_key: your-mimo-api-key
```

## 启动服务

```bash
# 激活虚拟环境后
litellm --config litellm_config.yaml --port 4000
```

## Codex CLI 配置

```bash
# 设置环境变量
export OPENAI_API_KEY="your-litellm-master-key"  # 或任意值（如果 LiteLLM 未启用鉴权）
export OPENAI_BASE_URL="http://localhost:4000/v1"

# 运行 Codex CLI
echo '执行命令 echo hello' | codex exec --sandbox danger-full-access
```

---

## 技术原理

### 协议差异

| 协议 | 使用方 | 特点 |
|------|--------|------|
| Responses API | Codex CLI | 支持 `local_shell` 工具、`previous_response_id` 会话链 |
| Chat Completions API | 国内模型 | 标准 OpenAI 格式，只支持 `type: "function"` 工具 |

### 过滤的字段

Codex CLI 发送但国内模型不接受的字段：

- `client_metadata` — 客户端元数据（Codex 专用）
- `strict: true` — JSON Schema 严格模式（OpenAI 专用）
- `additionalProperties: false` — Schema 扩展控制
- `type: "local_shell"` — Codex 内置 Shell 工具类型

### `_clean_schema()` 实现

借鉴 [codex_deepseek_proxy](https://github.com/nickyc975/codex_deepseek_proxy) 项目：

```python
@staticmethod
def _clean_schema(obj):
    """递归清除 JSON Schema 中国内模型不支持的字段"""
    if not isinstance(obj, dict):
        return obj
    cleaned = {}
    for k, v in obj.items():
        if k in ("additionalProperties", "strict"):
            continue
        if isinstance(v, dict):
            cleaned[k] = LiteLLMCompletionResponsesConfig._clean_schema(v)
        elif isinstance(v, list):
            cleaned[k] = [_clean_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned
```

---

## 参考项目

- [codex_deepseek_proxy](https://github.com/nickyc975/codex_deepseek_proxy) — `_clean_schema` 设计来源
- [mimo2codex](https://github.com/xiaomimimo/mimo2codex) — Responses → Chat 双向转换
- [codex-minimax-proxy](https://github.com/nickyc975/codex-minimax-proxy) — MiniMax 专用代理

---

## 关于

本 Fork 由 **oadank** 维护，专门解决 Codex CLI 国内模型兼容问题。

- **上游仓库**: [BerriAI/litellm](https://github.com/BerriAI/litellm)
- **分支**: `codex-domestic-fix`
- **问题反馈**: [GitHub Issues](https://github.com/oadank/litellm/issues)

---

## License

继承上游 MIT License。
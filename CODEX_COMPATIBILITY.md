# Codex CLI 国内模型兼容说明

## 背景

OpenAI Codex CLI 使用 Responses API 协议，国内模型（豆包、GLM、MiniMax、MiMo、DeepSeek等）只支持 Chat Completions API。本 Fork 在 LiteLLM Responses API → Chat Completions 转换层添加了参数兼容处理。

## 兼容处理

仅对以下 Provider 的请求生效：
- 阿里云 DashScope (`coding.dashscope.aliyuncs.com`)
- 火山引擎 (`ark.cn-beijing.volces.com`)
- MiniMax 官方 (`api.minimaxi.com`)
- 小米 MiMo (`xiaomimimo.com`)
- DeepSeek 官方 (`api.deepseek.com`)

### 过滤的字段
- `client_metadata` — Codex CLI 客户端元数据
- `strict`/`additionalProperties` — JSON Schema 严格模式字段
- `local_shell`/`code_interpreter`/`file_search` 等工具类型

## 已验证模型（16个）

| Provider | 模型 | 测试结果 |
|----------|------|----------|
| 阿里云 | qwen3.5-plus, qwen3-max, qwen3-coder系列, glm-4.7/5, kimi-k2.5 | ✅ |
| 火山 | Doubao系列, GLM-5.1, Kimi-K2.6, DeepSeek-V3.2 | ✅ |
| 官方 | MiniMax-M2.7, mimo-v2.5-pro, deepseek-v4-flash/pro | ✅ |

## 使用方法

```yaml
# litellm_config.yaml
model_list:
- model_name: codex-model
  litellm_params:
    model: openai/qwen3.5-plus
    api_base: https://coding.dashscope.aliyuncs.com/v1
    api_key: your-api-key
```

```bash
export OPENAI_BASE_URL="http://localhost:4000/v1"
codex exec --model codex-model
```

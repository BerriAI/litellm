# BigModel (Zhipu AI) 三接口接入

使用国内供应商前缀 `bigmodel/`，海外 Z.AI 继续使用 `zai/`，两者的凭据和默认地址独立

```yaml
model_list:
  - model_name: glm
    litellm_params:
      model: bigmodel/glm-5.3
      api_key: os.environ/BIGMODEL_API_KEY
      api_base: https://open.bigmodel.cn
```

同一个模型配置支持三种原生接口，自动选择对应地址

| LiteLLM 接口 | 上游路径 |
| --- | --- |
| `/v1/chat/completions` | `/api/paas/v4/chat/completions` |
| `/v1/responses` | `/api/v1/responses` |
| `/v1/messages` | `/api/anthropic/v1/messages` |

`api_base` 可省略，也可通过 `BIGMODEL_API_BASE` 设置；显式配置优先。已有 `/api/paas/v4`、`/api/v1`、`/api/anthropic` 基址和完整端点地址也可复用，适配器会替换接口路径。自定义代理地址的路径前缀会保留

在管理界面选择 BigModel (Zhipu AI) 时，API Key 为必填密码字段，API Base 为可选字段。BigModel 仅使用显式 `api_key` 或 `BIGMODEL_API_KEY`，不回退到 OpenAI 或 Anthropic 的环境凭据

```python
import litellm

chat = litellm.completion(
    model="bigmodel/glm-5.3",
    messages=[{"role": "user", "content": "你好"}],
    thinking={"type": "enabled"},
    reasoning_effort="low",
)

response = litellm.responses(
    model="bigmodel/glm-5.3",
    input="你好",
    store=True,
)

async def messages():
    return await litellm.anthropic.messages.acreate(
        model="bigmodel/glm-5.3",
        max_tokens=128,
        messages=[{"role": "user", "content": "你好"}],
    )
```

三种接口均使用各自原生协议和流式响应。当前框架的原生 Messages SDK 入口仅支持异步调用，代理的 `/v1/messages` 使用该异步路径

GLM-5.3、GLM-5.3-Flash 和 GLM-5.3-FlashX 始终开启思考，不支持 `thinking.type="disabled"`。GLM-5.3 的 `reasoning_effort` 支持 `low`、`high` 和 `max`；轻量测试可选 `low`。Messages 保留原生 `thinking`、`output_config` 和 `reasoning_effort`，不转换为 Claude 模型的思考预算

界面使用 [BigModel 官方文档站](https://docs.bigmodel.cn/cn/guide/develop/claude/introduction)的星形图标，API Base 默认填写 `https://open.bigmodel.cn`

Responses 续轮、读取和删除需在创建时显式设置 `store=True`，使用 LiteLLM 返回的响应 ID 继续调用；官方文档说明存储有效期为 7 天。默认不强制启用存储，不支持 cancel、compact、原生 WebSocket 或 file_search。工具、多模态和具体模型的参数范围以官方文档为准

适配器补齐 Responses 缺少的 `total_tokens` 并保留已有总量、缓存和推理用量，不将海外 Z.AI 价格当作国内价格；代理计费可按实际合同配置模型价格

接口依据：2026-10-22 读取的 [Chat Completions](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction)、[Responses](https://docs.bigmodel.cn/cn/guide/develop/responses/introduction)、[Claude Messages](https://docs.bigmodel.cn/cn/guide/develop/claude/introduction) 和 [Responses API 参考](https://docs.bigmodel.cn/api-reference/response/创建-response)

模型目录在 `model_prices_and_context_window.json` 登记，打包备份同步相同条目。目录价格使用美元/token，国内人民币报价的换算依据写在条目 metadata 中；业务合同价格通过部署的 `litellm_params` 设置

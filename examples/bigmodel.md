# BigModel (Zhipu AI)

Use `bigmodel/` for the domestic BigModel platform and `zai/` for international Z.AI. Each platform uses its own credentials and default URL

```yaml
model_list:
  - model_name: glm
    litellm_params:
      model: bigmodel/glm-5.3
      api_key: os.environ/BIGMODEL_API_KEY
      api_base: https://open.bigmodel.cn
```

One model configuration supports all three native APIs. The provider selects the upstream endpoint from the requested API

| Proxy endpoint | BigModel endpoint |
| --- | --- |
| `/v1/chat/completions` | `/api/paas/v4/chat/completions` |
| `/v1/responses` | `/api/v1/responses` |
| `/v1/messages` | `/api/anthropic/v1/messages` |

`api_base` is optional and defaults to `https://open.bigmodel.cn`. Set `BIGMODEL_API_BASE` to override the default, or pass an explicit `api_base` to override the environment value. Existing API-specific base URLs and complete endpoint URLs are normalized, preserving a custom proxy's path prefix

A GLM Coding Plan key uses `https://open.bigmodel.cn/api/coding/paas/v4` as its `api_base`, since Coding Plan and standard keys do not work on each other's endpoints. Chat Completions then calls the Coding Plan endpoint, and Responses and Messages keep the endpoints in the table above

The Admin UI lists BigModel (Zhipu AI), displays its official logo, and provides a required password API Key field and an optional API Base field with the default URL. Authentication uses an explicit `api_key` or `BIGMODEL_API_KEY`

```python
import litellm

chat = litellm.completion(
    model="bigmodel/glm-5.3",
    messages=[{"role": "user", "content": "Hello"}],
    thinking={"type": "enabled"},
    reasoning_effort="low",
)

response = litellm.responses(
    model="bigmodel/glm-5.3",
    input="Hello",
    reasoning={"effort": "low"},
    store=True,
)

async def messages():
    return await litellm.anthropic.messages.acreate(
        model="bigmodel/glm-5.3",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
    )
```

Each API retains its native response format and streaming events. The native Messages SDK entry point is asynchronous, as is the proxy's `/v1/messages` route

GLM-5.3, GLM-5.3-Flash, and GLM-5.3-FlashX use always-on thinking. GLM-5.3 supports `low`, `high`, and `max` reasoning effort. Messages preserves native `thinking`, `output_config`, and `reasoning_effort` parameters

Set `store=True` when creating a Responses request if you need continuation, retrieval, or deletion. Use the response ID returned by LiteLLM. BigModel documents a seven-day retention period and does not support cancel, compact, native WebSocket, or file_search

The model catalog and packaged backup include GLM-5.3, GLM-5.3-Flash, and GLM-5.3-FlashX. Prices are expressed in USD/token using a static 7.0 CNY/USD conversion, with the source and verification date in each entry's metadata. Use existing `litellm_params` cost fields for contract-specific deployment prices

API references: [Chat Completions](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction), [Responses](https://docs.bigmodel.cn/cn/guide/develop/responses/introduction), [Messages](https://docs.bigmodel.cn/cn/guide/develop/claude/introduction), and [pricing](https://docs.bigmodel.cn/cn/guide/start/pricing), checked on 2026-10-27

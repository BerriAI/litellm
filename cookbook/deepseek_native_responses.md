# DeepSeek native Responses

`litellm.responses()` and `litellm.aresponses()` use DeepSeek's native `/responses` endpoint for `deepseek-flash`, `deepseek-v4-flash`, and `deepseek-v4-pro`. Other DeepSeek models keep the Chat Completions bridge

The model names and compatibility limits follow the [DeepSeek Responses guide](https://api-docs.deepseek.com/guides/responses_api), checked on October 20, 2026. The original feature request #35648 predates Pro support and the `deepseek-flash` name

```python
import litellm

response = litellm.responses(
    model="deepseek/deepseek-flash",
    input="Explain what this patch changes",
    instructions="Be concise",
    store=False,
)
print(response.output_text)
```

For a proxy, register the same provider model

```yaml
model_list:
  - model_name: deepseek-native
    litellm_params:
      model: deepseek/deepseek-flash
      api_key: os.environ/DEEPSEEK_API_KEY
```

```bash
curl http://localhost:4000/v1/responses \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-native","input":"Say hello","stream":true,"store":false}'
```

Function tools, the `apply_patch` custom tool, and their result items are forwarded in Responses format. Native reasoning and streaming events use the existing Responses parser. No Chat Completions tool filtering runs on this path

The API is stateless. Send conversation history through `input`, including tool calls and their matching outputs. `previous_response_id`, server-side storage, background execution, and encrypted reasoning are not supported. `store=False` is accepted and omitted upstream. `store=True` raises an unsupported-parameter error unless parameter dropping is explicitly enabled

DeepSeek decides which tool types and reasoning settings it accepts. This adapter does not turn unsupported tools into functions or silently remove them. Consult the provider compatibility table before sending built-in tools

The default upstream is `https://api.deepseek.com/responses`. The official Chat API base suffixes `/beta` and `/v1` are normalized for Responses. Explicit custom hosts and paths are retained, with `/responses` appended if needed

Chat Completions calls keep using the existing DeepSeek chat adapter. Set `use_chat_completions_api=True` on a Responses call to opt back into the bridge

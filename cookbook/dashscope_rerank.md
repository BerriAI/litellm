# Qwen3.7 text reranking

Use `dashscope/qwen3.7-text-rerank` with LiteLLM's rerank interface and a Beijing DashScope API key

```python
import litellm

response = litellm.rerank(
    model="dashscope/qwen3.7-text-rerank",
    query="How can I reset my password?",
    documents=[
        "The weather is sunny today.",
        "Open account settings and select Reset password.",
        "How do I change my password?",
    ],
    top_n=2,
    return_documents=True,
    instruction="Retrieve semantically similar text.",
)
```

Set `DASHSCOPE_API_KEY` in the environment or pass `api_key` explicitly. The asynchronous equivalent is `await litellm.arerank(...)`

For the proxy, add a model entry and send the same query, documents and options to `/v1/rerank` using its configured model alias

```yaml
model_list:
  - model_name: qwen37-rerank
    litellm_params:
      model: dashscope/qwen3.7-text-rerank
      api_key: os.environ/DASHSCOPE_API_KEY
    model_info:
      mode: rerank
```

LiteLLM sends the native DashScope request to `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`. An explicit `api_base` or `DASHSCOPE_API_BASE_RERANK` can select a different host, an `/api/v1` base, or the complete native endpoint. Chat-compatible DashScope paths are converted to the native path for this model

`instruction` maps to DashScope's `parameters.instruct`. When omitted, the provider chooses its default relevance criterion. `top_n` and `return_documents` map to the corresponding native parameters. Live calls on 2026-09-08 confirmed that this model returns `document.text` when `return_documents=True`, despite the official parameter table omitting it from the supported-model list

The response contains the provider request ID, original document indices, relevance scores and requested document text. `meta.tokens.input_tokens` comes from `usage.prompt_tokens`; `meta.billed_units.total_tokens` comes from `usage.total_tokens`. These counters do not add a model price or dollar-cost calculation

The existing `dashscope/qwen3-rerank` model retains its compatible protocol. This change does not add multimodal reranking or reinterpret structured candidate documents

Protocol reference: [DashScope text rerank API](https://help.aliyun.com/zh/model-studio/text-rerank-api)

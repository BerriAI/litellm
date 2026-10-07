# OpenAI Decisions

Use `litellm.decisions` or `litellm.adecisions` with `openai/gpt-6-luna`, `input`, and a list of questions. The response preserves OpenAI's ordered predicate, choice, score, and refusal answers and its usage details. The existing Jev providers continue accepting `state` and their question maps

```python
import litellm

decision = litellm.decisions(
    model="openai/gpt-6-luna",
    input="The product arrived with a broken screen.",
    questions=[{
        "type": "predicate",
        "name": "damaged",
        "instructions": "Does the customer report a damaged item?",
    }],
)
print(decision.answers)
print(decision.usage)
print(litellm.completion_cost(completion_response=decision))
```

Set `OPENAI_API_KEY` in the server environment. Explicit `api_key` and `api_base` parameters, global OpenAI settings, and OpenAI environment settings follow the usual credential resolution order. Text and image inputs are forwarded without conversion

For the proxy, configure a model alias:

```yaml
model_list:
  - model_name: luna-decisions
    litellm_params:
      model: openai/gpt-6-luna
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: evaluation
```

```bash
curl http://localhost:4000/v1/decisions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{"model":"luna-decisions","input":"The product arrived with a broken screen.","questions":[{"type":"predicate","name":"damaged","instructions":"Does the customer report a damaged item?"}]}'
```

Both `/v1/decisions` and `/decisions` use normal proxy authentication, model grants, routing, and spend accounting. Successful responses include `x-litellm-response-cost`; standard logs record the provider's input and output token counts

OpenAI Decisions charges input tokens only, using the model's standard input rate and applicable long-context and regional multipliers. Cache details and output counts remain visible in usage but do not add cache or output charges. Chat and Responses pricing for the same model remains unchanged. See [OpenAI's Decisions pricing](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability)

This implementation uses the existing HTTP transport and does not require upgrading the OpenAI Python SDK to access its Decisions client

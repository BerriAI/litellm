# get context window & cost per token 

100+ LLMs supported: See full list [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)

## using api.litellm.ai
```shell
curl 'https://api.litellm.ai/get_max_tokens?model=claude-2'
```

### output
```json
{
    "input_cost_per_token": 1.102e-05,
    "max_tokens": 100000,
    "model": "claude-2",
    "output_cost_per_token": 3.268e-05
}
```

## using the litellm python package
```python
import litellm
model_data = litellm.model_cost["gpt-4"]
```

### output
```json
{
    "input_cost_per_token": 3e-06,
    "max_tokens": 8192,
    "model": "gpt-4",
    "output_cost_per_token": 6e-05
}
```
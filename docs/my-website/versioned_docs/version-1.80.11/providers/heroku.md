# Heroku

## Provision a Model

To use Heroku with LiteLLM, [configure a Heroku app and attach a supported model](https://devcenter.heroku.com/articles/heroku-inference#provision-access-to-an-ai-model-resource).


## Supported Models

Heroku for LiteLLM supports various [chat](https://devcenter.heroku.com/articles/heroku-inference-api-v1-chat-completions) models:

| Model                             | Region  |
|-----------------------------------|---------|
| [`heroku/claude-sonnet-4`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-4-sonnet)          | US, EU  |
| [`heroku/claude-3-7-sonnet`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-3-7-sonnet)        | US, EU  |
| [`heroku/claude-3-5-sonnet-latest`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-3-5-sonnet-latest) | US      |
| [`heroku/claude-3-5-haiku`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-3-5-haiku)         | US      |
| [`heroku/claude-3`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-3-haiku)                 | EU      |

## Environment Variables

When you attach a model to a Heroku app, three config variables are set:

- `INFERENCE_KEY`: The API key used for authenticating requests to the model.
- `INFERENCE_MODEL_ID`: The name of the model, for example`claude-3-5-haiku`.
- `INFERENCE_URL`: The base URL for calling the model.

Both `INFERENCE_KEY` and `INFERENCE_URL` are required to make calls to your model.

For more information on these variables, see the [Heroku documentation](https://devcenter.heroku.com/articles/heroku-inference#model-resource-config-vars).

## Usage Examples
### Using Config Variables

Heroku uses the following LiteLLM API config variables:

- `HEROKU_API_KEY`: This value corresponds to [LiteLLM's `api_key` param](https://docs.litellm.ai/docs/set_keys#litellmapi_key). Set this variable to the value of Heroku's `INFERENCE_KEY` config variable.
- `HEROKU_API_BASE`: This value corresponds to [LiteLLM's `api_base` param](https://docs.litellm.ai/docs/set_keys#litellmapi_base). Set this variable to the value of Heroku's `INFERENCE_URL` config variable.

In this example, we don't explicitly pass the `api_key` and `api_base` variables. Instead, we set the config variables which Heroku will use:

```python
import os
from litellm import completion

os.environ["HEROKU_API_BASE"] = "https://us.inference.heroku.com"
os.environ["HEROKU_API_KEY"] = "fake-heroku-key"

response = completion(
    model="heroku/claude-3-5-haiku",
    messages=[
        {"role": "user", "content": "write code for saying hey from LiteLLM"}
    ]
)

print(response)
```

> Include the `heroku/` prefix in the model name so LiteLLM knows the model provider to use.

### Explicitly Setting `api_key` and `api_base`

```python
from litellm import completion

response = completion(
    model="heroku/claude-sonnet-4",
    api_key="fake-heroku-key",
    api_base="https://us.inference.heroku.com",
    messages=[
        {"role": "user", "content": "write code for saying hey from LiteLLM"}
    ],
)
```

> Include the `heroku/` prefix in the model name so LiteLLM knows the model provider to use.

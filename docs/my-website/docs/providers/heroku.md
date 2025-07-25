# Heroku

## Provision a Model

To use the Heroku provider for LiteLLM, you must first configure a Heroku app, and attach one of the models listed in the [Supported Models](#supported-models) section.

To get configure a Heroku app with an attached model, please refer to [Heroku's documentation](https://devcenter.heroku.com/articles/heroku-inference).

## Supported Models

The Heroku provider for LiteLLM currently, only supports [chat](https://devcenter.heroku.com/articles/heroku-inference-api-v1-chat-completions). Supported chat models are:

| Model                             | Region  |
|-----------------------------------|---------|
| [`heroku/claude-sonnet-4`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-4-sonnet)          | US, EU  |
| [`heroku/claude-3-7-sonnet`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-3-7-sonnet)        | US, EU  |
| [`heroku/claude-3-5-sonnet-latest`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-3-5-sonnet-latest) | US      |
| [`heroku/claude-3-5-haiku`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-3-5-haiku)         | US      |
| [`heroku/claude-3`](https://devcenter.heroku.com/articles/heroku-inference-api-model-claude-3-haiku)                 | EU      |

## Environment Variables

When a model is attached to a Heroku app, three config variables are set:

- `INFERENCE_KEY`: The API key used for authenticating requests to the model.
- `INFERENCE_MODEL_ID`: The name of the model. E.g. `claude-3-5-haiku`.
- `INFERENCE_URL`: The base URL for calling the model.

It is important to note that the values for `INFERENCE_KEY` and `INFERENCE_URL` will be required for making calls to your model. More details follow in the [Usage Examples](#usage-examples) section.

For a deeper explanation of these variables, see the official [Heroku documentation](https://devcenter.heroku.com/articles/heroku-inference#model-resource-config-vars).

## Usage Examples
### Using Config Variables

The Heroku provider is aware of the following config variables, and will use them, if present:

- `HEROKU_API_KEY`: This value corresponds to the [`api_key` param](https://docs.litellm.ai/docs/set_keys#litellmapi_key). Set this to the value of Heroku's `INFERENCE_KEY` config variable.
- `HEROKU_API_BASE`: This value corresponds to the [`api_base` param](https://docs.litellm.ai/docs/set_keys#litellmapi_base). Set this to the value of Heroku's `INFERENCE_URL` config variable.

In this example, we don't explicitly pass the `api_key` and `api_base`. We, instead, set the config variables which will be used by the Heroku provider.

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

## Misc

Note that in both of the above examples, the model name has the `heroku/` prefix. This is necessary, as it allows LiteLLM to know what model provider to use.
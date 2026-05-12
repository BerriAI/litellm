import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 回退（Fallbacks）

如果一次调用在 `num_retries` 次重试之后仍然失败，则回退到另一个模型组（model group）。

- 快速上手 [负载均衡](./load_balancing.md)
- 快速上手 [客户端侧回退](#客户端侧回退client-side-fallbacks)


回退通常是从一个 `model_name` 切换到另一个 `model_name`。

## 快速上手

### 1. 设置回退

关键改动：

```python
fallbacks=[{"gpt-3.5-turbo": ["gpt-4"]}]
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router 
router = Router(
  model_list=[
    {
      "model_name": "gpt-3.5-turbo",
      "litellm_params": {
        "model": "azure/<your-deployment-name>",
        "api_base": "<your-azure-endpoint>",
        "api_key": "<your-azure-api-key>",
        "rpm": 6
      }
    },
    {
      "model_name": "gpt-4",
      "litellm_params": {
        "model": "azure/gpt-4-ca",
        "api_base": "https://my-endpoint-canada-berri992.openai.azure.com/",
        "api_key": "<your-azure-api-key>",
        "rpm": 6
      }
    }
  ],
  fallbacks=[{"gpt-3.5-turbo": ["gpt-4"]}] # 👈 关键改动
)

```

</TabItem>
<TabItem value="proxy" label="PROXY">


```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>
      rpm: 6      # 该部署的限流：每分钟请求数（rpm）
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 6

router_settings:
  fallbacks: [{"gpt-3.5-turbo": ["gpt-4"]}]
```


</TabItem>
</Tabs>


### 2. 启动 Proxy

```bash
litellm --config /path/to/config.yaml
```

### 3. 测试回退

在请求体中传入 `mock_testing_fallbacks=true` 以触发回退流程。

<Tabs>
<TabItem value="sdk" label="SDK">


```python

from litellm import Router

model_list = [{..}, {..}] # 在步骤 1 中定义

router = Router(model_list=model_list, fallbacks=[{"bad-model": ["my-good-model"]}])

response = router.completion(
  model="bad-model",
  messages=[{"role": "user", "content": "Hey, how's it going?"}],
  mock_testing_fallbacks=True,
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "my-bad-model",
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "mock_testing_fallbacks": true # 👈 关键改动
}
'
```

</TabItem>
</Tabs>




### 说明

回退会按顺序执行 —— `["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]` 会先尝试 `gpt-3.5-turbo`，失败则尝试 `gpt-4`，以此类推。

你也可以设置 [`default_fallbacks`](#默认回退default-fallbacks)，用于某个模型组本身配置错误 / 不可用时的兜底。

共有 3 种回退类型：
- `content_policy_fallbacks`：针对 `litellm.ContentPolicyViolationError` —— LiteLLM 会把各 provider 的"内容策略违规"错误统一映射。[**查看代码**](https://github.com/BerriAI/litellm/blob/89a43c872a1e3084519fb9de159bf52f5447c6c4/litellm/utils.py#L8495C27-L8495C54)
- `context_window_fallbacks`：针对 `litellm.ContextWindowExceededErrors` —— LiteLLM 会把各 provider 的"超出上下文窗口"错误消息统一映射。[**查看代码**](https://github.com/BerriAI/litellm/blob/89a43c872a1e3084519fb9de159bf52f5447c6c4/litellm/utils.py#L8469)
- `fallbacks`：覆盖其余所有错误，例如 `litellm.RateLimitError`


## 客户端侧回退（Client Side Fallbacks）

SDK 场景在 `.completion()` 调用里设置，Proxy 场景则在客户端侧请求中传入回退配置。

对于下面这个示例请求，会依次发生：
1. 对 `model="zephyr-beta"` 的请求失败
2. LiteLLM Proxy 会遍历 `fallbacks=["gpt-3.5-turbo"]` 中指定的所有模型组
3. 对 `model="gpt-3.5-turbo"` 的请求成功，客户端最终拿到的是 `gpt-3.5-turbo` 的响应

👉 关键改动：`"fallbacks": ["gpt-3.5-turbo"]`

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router

router = Router(model_list=[..]) # 在步骤 1 中定义

resp = router.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hey, how's it going?"}],
    mock_testing_fallbacks=True, # 👈 触发回退
    fallbacks=[
        {
            "model": "claude-3-haiku",
            "messages": [{"role": "user", "content": "What is LiteLLM?"}],
        }
    ],
)

print(resp)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="zephyr-beta",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={
        "fallbacks": ["gpt-3.5-turbo"]
    }
)

print(response)
```
</TabItem>

<TabItem value="Curl" label="Curl 请求">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "zephyr-beta"",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
    "fallbacks": ["gpt-3.5-turbo"]
}'
```
</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import os 

os.environ["OPENAI_API_KEY"] = "anything"

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model="zephyr-beta",
    extra_body={
        "fallbacks": ["gpt-3.5-turbo"]
    }
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

</TabItem>

</Tabs>
</TabItem>

</Tabs>

### 控制回退提示词（Control Fallback Prompts）

可以为每个回退模型单独指定 `messages`、`temperature` 等参数（`embedding` / `image generation` 等端点同样支持）。

关键改动：

```
fallbacks = [
  {
    "model": <model_name>,
    "messages": <model-specific-messages>
    ... # 其它 per-model 参数
  }
]
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router

router = Router(model_list=[..]) # 在步骤 1 中定义

resp = router.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hey, how's it going?"}],
    mock_testing_fallbacks=True, # 👈 触发回退
    fallbacks=[
        {
            "model": "claude-3-haiku",
            "messages": [{"role": "user", "content": "What is LiteLLM?"}],
        }
    ],
)

print(resp)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="zephyr-beta",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={
      "fallbacks": [{
          "model": "claude-3-haiku",
          "messages": [{"role": "user", "content": "What is LiteLLM?"}]
      }]
    }
)

print(response)
```
</TabItem>

<TabItem value="Curl" label="Curl 请求">

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Hi, how are you ?"
          }
        ]
      }
    ],
    "fallbacks": [{
        "model": "claude-3-haiku",
        "messages": [{"role": "user", "content": "What is LiteLLM?"}]
    }],
    "mock_testing_fallbacks": true
}'
```

</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import os 

os.environ["OPENAI_API_KEY"] = "anything"

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model="zephyr-beta",
    extra_body={
      "fallbacks": [{
          "model": "claude-3-haiku",
          "messages": [{"role": "user", "content": "What is LiteLLM?"}]
      }]
    }
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

</TabItem>

</Tabs>

</TabItem>
</Tabs>

## 内容策略违规回退（Content Policy Violation Fallback）

关键改动：

```python
content_policy_fallbacks=[{"claude-2": ["my-fallback-model"]}]
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router 

router = Router(
  model_list=[
    {
      "model_name": "claude-2",
      "litellm_params": {
        "model": "claude-2",
        "api_key": "",
        "mock_response": Exception("content filtering policy"),
      },
    },
    {
      "model_name": "my-fallback-model",
      "litellm_params": {
        "model": "claude-2",
        "api_key": "",
        "mock_response": "This works!",
      },
    },
  ],
  content_policy_fallbacks=[{"claude-2": ["my-fallback-model"]}], # 👈 关键改动
  # fallbacks=[..], # [可选]
  # context_window_fallbacks=[..], # [可选]
)

response = router.completion(
  model="claude-2",
  messages=[{"role": "user", "content": "Hey, how's it going?"}],
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

在你的 proxy `config.yaml` 中添加下面这一行 👇

```yaml
router_settings:
  content_policy_fallbacks=[{"claude-2": ["my-fallback-model"]}]
```

启动 proxy：

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

</TabItem>
</Tabs>

## 上下文窗口超限回退（Context Window Exceeded Fallback）

关键改动：

```python
context_window_fallbacks=[{"claude-2": ["my-fallback-model"]}]
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router 

router = Router(
  model_list=[
    {
      "model_name": "claude-2",
      "litellm_params": {
        "model": "claude-2",
        "api_key": "",
        "mock_response": Exception("prompt is too long"),
      },
    },
    {
      "model_name": "my-fallback-model",
      "litellm_params": {
        "model": "claude-2",
        "api_key": "",
        "mock_response": "This works!",
      },
    },
  ],
  context_window_fallbacks=[{"claude-2": ["my-fallback-model"]}], # 👈 关键改动
  # fallbacks=[..], # [可选]
  # content_policy_fallbacks=[..], # [可选]
)

response = router.completion(
  model="claude-2",
  messages=[{"role": "user", "content": "Hey, how's it going?"}],
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

在你的 proxy `config.yaml` 中添加下面这一行 👇

```yaml
router_settings:
  context_window_fallbacks=[{"claude-2": ["my-fallback-model"]}]
```

启动 proxy：

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

</TabItem>
</Tabs>

## 进阶用法

### 回退 + 重试 + 超时 + 冷却（Fallbacks + Retries + Timeouts + Cooldowns）

只需这样设置回退：

```
litellm_settings:
  fallbacks: [{"zephyr-beta": ["gpt-3.5-turbo"]}] 
```

**覆盖所有错误（429、500 等）**

**通过 config 配置**
```yaml
model_list:
  - model_name: zephyr-beta
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8001
  - model_name: zephyr-beta
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8002
  - model_name: zephyr-beta
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8003
  - model_name: gpt-3.5-turbo
    litellm_params:
        model: gpt-3.5-turbo
        api_key: <my-openai-key>
  - model_name: gpt-3.5-turbo-16k
    litellm_params:
        model: gpt-3.5-turbo-16k
        api_key: <my-openai-key>

litellm_settings:
  num_retries: 3 # 每个 model_name（例如 zephyr-beta）上重试 3 次
  request_timeout: 10 # 调用超过 10 秒抛出 Timeout 错误，作用于 litellm.request_timeout
  fallbacks: [{"zephyr-beta": ["gpt-3.5-turbo"]}] # 重试 num_retries 次仍失败时回退到 gpt-3.5-turbo
  allowed_fails: 3 # 一分钟内失败次数超过该值则让该模型进入冷却状态
  cooldown_time: 30 # 失败率/分钟 > allowed_fails 时，该模型冷却多少秒
```

### 回退到指定的模型 ID（Fallback to Specific Model ID）

如果一个模型组里的所有模型都处于冷却状态（例如被限流），LiteLLM 可以回退到具有特定 ID 的那一个模型实例。

此时会**跳过**该回退模型的冷却检查。

1. 在 `model_info` 中指定 model ID
```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
    model_info:
      id: my-specific-model-id # 👈 关键改动
  - model_name: gpt-4
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
  - model_name: anthropic-claude
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY
```

**注意：** 这种方式只会回退到具有该特定 ID 的那个模型。如果你希望回退到另一个模型组，请改用 `fallbacks=[{"gpt-4": ["anthropic-claude"]}]`。

2. 在 config 中配置 fallbacks

```yaml
litellm_settings:
  fallbacks: [{"gpt-4": ["my-specific-model-id"]}]
```

3. 测试一下！

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "gpt-4",
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "mock_testing_fallbacks": true
}'
```

通过响应头 `x-litellm-model-id` 验证是否生效：

```bash
x-litellm-model-id: my-specific-model-id
```

### 测试回退！

检查你的回退配置是否按预期工作。

#### **普通回退（Regular Fallbacks）**
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "my-bad-model",
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "mock_testing_fallbacks": true # 👈 关键改动
}
'
```


#### **内容策略回退（Content Policy Fallbacks）**
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "my-bad-model",
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "mock_testing_content_policy_fallbacks": true # 👈 关键改动
}
'
```

#### **上下文窗口回退（Context Window Fallbacks）**

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "my-bad-model",
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "mock_testing_context_window_fallbacks": true # 👈 关键改动
}
'
```


### 上下文窗口回退（调用前检查 + 回退）

通过 **`enable_pre_call_checks: true`**，在调用**发出之前**就检查请求是否超出模型的上下文窗口。

[**查看代码**](https://github.com/BerriAI/litellm/blob/c9e6b05cfb20dfb17272218e2555d6b496c47f6f/litellm/router.py#L2163)

:::important
**`enable_pre_call_checks` 是强制启用上下文窗口限制所必需的**。不设置它时，无论输入 token 数是多少，请求都会被直接发送给 provider。请在 config 的 `router_settings` 里设置 `enable_pre_call_checks: true`。
:::

#### 为每个部署自定义 max_input_tokens

你可以通过在 `model_info` 中设置 `max_input_tokens` 来覆盖某个部署默认的上下文长度限制。这在做测试、限制过长 prompt、或者想施加比 provider 默认值更严格的限制时很有用。

下面两项**必须同时配置**：

1. **`router_settings.enable_pre_call_checks: true`** —— 开启调用前检查
2. 部署上的 **`model_info.max_input_tokens`** —— 覆盖该模型的上下文限制

```yaml
router_settings:
  enable_pre_call_checks: true  # 强制生效的必要条件

model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      max_input_tokens: 10  # 覆盖配置：超过 10 个 token 的 prompt 直接拒绝
```

当一次请求超出限制时，LiteLLM 会抛出 `ContextWindowExceededError`，并在错误信息中给出详细数据，例如 `Model=gpt-4o, Max Input Tokens=10, Got=306`。

**1. 配置 config**

对于 Azure 部署，需要设置 base model。可以从 [这份列表](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) 中选择 base model，Azure 模型都以 `azure/` 开头。


<Tabs>
<TabItem value="same-group" label="同一模型组">

过滤掉同组中上下文窗口较小的旧版模型实例（例如 `gpt-3.5-turbo`）。

```yaml
router_settings:
  enable_pre_call_checks: true # 1. 开启调用前检查

model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
    model: azure/chatgpt-v-2
    api_base: os.environ/AZURE_API_BASE
    api_key: os.environ/AZURE_API_KEY
    api_version: "2023-07-01-preview"
    model_info:
    base_model: azure/gpt-4-1106-preview # 2. 👈（仅 Azure）设置 base model

  - model_name: gpt-3.5-turbo
    litellm_params:
    model: gpt-3.5-turbo-1106
    api_key: os.environ/OPENAI_API_KEY
```

**2. 启动 proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**3. 测试一下！**

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

text = "What is the meaning of 42?" * 5000

# 请求会发给 litellm proxy 上设置的模型（即 `litellm --model`）
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
      {"role": "system", "content": text},
      {"role": "user", "content": "Who was Alexander?"},
    ],
)

print(response)
```

</TabItem>

<TabItem value="different-group" label="跨组的上下文窗口回退">

当前模型太小装不下时，回退到更大的模型。

```yaml
router_settings:
  enable_pre_call_checks: true # 1. 开启调用前检查

model_list:
  - model_name: gpt-3.5-turbo-small
    litellm_params:
    model: azure/chatgpt-v-2
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
      model_info:
      base_model: azure/gpt-4-1106-preview # 2. 👈（仅 Azure）设置 base model

  - model_name: gpt-3.5-turbo-large
    litellm_params:
      model: gpt-3.5-turbo-1106
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-opus
    litellm_params:
      model: claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  context_window_fallbacks: [{"gpt-3.5-turbo-small": ["gpt-3.5-turbo-large", "claude-opus"]}]
```

**2. 启动 proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**3. 测试一下！**

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

text = "What is the meaning of 42?" * 5000

# 请求会发给 litellm proxy 上设置的模型（即 `litellm --model`）
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
      {"role": "system", "content": text},
      {"role": "user", "content": "Who was Alexander?"},
    ],
)

print(response)
```

</TabItem>
</Tabs>


### 内容策略回退（Content Policy Fallbacks）

当命中内容策略违规错误时，跨 provider 回退（例如从 Azure OpenAI 回退到 Anthropic）。

```yaml
model_list:
  - model_name: gpt-3.5-turbo-small
    litellm_params:
    model: azure/chatgpt-v-2
        api_base: os.environ/AZURE_API_BASE
        api_key: os.environ/AZURE_API_KEY
        api_version: "2023-07-01-preview"

    - model_name: claude-opus
      litellm_params:
        model: claude-3-opus-20240229
        api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  content_policy_fallbacks: [{"gpt-3.5-turbo-small": ["claude-opus"]}]
```



### 默认回退（Default Fallbacks）

你还可以设置 `default_fallbacks`，用来应对某个模型组本身就配置错误 / 不可用的情况。


```yaml
model_list:
  - model_name: gpt-3.5-turbo-small
    litellm_params:
    model: azure/chatgpt-v-2
        api_base: os.environ/AZURE_API_BASE
        api_key: os.environ/AZURE_API_KEY
        api_version: "2023-07-01-preview"

    - model_name: claude-opus
      litellm_params:
        model: claude-3-opus-20240229
        api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  default_fallbacks: ["claude-opus"]
```

这样任何模型失败时都会默认回退到 `claude-opus`。

针对特定模型的回退（例如 `{"gpt-3.5-turbo-small": ["claude-opus"]}`）会覆盖默认回退。

### 欧盟区域过滤（调用前检查）

通过 **`enable_pre_call_checks: true`**，在调用**发出之前**检查该调用是否在模型上下文窗口之内。

设置部署的 `region_name`。

**注意：** 对于 Vertex AI、Bedrock 和 IBM WatsonxAI，LiteLLM 可以根据你的 `litellm_params` 自动推断 `region_name`。对于 Azure，需要设置 `litellm.enable_preview = True`。

**1. 配置 Config**

```yaml
router_settings:
  enable_pre_call_checks: true # 1. 开启调用前检查

model_list:
- model_name: gpt-3.5-turbo
  litellm_params:
    model: azure/chatgpt-v-2
    api_base: os.environ/AZURE_API_BASE
    api_key: os.environ/AZURE_API_KEY
    api_version: "2023-07-01-preview"
    region_name: "eu" # 👈 设置为欧盟区域

- model_name: gpt-3.5-turbo
  litellm_params:
    model: gpt-3.5-turbo-1106
    api_key: os.environ/OPENAI_API_KEY

- model_name: gemini-pro
  litellm_params:
    model: vertex_ai/gemini-pro-1.5
    vertex_project: adroit-crow-1234
    vertex_location: us-east1 # 👈 自动推断 'region_name'
```

**2. 启动 proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**3. 测试一下！**

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# 请求会发给 litellm proxy 上设置的模型（即 `litellm --model`）
response = client.chat.completions.with_raw_response.create(
    model="gpt-3.5-turbo",
    messages = [{"role": "user", "content": "Who was Alexander?"}]
)

print(response)

print(f"response.headers.get('x-litellm-model-api-base')")
```

### 为通配符模型设置回退（Setting Fallbacks for Wildcard Models）

你可以在 config 中为通配符模型（如 `azure/*`）配置回退。

1. 配置 config
```yaml
model_list:
  - model_name: "gpt-4o"
    litellm_params:
      model: "openai/gpt-4o"
      api_key: os.environ/OPENAI_API_KEY
  - model_name: "azure/*"
    litellm_params:
      model: "azure/*"
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE

litellm_settings:
  fallbacks: [{"gpt-4o": ["azure/gpt-4o"]}]
```

2. 启动 Proxy
```bash
litellm --config /path/to/config.yaml
```

3. 测试一下！

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": [    
          {
            "type": "text",
            "text": "what color is red"
          }
        ]
      }
    ],
    "max_tokens": 300,
    "mock_testing_fallbacks": true
}'
```

### 关闭回退（按请求 / 按 Key）


<Tabs>

<TabItem value="request" label="按请求关闭">

在请求体中设置 `disable_fallbacks: true` 即可按单次请求关闭回退。

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "messages": [
        {
            "role": "user",
            "content": "List 5 important events in the XIX century"
        }
    ],
    "model": "gpt-3.5-turbo",
    "disable_fallbacks": true # 👈 关闭回退
}'
```

</TabItem>

<TabItem value="key" label="按 Key 关闭">

在 key 的 metadata 中设置 `disable_fallbacks: true`，即可按 Virtual Key 关闭回退。

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "metadata": {
        "disable_fallbacks": true
    }
}'
```

</TabItem>
</Tabs>

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Provider-specific Params

Providers might offer params not supported by OpenAI (e.g. top_k). LiteLLM treats any non-openai param, as a provider-specific param, and passes it to the provider in the request body, as a kwarg. [**See Reserved Params**](https://github.com/BerriAI/litellm/blob/aa2fd29e48245f360e771a8810a69376464b195e/litellm/main.py#L700)

You can pass those in 2 ways: 
- via completion(): We'll pass the non-openai param, straight to the provider as part of the request body.
    - e.g. `completion(model="claude-instant-1", top_k=3)`
- via provider-specific config variable (e.g. `litellm.OpenAIConfig()`). 

## SDK Usage
<Tabs>
<TabItem value="openai" label="OpenAI">

```python
import litellm, os

# set env variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"

## SET MAX TOKENS - via completion() 
response_1 = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.OpenAIConfig(max_tokens=10)

response_2 = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>
<TabItem value="openai-text" label="OpenAI Text Completion">

```python
import litellm, os

# set env variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"


## SET MAX TOKENS - via completion() 
response_1 = litellm.completion(
            model="text-davinci-003",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.OpenAITextCompletionConfig(max_tokens=10)
response_2 = litellm.completion(
            model="text-davinci-003",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>
<TabItem value="azure-openai" label="Azure OpenAI">

```python
import litellm, os

# set env variables
os.environ["AZURE_API_BASE"] = "your-azure-api-base"
os.environ["AZURE_API_TYPE"] = "azure" # [OPTIONAL] 
os.environ["AZURE_API_VERSION"] = "2023-07-01-preview" # [OPTIONAL]

## SET MAX TOKENS - via completion() 
response_1 = litellm.completion(
            model="azure/chatgpt-v-2",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.AzureOpenAIConfig(max_tokens=10)
response_2 = litellm.completion(
            model="azure/chatgpt-v-2",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```python
import litellm, os 

# set env variables
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

## SET MAX TOKENS - via completion()
response_1 = litellm.completion(
            model="claude-instant-1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.AnthropicConfig(max_tokens_to_sample=200)
response_2 = litellm.completion(
            model="claude-instant-1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>

<TabItem value="huggingface" label="Huggingface">

```python
import litellm, os 

# set env variables
os.environ["HUGGINGFACE_API_KEY"] = "your-huggingface-key" #[OPTIONAL]

## SET MAX TOKENS - via completion()
response_1 = litellm.completion(
            model="huggingface/mistralai/Mistral-7B-Instruct-v0.1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            api_base="https://your-huggingface-api-endpoint",
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.HuggingfaceConfig(max_new_tokens=200)
response_2 = litellm.completion(
            model="huggingface/mistralai/Mistral-7B-Instruct-v0.1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            api_base="https://your-huggingface-api-endpoint"
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>

<TabItem value="together_ai" label="TogetherAI">


```python
import litellm, os 

# set env variables
os.environ["TOGETHERAI_API_KEY"] = "your-togetherai-key" 

## SET MAX TOKENS - via completion()
response_1 = litellm.completion(
            model="together_ai/togethercomputer/llama-2-70b-chat",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.TogetherAIConfig(max_tokens_to_sample=200)
response_2 = litellm.completion(
            model="together_ai/togethercomputer/llama-2-70b-chat",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>

<TabItem value="ollama" label="Ollama">

```python
import litellm, os 

## SET MAX TOKENS - via completion()
response_1 = litellm.completion(
            model="ollama/llama2",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.OllamConfig(num_predict=200)
response_2 = litellm.completion(
            model="ollama/llama2",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>

<TabItem value="replicate" label="Replicate">

```python
import litellm, os 

# set env variables
os.environ["REPLICATE_API_KEY"] = "your-replicate-key" 

## SET MAX TOKENS - via completion()
response_1 = litellm.completion(
            model="replicate/meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.ReplicateConfig(max_new_tokens=200)
response_2 = litellm.completion(
            model="replicate/meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>

<TabItem value="petals" label="Petals">


```python
import litellm

## SET MAX TOKENS - via completion()
response_1 = litellm.completion(
            model="petals/petals-team/StableBeluga2",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            api_base="https://chat.petals.dev/api/v1/generate",
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.PetalsConfig(max_new_tokens=10)
response_2 = litellm.completion(
            model="petals/petals-team/StableBeluga2",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            api_base="https://chat.petals.dev/api/v1/generate",
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>

<TabItem value="palm" label="Palm">

```python
import litellm, os 

# set env variables
os.environ["PALM_API_KEY"] = "your-palm-key"  

## SET MAX TOKENS - via completion()
response_1 = litellm.completion(
            model="palm/chat-bison",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.PalmConfig(maxOutputTokens=10)
response_2 = litellm.completion(
            model="palm/chat-bison",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```
</TabItem>

<TabItem value="ai21" label="AI21">

```python
import litellm, os 

# set env variables
os.environ["AI21_API_KEY"] = "your-ai21-key"  

## SET MAX TOKENS - via completion()
response_1 = litellm.completion(
            model="j2-mid",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.AI21Config(maxOutputTokens=10)
response_2 = litellm.completion(
            model="j2-mid",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>

<TabItem value="cohere" label="Cohere">

```python
import litellm, os 

# set env variables
os.environ["COHERE_API_KEY"] = "your-cohere-key"   

## SET MAX TOKENS - via completion()
response_1 = litellm.completion(
            model="command-nightly",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

response_1_text = response_1.choices[0].message.content

## SET MAX TOKENS - via config
litellm.CohereConfig(max_tokens=200)
response_2 = litellm.completion(
            model="command-nightly",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
        )

response_2_text = response_2.choices[0].message.content

## TEST OUTPUT
assert len(response_2_text) > len(response_1_text)
```

</TabItem>

</Tabs>


[**Check out the tutorial!**](../tutorials/provider_specific_params.md)


## Proxy Usage 

**via Config**

```yaml
model_list:
    - model_name: llama-3-8b-instruct
      litellm_params:
        model: predibase/llama-3-8b-instruct
        api_key: os.environ/PREDIBASE_API_KEY
        tenant_id: os.environ/PREDIBASE_TENANT_ID
        max_tokens: 256
        adapter_base: <my-special_base> # 👈 PROVIDER-SPECIFIC PARAM
```

**via Request**

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "llama-3-8b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"
    }
  ],
  "adapater_id": "my-special-adapter-id"
}'
```

## Provider-Specific Metadata Parameters

| Provider | Parameter | Use Case |
|----------|-----------|----------|
| **AWS Bedrock** | `requestMetadata` | Cost attribution, logging |
| **Gemini/Vertex AI** | `labels` | Resource labeling |
| **Anthropic** | `metadata` | User identification |

<Tabs>
<TabItem value="bedrock" label="AWS Bedrock">

```python
import litellm

response = litellm.completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[{"role": "user", "content": "Hello!"}],
    requestMetadata={"cost_center": "engineering"}
)
```

</TabItem>
<TabItem value="gemini" label="Gemini/Vertex AI">

```python
import litellm

response = litellm.completion(
    model="vertex_ai/gemini-pro",
    messages=[{"role": "user", "content": "Hello!"}],
    labels={"environment": "production"}
)
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Hello!"}],
    metadata={"user_id": "user123"}
)
```

</TabItem>
</Tabs>
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [Beta] Fine-tuning API


:::info

This is an Enterprise only endpoint [Get Started with Enterprise here](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

## Supported Providers
- Azure OpenAI
- OpenAI

Add `finetune_settings` and `files_settings` to your litellm config.yaml to use the fine-tuning endpoints.
## Usage
```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

# For /fine_tuning/jobs endpoints
finetune_settings:
  - custom_llm_provider: azure
    api_base: https://exampleopenaiendpoint-production.up.railway.app
    api_key: os.environ/AZURE_API_KEY
    api_version: "2023-03-15-preview"
  - custom_llm_provider: openai
    api_key: os.environ/OPENAI_API_KEY

# for /files endpoints
files_settings:
  - custom_llm_provider: azure
    api_base: https://exampleopenaiendpoint-production.up.railway.app
    api_key: fake-key
    api_version: "2023-03-15-preview"
  - custom_llm_provider: openai
    api_key: os.environ/OPENAI_API_KEY
```

## Create File for Fine Tuning

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000") # base_url is your litellm proxy url

file_name = "openai_batch_completions.jsonl"
response = await client.files.create(
    extra_body={"custom_llm_provider": "azure"}, # tell litellm proxy which provider to use
    file=open(file_name, "rb"),
    purpose="fine-tune",
)
```
</TabItem>
</Tabs>

## Create fine-tuning job

<Tabs>
</Tabs>

## Cancel fine-tuning job

<Tabs>
</Tabs>

## List fine-tuning jobs

<Tabs>
</Tabs>

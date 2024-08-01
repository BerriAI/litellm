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
## Example config.yaml for `finetune_settings` and `files_settings`
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

## Create File for fine-tuning

<Tabs>
<TabItem value="openai" label="OpenAI Python SDK">

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
<TabItem value="curl" label="curl">

```shell
curl http://localhost:4000/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F custom_llm_provider="azure"\
    -F file="@mydata.jsonl"
```
</TabItem>
</Tabs>

## Create fine-tuning job

<Tabs>
<TabItem value="openai" label="OpenAI Python SDK">

```python
ft_job = await client.fine_tuning.jobs.create(
    model="gpt-35-turbo-1106",                   # Azure OpenAI model you want to fine-tune
    training_file="file-abc123",                 # file_id from create file response
    extra_body={"custom_llm_provider": "azure"}, # tell litellm proxy which provider to use
)
```
</TabItem>

<TabItem value="curl" label="curl">

```shell
curl http://localhost:4000/v1/fine_tuning/jobs \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
    "custom_llm_provider": "azure",
    "model": "gpt-35-turbo-1106",
    "training_file": "file-abc123"
    }'
```
</TabItem>
</Tabs>

## Cancel fine-tuning job

<Tabs>
<TabItem value="openai" label="OpenAI Python SDK">

```python
# cancel specific fine tuning job
cancel_ft_job = await client.fine_tuning.jobs.cancel(
    fine_tuning_job_id="123",                          # fine tuning job id
    extra_body={"custom_llm_provider": "azure"},       # tell litellm proxy which provider to use
)

print("response from cancel ft job={}".format(cancel_ft_job))
```
</TabItem>

<TabItem value="curl" label="curl">

```shell
curl -X POST http://localhost:4000/v1/fine_tuning/jobs/ftjob-abc123/cancel \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{"custom_llm_provider": "azure"}'
```
</TabItem>

</Tabs>

## List fine-tuning jobs

<Tabs>

<TabItem value="openai" label="OpenAI Python SDK">

```python
list_ft_jobs = await client.fine_tuning.jobs.list(
    extra_query={"custom_llm_provider": "azure"}   # tell litellm proxy which provider to use
)

print("list of ft jobs={}".format(list_ft_jobs))
```
</TabItem>

<TabItem value="curl" label="curl">

```shell
curl -X GET 'http://localhost:4000/v1/fine_tuning/jobs?custom_llm_provider=azure' \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-1234"
```
</TabItem>

</Tabs>



## [👉 Proxy API Reference](https://litellm-api.up.railway.app/#/fine-tuning)
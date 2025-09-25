import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /fine_tuning


:::info

This is an Enterprise only endpoint [Get Started with Enterprise here](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

| Feature | Supported | Notes | 
|-------|-------|-------|
| Supported Providers | OpenAI, Azure OpenAI, Vertex AI | - |

#### ⚡️See an exhaustive list of supported models and providers at [models.litellm.ai](https://models.litellm.ai/)
| Cost Tracking | 🟡 | [Let us know if you need this](https://github.com/BerriAI/litellm/issues) |
| Logging | ✅ | Works across all logging integrations |


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
  - custom_llm_provider: "vertex_ai"
    vertex_project: "adroit-crow-413218"
    vertex_location: "us-central1"
    vertex_credentials: "/Users/ishaanjaffer/Downloads/adroit-crow-413218-a956eef1a2a8.json"

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
<TabItem value="azure" label="Azure OpenAI">

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

</TabItem>

</Tabs>

### Request Body

<Tabs>
<TabItem value="params" label="Supported Params">

* `model`

    **Type:** string  
    **Required:** Yes  
    The name of the model to fine-tune

* `custom_llm_provider`

    **Type:** `Literal["azure", "openai", "vertex_ai"]`

    **Required:** Yes
    The name of the model to fine-tune. You can select one of the [**supported providers**](#supported-providers)

* `training_file`

    **Type:** string  
    **Required:** Yes  
    The ID of an uploaded file that contains training data.
    - See **upload file** for how to upload a file.
    - Your dataset must be formatted as a JSONL file.

* `hyperparameters`

    **Type:** object  
    **Required:** No  
    The hyperparameters used for the fine-tuning job.
    > #### Supported `hyperparameters`
    > #### batch_size
    **Type:** string or integer  
    **Required:** No  
    Number of examples in each batch. A larger batch size means that model parameters are updated less frequently, but with lower variance.
    > #### learning_rate_multiplier
    **Type:** string or number  
    **Required:** No  
    Scaling factor for the learning rate. A smaller learning rate may be useful to avoid overfitting.

    > #### n_epochs
    **Type:** string or integer  
    **Required:** No  
    The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset.

* `suffix`
    **Type:** string or null  
    **Required:** No  
    **Default:** null  
    A string of up to 18 characters that will be added to your fine-tuned model name.
    Example: A `suffix` of "custom-model-name" would produce a model name like `ft:gpt-4o-mini:openai:custom-model-name:7p4lURel`.

* `validation_file`
    **Type:** string or null  
    **Required:** No  
    The ID of an uploaded file that contains validation data.
    - If provided, this data is used to generate validation metrics periodically during fine-tuning.


* `integrations`
    **Type:** array or null  
    **Required:** No  
    A list of integrations to enable for your fine-tuning job.

* `seed`
    **Type:** integer or null  
    **Required:** No  
    The seed controls the reproducibility of the job. Passing in the same seed and job parameters should produce the same results, but may differ in rare cases. If a seed is not specified, one will be generated for you.

</TabItem>
<TabItem value="example" label="Example Request Body">

```json
{
  "model": "gpt-4o-mini",
  "training_file": "file-abcde12345",
  "hyperparameters": {
    "batch_size": 4,
    "learning_rate_multiplier": 0.1,
    "n_epochs": 3
  },
  "suffix": "custom-model-v1",
  "validation_file": "file-fghij67890",
  "seed": 42
}
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
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Batches API

Covers Batches, Files


## Quick Start 

Call an existing Assistant. 

- Create File for Batch Completion

- Create Batch Request

- Retrieve the Specific Batch and File Content


<Tabs>
<TabItem value="sdk" label="SDK">

**Create File for Batch Completion**

```python
from litellm
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

file_name = "openai_batch_completions.jsonl"
_current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_current_dir, file_name)
file_obj = await litellm.acreate_file(
    file=open(file_path, "rb"),
    purpose="batch",
    custom_llm_provider="openai",
)
print("Response from creating file=", file_obj)
```

**Create Batch Request**

```python
from litellm
import os 

create_batch_response = await litellm.acreate_batch(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=batch_input_file_id,
    custom_llm_provider="openai",
    metadata={"key1": "value1", "key2": "value2"},
)

print("response from litellm.create_batch=", create_batch_response)
```

**Retrieve the Specific Batch and File Content**

```python

retrieved_batch = await litellm.aretrieve_batch(
    batch_id=create_batch_response.id, custom_llm_provider="openai"
)
print("retrieved batch=", retrieved_batch)
# just assert that we retrieved a non None batch

assert retrieved_batch.id == create_batch_response.id

# try to get file content for our original file

file_content = await litellm.afile_content(
    file_id=batch_input_file_id, custom_llm_provider="openai"
)

print("file content = ", file_content)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
$ export OPENAI_API_KEY="sk-..."

$ litellm

# RUNNING on http://0.0.0.0:4000
```

**Create File for Batch Completion**

```shell
curl https://api.openai.com/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@mydata.jsonl"
```

**Create Batch Request**

```bash
curl http://localhost:4000/v1/batches \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "input_file_id": "file-abc123",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
    }'
```

**Retrieve the Specific Batch**

```bash
curl http://localhost:4000/v1/batches/batch_abc123 \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
```

</TabItem>
</Tabs>

## [👉 Proxy API Reference](https://litellm-api.up.railway.app/#/batch)

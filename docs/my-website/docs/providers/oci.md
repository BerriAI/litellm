import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Oracle Cloud Infrastructure (OCI)
LiteLLM supports the following models for OCI on-demand GenAI API.

Check the [OCI Models List](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm) to see if the model is available for your region.

- `cohere.command-a-03-2025`
- `cohere.command-r-08-2024`
- `cohere.command-plus-latest` (alias `cohere.command-r-plus-08-2024`)
- `cohere.command-r-16k` (deprecated)
- `cohere.command-r-plus` (deprecated)

- `meta.llama-4-maverick-17b-128e-instruct-fp8`
- `meta.llama-4-scout-17b-16e-instruct`
- `meta.llama-3.3-70b-instruct`
- `meta.llama-3.2-90b-vision-instruct`
- `meta.llama-3.2-11b-vision-instruct`
- `meta.llama-3.1-405b-instruct`
- `meta.llama-3.1-70b-instruct`
- `meta.llama-3-70b-instruct`

- `xai.grok-4`
- `xai.grok-3`
- `xai.grok-3-fast`
- `xai.grok-3-mini`
- `xai.grok-3-mini-fast`

## Authentication

LiteLLM uses OCI signing key authentication. Follow the [official Oracle tutorial](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm) to create a signing key and obtain the following parameters:

- `user`
- `fingerprint`
- `tenancy`
- `region`
- `key_file`

## Usage

Input the parameters obtained from the OCI signing key creation process into the `completion` function.

```python
import os
from litellm import completion

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",
    messages=messages,
    oci_region=<your_oci_region>,
    oci_user=<your_oci_user>,
    oci_fingerprint=<your_oci_fingerprint>,
    oci_tenancy=<your_oci_tenancy>,
    oci_key=<string_with_content_of_oci_key>,
    oci_compartment_id=<oci_compartment_id>,
)
print(response)
```


## Usage - Streaming
Just set `stream=True` when calling completion.

```python
import os
from litellm import completion

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",
    messages=messages,
    stream=True,
    oci_region=<your_oci_region>,
    oci_user=<your_oci_user>,
    oci_fingerprint=<your_oci_fingerprint>,
    oci_tenancy=<your_oci_tenancy>,
    oci_key=<string_with_content_of_oci_key>,
    oci_compartment_id=<oci_compartment_id>,
)
for chunk in response:
    print(chunk["choices"][0]["delta"]["content"])  # same as openai format
```

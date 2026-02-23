import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Oracle Cloud Infrastructure (OCI)
LiteLLM supports the following models for OCI on-demand GenAI API.

Check the [OCI Models List](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm) to see if the model is available for your region.

## Supported Models

### Meta Llama Models
- `meta.llama-4-maverick-17b-128e-instruct-fp8`
- `meta.llama-4-scout-17b-16e-instruct`
- `meta.llama-3.3-70b-instruct`
- `meta.llama-3.2-90b-vision-instruct`
- `meta.llama-3.1-405b-instruct`

### xAI Grok Models
- `xai.grok-4`
- `xai.grok-3`
- `xai.grok-3-fast`
- `xai.grok-3-mini`
- `xai.grok-3-mini-fast`

### Cohere Models
- `cohere.command-latest`
- `cohere.command-a-03-2025`
- `cohere.command-plus-latest`

## Authentication

LiteLLM supports two authentication methods for OCI:

### Method 1: Manual Credentials
Provide individual OCI credentials directly to LiteLLM. Follow the [official Oracle tutorial](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm) to create a signing key and obtain the following parameters:

- `user`
- `fingerprint`
- `tenancy`
- `region`
- `key_file` or `key`
- `compartment_id`

This is the default method for LiteLLM AI Gateway (LLM Proxy) access to OCI GenAI models.

### Method 2: OCI SDK Signer
Use an OCI SDK `Signer` object for authentication. This method:
- Leverages the official [OCI SDK for signing](https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html)
- Supports additional authentication methods (instance principals, workload identity, etc.)

To use this method, install the OCI SDK:
```bash
pip install oci
```

This method is an alternative when using the LiteLLM SDK on Oracle Cloud Infrastructure (instances or Oracle Kubernetes Engine).

## Usage

<Tabs>
<TabItem value="manual" label="Manual Credentials" default>

Input the parameters obtained from the OCI signing key creation process into the `completion` function:

```python
from litellm import completion

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",
    messages=messages,
    oci_region=<your_oci_region>,
    oci_user=<your_oci_user>,
    oci_fingerprint=<your_oci_fingerprint>,
    oci_tenancy=<your_oci_tenancy>,
    oci_serving_mode="ON_DEMAND",  # Optional, default is "ON_DEMAND". Other option is "DEDICATED"
    # Provide either the private key string OR the path to the key file:
    # Option 1: pass the private key as a string
    oci_key=<string_with_content_of_oci_key>,
    # Option 2: pass the private key file path
    # oci_key_file="<path/to/oci_key.pem>",
    oci_compartment_id=<oci_compartment_id>,
)
print(response)
```

</TabItem>
<TabItem value="oci-sdk" label="OCI SDK Signer">

Use the OCI SDK `Signer` for authentication:

```python
from litellm import completion
from oci.signer import Signer

# Create an OCI Signer
signer = Signer(
    tenancy="ocid1.tenancy.oc1..",
    user="ocid1.user.oc1..",
    fingerprint="xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx",
    private_key_file_location="~/.oci/key.pem",
    # Or use private_key_content="<your_private_key_content>"
)

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",
    messages=messages,
    oci_signer=signer,
    oci_region="us-chicago-1",  # Optional, defaults to us-ashburn-1
    oci_serving_mode="ON_DEMAND",  # Optional, default is "ON_DEMAND". Other option is "DEDICATED"
    oci_compartment_id="<oci_compartment_id>",
)
print(response)
```

**Alternative: Use OCI Config File**

The OCI SDK can automatically load credentials from `~/.oci/config`:

```python
from litellm import completion
from oci.config import from_file
from oci.signer import Signer

# Load config from file
config = from_file("~/.oci/config", "DEFAULT")  # "DEFAULT" is the profile name
signer = Signer(
    tenancy=config["tenancy"],
    user=config["user"],
    fingerprint=config["fingerprint"],
    private_key_file_location=config["key_file"],
    pass_phrase=config.get("pass_phrase")  # Optional if key is encrypted
)

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",
    messages=messages,
    oci_signer=signer,
    oci_region=config["region"],
    oci_compartment_id="<oci_compartment_id>",
)
print(response)
```

**Instance Principal Authentication**

For applications running on OCI compute instances:

```python
from litellm import completion
from oci.auth.signers import InstancePrincipalsSecurityTokenSigner

# Use instance principal authentication
signer = InstancePrincipalsSecurityTokenSigner()

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",
    messages=messages,
    oci_signer=signer,
    oci_region="us-chicago-1",
    oci_compartment_id="<oci_compartment_id>",
)
print(response)
```

**Workload Identity Authentication**

For applications running in Oracle Kubernetes Engine (OKE):

```python
from litellm import completion
from oci.auth.signers import get_oke_workload_identity_resource_principal_signer

# Use workload identity authentication
signer = get_oke_workload_identity_resource_principal_signer()

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",
    messages=messages,
    oci_signer=signer,
    oci_region="us-chicago-1",
    oci_compartment_id="<oci_compartment_id>",
)
print(response)
```
</TabItem>
</Tabs>

## Usage - Streaming
Just set `stream=True` when calling completion.

<Tabs>
<TabItem value="manual-stream" label="Manual Credentials" default>

```python
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
    oci_serving_mode="ON_DEMAND",  # Optional, default is "ON_DEMAND". Other option is "DEDICATED"
    # Provide either the private key string OR the path to the key file:
    # Option 1: pass the private key as a string
    oci_key=<string_with_content_of_oci_key>,
    # Option 2: pass the private key file path
    # oci_key_file="<path/to/oci_key.pem>",
    oci_compartment_id=<oci_compartment_id>,
)
for chunk in response:
    print(chunk["choices"][0]["delta"]["content"])  # same as openai format
```

</TabItem>
<TabItem value="oci-sdk-stream" label="OCI SDK Signer">

```python
from litellm import completion
from oci.signer import Signer

signer = Signer(
    tenancy="ocid1.tenancy.oc1..",
    user="ocid1.user.oc1..",
    fingerprint="xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx",
    private_key_file_location="~/.oci/key.pem",
)

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",
    messages=messages,
    stream=True,
    oci_signer=signer,
    oci_region="us-chicago-1",
    oci_compartment_id="<oci_compartment_id>",
)
for chunk in response:
    print(chunk["choices"][0]["delta"]["content"])  # same as openai format
```

</TabItem>
</Tabs>

## Usage Examples by Model Type

### Using Cohere Models

<Tabs>
<TabItem value="cohere-manual" label="Manual Credentials" default>

```python
from litellm import completion

messages = [{"role": "user", "content": "Explain quantum computing"}]
response = completion(
    model="oci/cohere.command-latest",
    messages=messages,
    oci_region="us-chicago-1",
    oci_user=<your_oci_user>,
    oci_fingerprint=<your_oci_fingerprint>,
    oci_tenancy=<your_oci_tenancy>,
    oci_key=<string_with_content_of_oci_key>,
    oci_compartment_id=<oci_compartment_id>,
)
print(response)
```

</TabItem>
<TabItem value="cohere-sdk" label="OCI SDK Signer">

```python
from litellm import completion
from oci.signer import Signer

signer = Signer(
    tenancy="ocid1.tenancy.oc1..",
    user="ocid1.user.oc1..",
    fingerprint="xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx",
    private_key_file_location="~/.oci/key.pem",
)

messages = [{"role": "user", "content": "Explain quantum computing"}]
response = completion(
    model="oci/cohere.command-latest",
    messages=messages,
    oci_signer=signer,
    oci_region="us-chicago-1",
    oci_compartment_id="<oci_compartment_id>",
)
print(response)
```

</TabItem>
</Tabs>

## Using Dedicated Endpoints

OCI supports dedicated endpoints for hosting models. Use the `oci_serving_mode="DEDICATED"` parameter along with `oci_endpoint_id` to specify the endpoint ID.

<Tabs>
<TabItem value="dedicated-manual" label="Manual Credentials" default>

```python
from litellm import completion

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",  # Must match the model type hosted on the endpoint
    messages=messages,
    oci_region=<your_oci_region>,
    oci_user=<your_oci_user>,
    oci_fingerprint=<your_oci_fingerprint>,
    oci_tenancy=<your_oci_tenancy>,
    oci_serving_mode="DEDICATED",
    oci_endpoint_id="ocid1.generativeaiendpoint.oc1...",  # Your dedicated endpoint OCID
    oci_key=<string_with_content_of_oci_key>,
    oci_compartment_id=<oci_compartment_id>,
)
print(response)
```

</TabItem>
<TabItem value="dedicated-sdk" label="OCI SDK Signer">

```python
from litellm import completion
from oci.signer import Signer

signer = Signer(
    tenancy="ocid1.tenancy.oc1..",
    user="ocid1.user.oc1..",
    fingerprint="xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx",
    private_key_file_location="~/.oci/key.pem",
)

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(
    model="oci/xai.grok-4",  # Must match the model type hosted on the endpoint
    messages=messages,
    oci_signer=signer,
    oci_region="us-chicago-1",
    oci_serving_mode="DEDICATED",
    oci_endpoint_id="ocid1.generativeaiendpoint.oc1...",  # Your dedicated endpoint OCID
    oci_compartment_id="<oci_compartment_id>",
)
print(response)
```

</TabItem>
</Tabs>

**Important:** When using `oci_serving_mode="DEDICATED"`:
- The `model` parameter **must match the type of model hosted on your dedicated endpoint** (e.g., use `"oci/cohere.command-latest"` for Cohere models, `"oci/xai.grok-4"` for Grok models)
- The model name determines the API format and vendor-specific handling (Cohere vs Generic)
- The `oci_endpoint_id` parameter specifies your dedicated endpoint's OCID
- If `oci_endpoint_id` is not provided, the `model` parameter will be used as the endpoint ID (for backward compatibility)

**Example with Cohere Dedicated Endpoint:**
```python
# For a dedicated endpoint hosting a Cohere model
response = completion(
    model="oci/cohere.command-latest",  # Use Cohere model name to get Cohere API format
    messages=messages,
    oci_region="us-chicago-1",
    oci_user=<your_oci_user>,
    oci_fingerprint=<your_oci_fingerprint>,
    oci_tenancy=<your_oci_tenancy>,
    oci_serving_mode="DEDICATED",
    oci_endpoint_id="ocid1.generativeaiendpoint.oc1...",  # Your Cohere endpoint OCID
    oci_key=<string_with_content_of_oci_key>,
    oci_compartment_id=<oci_compartment_id>,
)
```

## Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `oci_region` | string | `us-ashburn-1` | OCI region where the GenAI service is deployed |
| `oci_serving_mode` | string | `ON_DEMAND` | Service mode: `ON_DEMAND` for managed models or `DEDICATED` for dedicated endpoints |
| `oci_endpoint_id` | string | Same as `model` | (For DEDICATED mode) The OCID of your dedicated endpoint |
| `oci_compartment_id` | string | **Required** | The OCID of the OCI compartment containing your resources |
| `oci_user` | string | - | (Manual auth) The OCID of the OCI user |
| `oci_fingerprint` | string | - | (Manual auth) The fingerprint of the API signing key |
| `oci_tenancy` | string | - | (Manual auth) The OCID of your OCI tenancy |
| `oci_key` | string | - | (Manual auth) The private key content as a string |
| `oci_key_file` | string | - | (Manual auth) Path to the private key file |
| `oci_signer` | object | - | (SDK auth) OCI SDK Signer object for authentication |
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Oracle Cloud Infrastructure (OCI)
LiteLLM supports the following models for OCI on-demand GenAI API.

Check the [OCI Models List](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm) to see if the model is available for your region.

## Supported Models

For model lifecycle, retirement dates, and recommended replacements, see [OCI's on-demand model retirement page](https://docs.oracle.com/en-us/iaas/Content/generative-ai/deprecating-on-demand.htm) — Oracle is the authoritative source.

### Chat / Text Generation

#### Meta Llama Models
- `meta.llama-4-maverick-17b-128e-instruct-fp8` (multimodal)
- `meta.llama-4-scout-17b-16e-instruct` (multimodal)
- `meta.llama-3.3-70b-instruct`
- `meta.llama-3.3-70b-instruct-fp8-dynamic`
- `meta.llama-3.2-90b-vision-instruct` (multimodal)
- `meta.llama-3.2-11b-vision-instruct` (multimodal)

#### xAI Grok Models
- `xai.grok-4.3`
- `xai.grok-4.20`
- `xai.grok-4.20-multi-agent`
- `xai.grok-4`
- `xai.grok-4-fast`
- `xai.grok-4.1-fast`
- `xai.grok-3`
- `xai.grok-3-fast`
- `xai.grok-3-mini`
- `xai.grok-3-mini-fast`
- `xai.grok-code-fast-1`

#### Cohere Models
- `cohere.command-latest`
- `cohere.command-a-03-2025`
- `cohere.command-a-reasoning-08-2025`
- `cohere.command-a-vision-07-2025` (multimodal)
- `cohere.command-a-translate-08-2025`
- `cohere.command-plus-latest`
- `cohere.command-r-plus-08-2024`
- `cohere.command-r-08-2024`

#### Google Gemini Models (via OCI)
- `google.gemini-2.5-pro` (multimodal)
- `google.gemini-2.5-flash` (multimodal)
- `google.gemini-2.5-flash-lite` (multimodal)

#### OpenAI Open-Source Models (via OCI)
- `openai.gpt-oss-120b`
- `openai.gpt-oss-20b`

### Embedding Models
- `cohere.embed-v4.0` (1536 dimensions, multimodal)
- `cohere.embed-english-v3.0` (1024 dimensions)
- `cohere.embed-english-light-v3.0` (384 dimensions)
- `cohere.embed-multilingual-v3.0` (1024 dimensions)
- `cohere.embed-multilingual-light-v3.0` (384 dimensions)
- `cohere.embed-english-image-v3.0` (1024 dimensions, multimodal)
- `cohere.embed-english-light-image-v3.0` (384 dimensions, multimodal)
- `cohere.embed-multilingual-image-v3.0` (1024 dimensions, multimodal)
- `cohere.embed-multilingual-light-image-v3.0` (384 dimensions, multimodal)

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

**Environment Variables**

Instead of passing credentials in code, you can set the following environment variables — LiteLLM will read them automatically:

```bash
export OCI_REGION="us-chicago-1"
export OCI_USER="ocid1.user.oc1.."
export OCI_FINGERPRINT="xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx"
export OCI_TENANCY="ocid1.tenancy.oc1.."
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1.."
# Provide either the private key content OR the path to the key file:
export OCI_KEY_FILE="/path/to/oci_api_key.pem"
# export OCI_KEY="-----BEGIN PRIVATE KEY-----\n..."
```

### Method 2: OCI SDK Signer
Use an OCI SDK `Signer` object for authentication. This method:
- Leverages the official [OCI SDK for signing](https://docs.oracle.com/en-us/iaas/tools/python/latest/api/signing.html)
- Supports additional authentication methods (instance principals, workload identity, etc.)

To use this method, install the OCI SDK:
```bash
uv add oci
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

## LiteLLM Proxy Usage

Here's how to call OCI GenAI through the LiteLLM Proxy Server.

### 1. Setup config.yaml

```yaml
model_list:
  - model_name: oci-grok-4
    litellm_params:
      model: oci/xai.grok-4
      oci_region: os.environ/OCI_REGION
      oci_user: os.environ/OCI_USER
      oci_fingerprint: os.environ/OCI_FINGERPRINT
      oci_tenancy: os.environ/OCI_TENANCY
      oci_key_file: os.environ/OCI_KEY_FILE
      oci_compartment_id: os.environ/OCI_COMPARTMENT_ID

  - model_name: oci-cohere-command
    litellm_params:
      model: oci/cohere.command-latest
      oci_region: os.environ/OCI_REGION
      oci_user: os.environ/OCI_USER
      oci_fingerprint: os.environ/OCI_FINGERPRINT
      oci_tenancy: os.environ/OCI_TENANCY
      oci_key_file: os.environ/OCI_KEY_FILE
      oci_compartment_id: os.environ/OCI_COMPARTMENT_ID
```

All possible auth params:

```
oci_region: Optional[str],
oci_user: Optional[str],
oci_fingerprint: Optional[str],
oci_tenancy: Optional[str],
oci_key: Optional[str],          # private key content as string
oci_key_file: Optional[str],     # path to .pem file
oci_compartment_id: Optional[str],
oci_serving_mode: Optional[str], # "ON_DEMAND" (default) or "DEDICATED"
oci_endpoint_id: Optional[str],  # only used with DEDICATED
```

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml
```

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "model": "oci-grok-4",
  "messages": [
    {"role": "user", "content": "what llm are you"}
  ]
}'
```

</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai

client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="oci-grok-4",
    messages=[{"role": "user", "content": "write a short poem"}],
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

## Usage - Function Calling / Tool Calling

OCI GenAI supports OpenAI-compatible function calling. LiteLLM normalizes the request and response shape so the same code that targets OpenAI works with OCI Cohere and Generic (xAI Grok, Meta Llama, Google Gemini) models.

<Tabs>
<TabItem value="tool-sdk" label="SDK">

```python
from litellm import completion

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = completion(
    model="oci/xai.grok-4",
    messages=[{"role": "user", "content": "What's the weather in Boston today?"}],
    tools=tools,
    tool_choice="auto",
    oci_region="us-chicago-1",
    oci_user="<your_oci_user>",
    oci_fingerprint="<your_oci_fingerprint>",
    oci_tenancy="<your_oci_tenancy>",
    oci_key_file="<path/to/oci_key.pem>",
    oci_compartment_id="<oci_compartment_id>",
)

# Inspect the tool call
print(response.choices[0].message.tool_calls)
```

</TabItem>
<TabItem value="tool-proxy" label="PROXY">

```python
import openai

client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")

response = client.chat.completions.create(
    model="oci-grok-4",
    messages=[{"role": "user", "content": "What's the weather in Boston today?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ],
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls)
```

</TabItem>
</Tabs>

Tool calling works with both Cohere (`cohere.command-*`) and Generic (`xai.grok-*`, `meta.llama-*`, `google.gemini-*`) model families — LiteLLM adapts the OpenAI tool schema to each vendor's native format internally.

## Usage - Vision / Multimodal

OCI GenAI exposes vision-capable models that accept images alongside text. Pass images using the standard OpenAI `image_url` content block.

```python
from litellm import completion

response = completion(
    model="oci/meta.llama-4-maverick-17b-128e-instruct-fp8",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
        }
    ],
    oci_region="us-chicago-1",
    oci_user="<your_oci_user>",
    oci_fingerprint="<your_oci_fingerprint>",
    oci_tenancy="<your_oci_tenancy>",
    oci_key_file="<path/to/oci_key.pem>",
    oci_compartment_id="<oci_compartment_id>",
)
print(response.choices[0].message.content)
```

Vision-capable models on OCI include:

- `meta.llama-4-maverick-17b-128e-instruct-fp8`
- `meta.llama-4-scout-17b-16e-instruct`
- `meta.llama-3.2-11b-vision-instruct`
- `meta.llama-3.2-90b-vision-instruct`
- `cohere.command-a-vision-07-2025`
- `google.gemini-2.5-pro`, `google.gemini-2.5-flash`, `google.gemini-2.5-flash-lite`

Both URL and base64-encoded data URIs are supported.

## Usage - Reasoning / Thinking

OCI Generic-vendor models (xAI Grok reasoning variants, Google Gemini, etc.) support a reasoning step. LiteLLM exposes this via the OpenAI-compatible `reasoning_effort` parameter — accepted values are `"low"`, `"medium"`, `"high"`, and `"disable"` (mapped to OCI's `NONE`).

Returned reasoning tokens are surfaced on `usage.completion_tokens_details.reasoning_tokens`, matching the OpenAI shape.

<Tabs>
<TabItem value="reasoning-sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="oci/xai.grok-3-mini",
    messages=[{"role": "user", "content": "If 3x + 7 = 22, what is x? Show your reasoning."}],
    reasoning_effort="high",  # "low" | "medium" | "high" | "disable"
    oci_region="us-chicago-1",
    oci_user="<your_oci_user>",
    oci_fingerprint="<your_oci_fingerprint>",
    oci_tenancy="<your_oci_tenancy>",
    oci_key_file="<path/to/oci_key.pem>",
    oci_compartment_id="<oci_compartment_id>",
)

print(response.choices[0].message.content)
print("Reasoning tokens:", response.usage.completion_tokens_details.reasoning_tokens)
```

</TabItem>
<TabItem value="reasoning-proxy" label="PROXY">

```python
import openai

client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")

response = client.chat.completions.create(
    model="oci-grok-mini",
    messages=[{"role": "user", "content": "If 3x + 7 = 22, what is x?"}],
    reasoning_effort="high",
)
print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

:::note
`reasoning_effort` is only honored on Generic-vendor reasoning models (e.g., `xai.grok-3-mini`, `xai.grok-4`, `google.gemini-2.5-pro`). It is silently ignored for OCI Cohere models, which are not reasoning models.
:::

## Optional Parameters

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|----------------------|-------------|
| `oci_region` | string | `us-ashburn-1` | `OCI_REGION` | OCI region where the GenAI service is deployed |
| `oci_serving_mode` | string | `ON_DEMAND` | – | Service mode: `ON_DEMAND` for managed models or `DEDICATED` for dedicated endpoints |
| `oci_endpoint_id` | string | Same as `model` | – | (For DEDICATED mode) The OCID of your dedicated endpoint |
| `oci_compartment_id` | string | **Required** | `OCI_COMPARTMENT_ID` | The OCID of the OCI compartment containing your resources |
| `oci_user` | string | – | `OCI_USER` | (Manual auth) The OCID of the OCI user |
| `oci_fingerprint` | string | – | `OCI_FINGERPRINT` | (Manual auth) The fingerprint of the API signing key |
| `oci_tenancy` | string | – | `OCI_TENANCY` | (Manual auth) The OCID of your OCI tenancy |
| `oci_key` | string | – | `OCI_KEY` | (Manual auth) The private key content as a string |
| `oci_key_file` | string | – | `OCI_KEY_FILE` | (Manual auth) Path to the private key file |
| `oci_signer` | object | – | – | (SDK auth) OCI SDK Signer object for authentication |
| `reasoning_effort` | string | – | – | Reasoning level for Generic-vendor reasoning models: `low`, `medium`, `high`, `disable` |

## Embeddings

LiteLLM supports OCI Generative AI embedding models. These models use the same authentication methods described above.

<Tabs>
<TabItem value="embed-manual" label="Manual Credentials" default>

```python
from litellm import embedding

response = embedding(
    model="oci/cohere.embed-english-v3.0",
    input=["Hello world", "Goodbye world"],
    oci_region="us-ashburn-1",
    oci_user=<your_oci_user>,
    oci_fingerprint=<your_oci_fingerprint>,
    oci_tenancy=<your_oci_tenancy>,
    oci_key=<string_with_content_of_oci_key>,
    oci_compartment_id=<oci_compartment_id>,
)
print(response)
```

</TabItem>
<TabItem value="embed-sdk" label="OCI SDK Signer">

```python
from litellm import embedding
from oci.signer import Signer

signer = Signer(
    tenancy="ocid1.tenancy.oc1..",
    user="ocid1.user.oc1..",
    fingerprint="xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx",
    private_key_file_location="~/.oci/key.pem",
)

response = embedding(
    model="oci/cohere.embed-english-v3.0",
    input=["Hello world", "Goodbye world"],
    oci_signer=signer,
    oci_region="us-ashburn-1",
    oci_compartment_id="<oci_compartment_id>",
)
print(response)
```

</TabItem>
</Tabs>

### Embedding Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_type` | string | - | The type of input: `search_document`, `search_query`, `classification`, `clustering` |
| `truncate` | string | `END` | Truncation strategy when input exceeds max tokens: `END` or `START` |

### Using Dedicated Embedding Endpoints

```python
response = embedding(
    model="oci/cohere.embed-english-v3.0",
    input=["Hello world"],
    oci_serving_mode="DEDICATED",
    oci_endpoint_id="ocid1.generativeaiendpoint.oc1...",
    oci_region="us-ashburn-1",
    oci_compartment_id="<oci_compartment_id>",
    # ... auth params
)
```
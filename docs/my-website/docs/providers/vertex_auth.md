import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vertex AI Authentication

Set your vertex credentials via:
- dynamic params
OR
- env vars 


### **Dynamic Params**

You can set:
- `vertex_credentials` (str) - can be a json string or filepath to your vertex ai service account.json
- `vertex_location` (str) - place where vertex model is deployed (us-central1, asia-southeast1, etc.). Some models support the global location, please see [Vertex AI documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#supported_models)
- `vertex_project` Optional[str] - use if vertex project different from the one in vertex_credentials

as dynamic params for a `litellm.completion` call. 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import json 

## GET CREDENTIALS 
file_path = 'path/to/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)


response = completion(
  model="vertex_ai/gemini-2.5-pro",
  messages=[{"content": "You are a good bot.","role": "system"}, {"content": "Hello, how are you?","role": "user"}], 
  vertex_credentials=vertex_credentials_json,
  vertex_project="my-special-project", 
  vertex_location="my-special-location"
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
    - model_name: gemini-1.5-pro
      litellm_params:
        model: gemini-1.5-pro
        vertex_credentials: os.environ/VERTEX_FILE_PATH_ENV_VAR # os.environ["VERTEX_FILE_PATH_ENV_VAR"] = "/path/to/service_account.json" 
        vertex_project: "my-special-project"
        vertex_location: "my-special-location:
```

</TabItem>
</Tabs>




### **Environment Variables**

You can set:
- `GOOGLE_APPLICATION_CREDENTIALS` - store the filepath for your service_account.json in here (used by vertex sdk directly).
- VERTEXAI_LOCATION - place where vertex model is deployed (us-central1, asia-southeast1, etc.)
- VERTEXAI_PROJECT - Optional[str] - use if vertex project different from the one in vertex_credentials

1. GOOGLE_APPLICATION_CREDENTIALS

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
```

2. VERTEXAI_LOCATION

```bash
export VERTEXAI_LOCATION="us-central1" # can be any vertex location
```

3. VERTEXAI_PROJECT

```bash
export VERTEXAI_PROJECT="my-test-project" # ONLY use if model project is different from service account project
```

## AWS to GCP Federation (No Metadata Required)

Use AWS credentials to access Vertex AI without EC2 metadata endpoints. Ideal when `http://169.254.169.254` is blocked.

**Quick Setup:**

1. Create a credential file with your AWS auth params:

```json
{
  "type": "external_account",
  "audience": "//iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/providers/PROVIDER_ID",
  "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
  "token_url": "https://sts.googleapis.com/v1/token",
  "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/SA_EMAIL:generateAccessToken",
  "credential_source": {
    "environment_id": "aws1"
  },
  "aws_role_name": "arn:aws:iam::123456789012:role/MyRole",
  "aws_region_name": "us-east-1"
}
```

2. Use it in your code:

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
import json

with open('aws_gcp_credentials.json', 'r') as f:
    credentials = json.load(f)

response = litellm.completion(
    model="vertex_ai/gemini-pro",
    messages=[{"role": "user", "content": "Hello!"}],
    vertex_credentials=credentials,
    vertex_project="my-gcp-project",
    vertex_location="us-central1"
)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```yaml
model_list:
  - model_name: gemini-pro
    litellm_params:
      model: vertex_ai/gemini-pro
      vertex_credentials: /path/to/aws_gcp_credentials.json
      vertex_project: my-gcp-project
      vertex_location: us-central1
```

</TabItem>
</Tabs>

**Supported AWS auth methods:** `aws_role_name`, `aws_profile_name`, `aws_access_key_id`/`aws_secret_access_key`, `aws_web_identity_token`

**Prerequisites:** You need a GCP Workload Identity Pool configured with an AWS provider. [Setup guide](https://cloud.google.com/iam/docs/workload-identity-federation-with-other-clouds#aws)
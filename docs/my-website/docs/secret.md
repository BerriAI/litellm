# Secret Manager
LiteLLM supports reading secrets from Azure Key Vault and Infisical

- [Azure Key Vault](#azure-key-vault)
- Google Key Management Service
- [Infisical Secret Manager](#infisical-secret-manager)
- [.env Files](#env-files)

## Azure Key Vault

### Quick Start

```python 
### Instantiate Azure Key Vault Client ###
from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential

# Set your Azure Key Vault URI
KVUri = os.getenv("AZURE_KEY_VAULT_URI")

# Set your Azure AD application/client ID, client secret, and tenant ID - create an application with permission to call your key vault
client_id = os.getenv("AZURE_CLIENT_ID") 
client_secret = os.getenv("AZURE_CLIENT_SECRET")
tenant_id = os.getenv("AZURE_TENANT_ID") 

# Initialize the ClientSecretCredential
credential = ClientSecretCredential(client_id=client_id, client_secret=client_secret, tenant_id=tenant_id)

# Create the SecretClient using the credential
client = SecretClient(vault_url=KVUri, credential=credential)

### Connect to LiteLLM ###
import litellm
litellm.secret_manager = client

litellm.get_secret("your-test-key")
```

### Usage with OpenAI Proxy Server

1. Install Proxy dependencies 
```bash
pip install 'litellm[proxy]' 'litellm[extra_proxy]'
```

2. Save Azure details in your environment
```bash 
export["AZURE_CLIENT_ID"]="your-azure-app-client-id"
export["AZURE_CLIENT_SECRET"]="your-azure-app-client-secret"
export["AZURE_TENANT_ID"]="your-azure-tenant-id"
export["AZURE_KEY_VAULT_URI"]="your-azure-key-vault-uri"
```

3. Add to proxy config.yaml 
```yaml
model_list: 
    - model_name: "my-azure-models" # model alias 
        litellm_params:
            model: "azure/<your-deployment-name>"
            api_key: "os.environ/AZURE-API-KEY" # reads from key vault - get_secret("AZURE_API_KEY")
            api_base: "os.environ/AZURE-API-BASE" # reads from key vault - get_secret("AZURE_API_BASE")

general_settings:
  use_azure_key_vault: True
```

You can now test this by starting your proxy: 
```bash
litellm --config /path/to/config.yaml
```

[Quick Test Proxy](./proxy/quick_start#using-litellm-proxy---curl-request-openai-package-langchain-langchain-js)

## Google Key Management Service 

Use encrypted keys from Google KMS on the proxy

### Usage with OpenAI Proxy Server

## Step 1. Add keys to env 
```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export GOOGLE_KMS_RESOURCE_NAME="projects/*/locations/*/keyRings/*/cryptoKeys/*"
export PROXY_DATABASE_URL_ENCRYPTED=b'\n$\x00D\xac\xb4/\x8e\xc...'
```

## Step 2: Update Config

```yaml
general_settings:
  use_google_kms: true
  database_url: "os.environ/PROXY_DATABASE_URL_ENCRYPTED"
  master_key: sk-1234
```

## Step 3: Start + test proxy

```
$ litellm --config /path/to/config.yaml
```

And in another terminal
```
$ litellm --test 
```

[Quick Test Proxy](./proxy/quick_start#using-litellm-proxy---curl-request-openai-package-langchain-langchain-js)


## Infisical Secret Manager
Integrates with [Infisical's Secret Manager](https://infisical.com/) for secure storage and retrieval of API keys and sensitive data.

### Usage
liteLLM manages reading in your LLM API secrets/env variables from Infisical for you

```python
import litellm
from infisical import InfisicalClient

litellm.secret_manager = InfisicalClient(token="your-token")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like today?"},
]

response = litellm.completion(model="gpt-3.5-turbo", messages=messages)

print(response)
```


## .env Files
If no secret manager client is specified, Litellm automatically uses the `.env` file to manage sensitive data.

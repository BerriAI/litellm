import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Secret Manager
LiteLLM supports reading secrets from Azure Key Vault, Google Secret Manager

:::info

✨ **This is an Enterprise Feature**

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

## Supported Secret Managers

- AWS Key Management Service
- AWS Secret Manager
- [Azure Key Vault](#azure-key-vault)
- [Google Secret Manager](#google-secret-manager)
- Google Key Management Service
- [Infisical Secret Manager](#infisical-secret-manager)
- [.env Files](#env-files)

## AWS Key Management V1

:::tip

[BETA] AWS Key Management v2 is on the enterprise tier. Go [here for docs](./proxy/enterprise.md#beta-aws-key-manager---key-decryption)

:::

Use AWS KMS to storing a hashed copy of your Proxy Master Key in the environment. 

```bash
export LITELLM_MASTER_KEY="djZ9xjVaZ..." # 👈 ENCRYPTED KEY
export AWS_REGION_NAME="us-west-2"
```

```yaml
general_settings:
  key_management_system: "aws_kms"
  key_management_settings:
    hosted_keys: ["LITELLM_MASTER_KEY"] # 👈 WHICH KEYS ARE STORED ON KMS
```

[**See Decryption Code**](https://github.com/BerriAI/litellm/blob/a2da2a8f168d45648b61279d4795d647d94f90c9/litellm/utils.py#L10182)

## AWS Secret Manager

Store your proxy keys in AWS Secret Manager.

### Proxy Usage

1. Save AWS Credentials in your environment
```bash
os.environ["AWS_ACCESS_KEY_ID"] = ""  # Access key
os.environ["AWS_SECRET_ACCESS_KEY"] = "" # Secret access key
os.environ["AWS_REGION_NAME"] = "" # us-east-1, us-east-2, us-west-1, us-west-2
```

2. Enable AWS Secret Manager in config. 

<Tabs>
<TabItem value="read_only" label="Read Keys from AWS Secret Manager">

```yaml
general_settings:
  master_key: os.environ/litellm_master_key 
  key_management_system: "aws_secret_manager" # 👈 KEY CHANGE
  key_management_settings: 
    hosted_keys: ["litellm_master_key"] # 👈 Specify which env keys you stored on AWS 

```

</TabItem>

<TabItem value="write_only" label="Write Virtual Keys to AWS Secret Manager">

This will only store virtual keys in AWS Secret Manager. No keys will be read from AWS Secret Manager.

```yaml
general_settings:
  key_management_system: "aws_secret_manager" # 👈 KEY CHANGE
  key_management_settings: 
    store_virtual_keys: true
    access_mode: "write_only" # Literal["read_only", "write_only", "read_and_write"]
```
</TabItem>
</Tabs>

3. Run proxy

```bash
litellm --config /path/to/config.yaml
```

## Azure Key Vault
<!-- 
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
``` -->

### Usage with LiteLLM Proxy Server

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
  key_management_system: "azure_key_vault"
```

You can now test this by starting your proxy: 
```bash
litellm --config /path/to/config.yaml
```

[Quick Test Proxy](./proxy/quick_start#using-litellm-proxy---curl-request-openai-package-langchain-langchain-js)

## Google Secret Manager

Support for [Google Secret Manager](https://cloud.google.com/security/products/secret-manager)


1. Save Google Secret Manager details in your environment

```shell 
GOOGLE_SECRET_MANAGER_PROJECT_ID="your-project-id-on-gcp" # example: adroit-crow-413218
```

Optional Params

```shell
export GOOGLE_SECRET_MANAGER_REFRESH_INTERVAL = ""            # (int) defaults to 86400
export GOOGLE_SECRET_MANAGER_ALWAYS_READ_SECRET_MANAGER = ""  # (str) set to "true" if you want to always read from google secret manager without using in memory caching. NOT RECOMMENDED in PROD
```

2. Add to proxy config.yaml 
```yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/fake
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
      api_key: os.environ/OPENAI_API_KEY # this will be read from Google Secret Manager

general_settings:
  key_management_system: "google_secret_manager"
```

You can now test this by starting your proxy: 
```bash
litellm --config /path/to/config.yaml
```

[Quick Test Proxy](./proxy/quick_start#using-litellm-proxy---curl-request-openai-package-langchain-langchain-js)


## Google Key Management Service 

Use encrypted keys from Google KMS on the proxy

Step 1. Add keys to env 
```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export GOOGLE_KMS_RESOURCE_NAME="projects/*/locations/*/keyRings/*/cryptoKeys/*"
export PROXY_DATABASE_URL_ENCRYPTED=b'\n$\x00D\xac\xb4/\x8e\xc...'
```

Step 2: Update Config

```yaml
general_settings:
  key_management_system: "google_kms"
  database_url: "os.environ/PROXY_DATABASE_URL_ENCRYPTED"
  master_key: sk-1234
```

Step 3: Start + test proxy

```
$ litellm --config /path/to/config.yaml
```

And in another terminal
```
$ litellm --test 
```

[Quick Test Proxy](./proxy/quick_start#using-litellm-proxy---curl-request-openai-package-langchain-langchain-js)

<!-- 
## .env Files
If no secret manager client is specified, Litellm automatically uses the `.env` file to manage sensitive data. -->


## All Secret Manager Settings

All settings related to secret management

```yaml
general_settings:
  key_management_system: "aws_secret_manager" # REQUIRED
  key_management_settings:  
    store_virtual_keys: true # OPTIONAL. Defaults to False, when True will store virtual keys in secret manager
    access_mode: "write_only" # OPTIONAL. Literal["read_only", "write_only", "read_and_write"]. Defaults to "read_only"
    hosted_keys: ["litellm_master_key"] # OPTIONAL. Specify which env keys you stored on AWS
```
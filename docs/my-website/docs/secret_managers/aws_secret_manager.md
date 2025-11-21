import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# AWS Secret Manager

:::info

✨ **This is an Enterprise Feature**

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

Store your proxy keys in AWS Secret Manager.

| Feature | Support | Description |
|---------|----------|-------------|
| Reading Secrets | ✅ | Read secrets e.g `OPENAI_API_KEY` |
| Writing Secrets | ✅ | Store secrets e.g `Virtual Keys` |

## Proxy Usage

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
    store_virtual_keys: true # OPTIONAL. Defaults to False, when True will store virtual keys in secret manager
    prefix_for_stored_virtual_keys: "litellm/" # OPTIONAL. If set, this prefix will be used for stored virtual keys in the secret manager
    access_mode: "write_only" # Literal["read_only", "write_only", "read_and_write"]
    description: "litellm virtual key" # OPTIONAL, if set will set this as the description for all virtual keys
    tags: # OPTIONAL, if set will set this as the tags for all virtual keys
      Environment: "Prod"
      Owner: "AI Platform team"
```
</TabItem>
<TabItem value="read_and_write" label="Read + Write Keys with AWS Secret Manager">

```yaml
general_settings:
  master_key: os.environ/litellm_master_key 
  key_management_system: "aws_secret_manager" # 👈 KEY CHANGE
  key_management_settings: 
    store_virtual_keys: true # OPTIONAL. Defaults to False, when True will store virtual keys in secret manager
    prefix_for_stored_virtual_keys: "litellm/" # OPTIONAL. If set, this prefix will be used for stored virtual keys in the secret manager
    access_mode: "read_and_write" # Literal["read_only", "write_only", "read_and_write"]
    hosted_keys: ["litellm_master_key"] # OPTIONAL. Specify which env keys you stored on AWS
```

</TabItem>
</Tabs>

3. Run proxy

```bash
litellm --config /path/to/config.yaml
```

## Using K/V pairs in 1 AWS Secret

You can read multiple keys from a single AWS Secret using the `primary_secret_name` parameter:

```yaml
general_settings:
  key_management_system: "aws_secret_manager"
  key_management_settings:
    hosted_keys: [
      "OPENAI_API_KEY_MODEL_1",
      "OPENAI_API_KEY_MODEL_2",
    ]
    primary_secret_name: "litellm_secrets" # 👈 Read multiple keys from one JSON secret
```

The `primary_secret_name` allows you to read multiple keys from a single AWS Secret as a JSON object. For example, the "litellm_secrets" would contain:

```json
{
  "OPENAI_API_KEY_MODEL_1": "sk-key1...",
  "OPENAI_API_KEY_MODEL_2": "sk-key2..."
}
```

This reduces the number of AWS Secrets you need to manage.

## IAM Role Assumption

Use IAM roles instead of static AWS credentials for better security.

### Basic IAM Role

```yaml
general_settings:
  key_management_system: "aws_secret_manager"
  key_management_settings:
    store_virtual_keys: true
    aws_region_name: "us-east-1"
    aws_role_name: "arn:aws:iam::123456789012:role/LiteLLMSecretManagerRole"
    aws_session_name: "litellm-session"
```

### Cross-Account Access

```yaml
general_settings:
  key_management_system: "aws_secret_manager"
  key_management_settings:
    store_virtual_keys: true
    aws_region_name: "us-east-1"
    aws_role_name: "arn:aws:iam::999999999999:role/CrossAccountRole"
    aws_external_id: "unique-external-id"
```

### EKS with IRSA

```yaml
general_settings:
  key_management_system: "aws_secret_manager"
  key_management_settings:
    store_virtual_keys: true
    aws_region_name: "us-east-1"
    aws_role_name: "arn:aws:iam::123456789012:role/LiteLLMServiceAccountRole"
    aws_web_identity_token: "os.environ/AWS_WEB_IDENTITY_TOKEN_FILE"
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `aws_region_name` | AWS region |
| `aws_role_name` | IAM role ARN to assume |
| `aws_session_name` | Session name (optional) |
| `aws_external_id` | External ID for cross-account |
| `aws_profile_name` | AWS profile from `~/.aws/credentials` |
| `aws_web_identity_token` | OIDC token path for IRSA |
| `aws_sts_endpoint` | Custom STS endpoint for VPC |




# Secret Managers Overview

:::info

✨ **This is an Enterprise Feature**

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

LiteLLM supports **reading secrets (eg. `OPENAI_API_KEY`)** and **writing secrets (eg. Virtual Keys)** from Azure Key Vault, Google Secret Manager, Hashicorp Vault, CyberArk Conjur, and AWS Secret Manager.

## Supported Secret Managers

- [AWS Key Management Service](./secret_managers/aws_kms)
- [AWS Secret Manager](./secret_managers/aws_secret_manager)
- [Azure Key Vault](./secret_managers/azure_key_vault)
- [CyberArk Conjur](./secret_managers/cyberark)
- [Google Secret Manager](./secret_managers/google_secret_manager)
- [Google Key Management Service](./secret_managers/google_kms)
- [Hashicorp Vault](./secret_managers/hashicorp_vault)

## All Secret Manager Settings

All settings related to secret management

```yaml
general_settings:
  key_management_system: "aws_secret_manager" # REQUIRED
  key_management_settings:  

    # Storing Virtual Keys Settings
    store_virtual_keys: true # OPTIONAL. Defaults to False, when True will store virtual keys in secret manager
    prefix_for_stored_virtual_keys: "litellm/" # OPTIONAL.I f set, this prefix will be used for stored virtual keys in the secret manager
    
    # Access Mode Settings
    access_mode: "write_only" # OPTIONAL. Literal["read_only", "write_only", "read_and_write"]. Defaults to "read_only"
    
    # Hosted Keys Settings
    hosted_keys: ["litellm_master_key"] # OPTIONAL. Specify which env keys you stored on AWS

    # K/V pairs in 1 AWS Secret Settings
    primary_secret_name: "litellm_secrets" # OPTIONAL. Read multiple keys from one JSON secret on AWS Secret Manager
```
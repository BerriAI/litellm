import Image from '@theme/IdealImage';

# Secret Managers Overview

:::info

✨ **This is an Enterprise Feature**

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

LiteLLM supports **reading secrets (eg. `OPENAI_API_KEY`)** and **writing secrets (eg. Virtual Keys)** from Azure Key Vault, Google Secret Manager, Hashicorp Vault, CyberArk Conjur, and AWS Secret Manager.

## Supported Secret Managers

- [AWS Key Management Service](./aws_kms)
- [AWS Secret Manager](./aws_secret_manager)
- [Azure Key Vault](./azure_key_vault)
- [CyberArk Conjur](./cyberark)
- [Google Secret Manager](./google_secret_manager)
- [Google Key Management Service](./google_kms)
- [Hashicorp Vault](./hashicorp_vault)

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

## Team-Level Secret Manager Settings

From the **Teams** page in the LiteLLM dashboard you can configure a secret manager per team. Open the team (or the “Create New Team” modal), find the **Secret Manager Settings** panel, and enter the provider-specific JSON configuration (e.g. `{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}`). This configuration is applied whenever LiteLLM writes secrets (e.g., storing virtual keys) on behalf of that team.

<Image img={require('../../img/secret_manager_settings.png')} />


Refer to each provider’s documentation (AWS, Azure, Google, Hashicorp, etc.) for the supported keys/values you can place inside `secret_manager_settings`.

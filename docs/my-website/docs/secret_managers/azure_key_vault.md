# Azure Key Vault

:::info

✨ **This is an Enterprise Feature**

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

## Usage with LiteLLM Proxy Server

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

[Quick Test Proxy](../proxy/quick_start#using-litellm-proxy---curl-request-openai-package-langchain-langchain-js)


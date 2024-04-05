# Azure AI Studio

## Sample Usage
The `azure/` prefix sends this to Azure

Ensure you add `/v1` to your api_base. Your Azure AI studio `api_base` passed to litellm should look something like this
```python
api_base = "https://Mistral-large-dfgfj-serverless.eastus2.inference.ai.azure.com/v1/"
```

```python
import litellm
response = litellm.completion(
    model="azure/command-r-plus",
    api_base="<your-deployment-base>/v1/"
    api_key="eskk******"
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
)
```

## Sample Usage - LiteLLM Proxy

Set this on your litellm proxy config.yaml
```yaml
model_list:
  - model_name: mistral
    litellm_params:
      model: mistral/Mistral-large-dfgfj
      api_base: https://Mistral-large-dfgfj-serverless.eastus2.inference.ai.azure.com/v1/
      api_key: JGbKodRcTp****
  - model_name: command-r-plus
    litellm_params:
        model: azure/command-r-plus
        api_key: os.environ/AZURE_COHERE_API_KEY
        api_base: os.environ/AZURE_COHERE_API_BASE
```

## Supported Models

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| command-r-plus | `completion(model="azure/command-r-plus", messages)` | 
| command-r | `completion(model="azure/command-r", messages)` | 
| mistral-large-latest | `completion(model="azure/mistral-large-latest", messages)` | 




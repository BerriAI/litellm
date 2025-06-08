import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Perplexity AI (pplx-api)
https://www.perplexity.ai

## API Key
```python
# env variable
os.environ['PERPLEXITYAI_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/sonar-pro", 
    messages=messages
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/sonar-pro", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models
All models listed here https://docs.perplexity.ai/docs/model-cards are supported.  Just do `model=perplexity/<model-name>`.

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| sonar-deep-research | `completion(model="perplexity/sonar-deep-research", messages)` | 
| sonar-reasoning-pro | `completion(model="perplexity/sonar-reasoning-pro", messages)` | 
| sonar-reasoning | `completion(model="perplexity/sonar-reasoning", messages)` | 
| sonar-pro | `completion(model="perplexity/sonar-pro", messages)` | 
| sonar | `completion(model="perplexity/sonar", messages)` | 
| r1-1776 | `completion(model="perplexity/r1-1776", messages)` | 






:::info

For more information about passing provider-specific parameters, [go here](../completion/provider_specific_params.md)
:::

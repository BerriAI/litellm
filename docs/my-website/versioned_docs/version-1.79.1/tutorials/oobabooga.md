# Oobabooga Text Web API Tutorial

### Install + Import LiteLLM 
```python 
!pip install litellm
from litellm import completion 
import os
```

### Call your oobabooga model
Remember to set your api_base
```python
response = completion(
  model="oobabooga/WizardCoder-Python-7B-V1.0-GPTQ",
  messages=[{ "content": "can you write a binary tree traversal preorder","role": "user"}], 
  api_base="http://localhost:5000",
  max_tokens=4000
)
```

### See your response 
```python 
print(response)
```

Credits to [Shuai Shao](https://www.linkedin.com/in/shuai-sh/), for this tutorial. 
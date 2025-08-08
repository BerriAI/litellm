import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Parasail
https://parasail.io/

**We support ALL Parasail models, just set `parasail/` as a prefix when sending completion requests**

## API Key
```python
# env variable
os.environ['PARASAIL_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['PARASAIL_API_KEY'] = ""
response = completion(
    model="parasail/your-model", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['PARASAIL_API_KEY'] = ""
response = completion(
    model="parasail/your-model", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Supported Models - ALL Parasail Models Supported!
We support ALL Parasail models, just set `parasail/` as a prefix when sending completion requests

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| parasail-deepseek-r1-0528 | `completion(model="parasail/parasail-deepseek-r1-0528", messages)` | 
| your-model | `completion(model="parasail/your-model", messages)` | 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ['PARASAIL_API_KEY'] = ""
resp = completion(
    model="parasail/parasail-deepseek-r1-0528",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

print(resp.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: parasail-deepseek-r1-0528
    litellm_params:
        model: parasail/parasail-deepseek-r1-0528
        api_key: os.environ/PARASAIL_API_KEY
```

2. Run proxy

```bash
python litellm/proxy/main.py
```

3. Test it!

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "parasail-deepseek-r1-0528",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Hello, how are you?"
          }
        ]
      }
    ]
}'
```

</TabItem>

</Tabs>

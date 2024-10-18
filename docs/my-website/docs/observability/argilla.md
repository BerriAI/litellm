import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Argilla 

Argilla is a tool for annotating datasets. 



## Usage 

<Tabs>
<Tab value="sdk" label="SDK">

```python
from litellm import completion
import litellm
import os 

# add env vars
os.environ["ARGILLA_API_KEY"]="argilla.apikey"
os.environ["ARGILLA_BASE_URL"]="http://localhost:6900"
os.environ["ARGILLA_DATASET_NAME"]="my_second_dataset"   
os.environ["OPENAI_API_KEY"]="sk-proj-..."

litellm.callbacks = ["argilla"]

# add argilla transformation object
litellm.argilla_transformation_object = {
    "user_input": "messages", # 👈 key= argilla field, value = either message (argilla.ChatField) | response (argilla.TextField)
    "llm_output": "response"
}

## LLM CALL ## 
response = completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)
```

</Tab>

<Tab value="proxy" label="PROXY">

```yaml
litellm_settings:
  callbacks: ["argilla"]
  argilla_transformation_object:
    user_input: "messages" # 👈 key= argilla field, value = either message (argilla.ChatField) | response (argilla.TextField)
    llm_output: "response"
```

</Tab>
</Tabs>

## Example Output

<Image img={require('../../img/argilla.png')} />

## Add sampling rate to Argilla calls

To just log a sample of calls to argilla, add `ARGILLA_SAMPLING_RATE` to your env vars.

```bash
ARGILLA_SAMPLING_RATE=0.1 # log 10% of calls to argilla
```
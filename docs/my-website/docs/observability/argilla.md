import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Argilla 

Argilla is a collaborative annotation tool for AI engineers and domain experts who need to build high-quality datasets for their projects.


## Getting Started

To log the data to Argilla, first you need to deploy the Argilla server. If you have not deployed the Argilla server, please follow the instructions [here](https://docs.argilla.io/latest/getting_started/quickstart/).

Next, you will need to configure and create the Argilla dataset.

```python
import argilla as rg

client = rg.Argilla(api_url="<api_url>", api_key="<api_key>")

settings = rg.Settings(
    guidelines="These are some guidelines.",
    fields=[
        rg.ChatField(
            name="user_input",
        ),
        rg.TextField(
            name="llm_output",
        ),
    ],
    questions=[
        rg.RatingQuestion(
            name="rating",
            values=[1, 2, 3, 4, 5, 6, 7],
        ),
    ],
)

dataset = rg.Dataset(
    name="my_first_dataset",
    settings=settings,
)

dataset.create()
```

For further configuration, please refer to the [Argilla documentation](https://docs.argilla.io/latest/how_to_guides/dataset/).


## Usage

<Tabs>
<Tab value="sdk" label="SDK">

```python
import os
import litellm
from litellm import completion

# add env vars
os.environ["ARGILLA_API_KEY"]="argilla.apikey"
os.environ["ARGILLA_BASE_URL"]="http://localhost:6900"
os.environ["ARGILLA_DATASET_NAME"]="my_first_dataset"   
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
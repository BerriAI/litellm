import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Evaluate LLMs - ML Flow Evals, Auto Eval

## Using LiteLLM with ML Flow 
MLflow provides an API `mlflow.evaluate()` to help evaluate your LLMs https://mlflow.org/docs/latest/llms/llm-evaluate/index.html

### Pre Requisites
```shell
pip install litellm
```
```shell
pip install mlflow
```


### Step 1: Start LiteLLM Proxy on the CLI
LiteLLM allows you to create an OpenAI compatible server for all supported LLMs. [More information on litellm proxy here](https://docs.litellm.ai/docs/simple_proxy)

```shell
$ litellm --model huggingface/bigcode/starcoder

#INFO: Proxy running on http://0.0.0.0:8000
```

**Here's how you can create the proxy for other supported llms**
<Tabs>
<TabItem value="bedrock" label="Bedrock">

```shell
$ export AWS_ACCESS_KEY_ID=""
$ export AWS_REGION_NAME="" # e.g. us-west-2
$ export AWS_SECRET_ACCESS_KEY=""
```

```shell
$ litellm --model bedrock/anthropic.claude-v2
```
</TabItem>
<TabItem value="huggingface" label="Huggingface (TGI)">

```shell
$ export HUGGINGFACE_API_KEY=my-api-key #[OPTIONAL]
```
```shell
$ litellm --model huggingface/<your model name> --api_base https://k58ory32yinf1ly0.us-east-1.aws.endpoints.huggingface.cloud
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```shell
$ export ANTHROPIC_API_KEY=my-api-key
```
```shell
$ litellm --model claude-instant-1
```

</TabItem>
<TabItem value="vllm-local" label="VLLM">
Assuming you're running vllm locally

```shell
$ litellm --model vllm/facebook/opt-125m
```
</TabItem>
<TabItem value="openai-proxy" label="OpenAI Compatible Server">

```shell
$ litellm --model openai/<model_name> --api_base <your-api-base>
```
</TabItem>
<TabItem value="together_ai" label="TogetherAI">

```shell
$ export TOGETHERAI_API_KEY=my-api-key
```
```shell
$ litellm --model together_ai/lmsys/vicuna-13b-v1.5-16k
```

</TabItem>

<TabItem value="replicate" label="Replicate">

```shell
$ export REPLICATE_API_KEY=my-api-key
```
```shell
$ litellm \
  --model replicate/meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3
```

</TabItem>

<TabItem value="petals" label="Petals">

```shell
$ litellm --model petals/meta-llama/Llama-2-70b-chat-hf
```

</TabItem>

<TabItem value="palm" label="Palm">

```shell
$ export PALM_API_KEY=my-palm-key
```
```shell
$ litellm --model palm/chat-bison
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```shell
$ export AZURE_API_KEY=my-api-key
$ export AZURE_API_BASE=my-api-base
```
```
$ litellm --model azure/my-deployment-name
```

</TabItem>

<TabItem value="ai21" label="AI21">

```shell
$ export AI21_API_KEY=my-api-key
```

```shell
$ litellm --model j2-light
```

</TabItem>

<TabItem value="cohere" label="Cohere">

```shell
$ export COHERE_API_KEY=my-api-key
```

```shell
$ litellm --model command-nightly
```

</TabItem>

</Tabs>


### Step 2: Run ML Flow
Before running the eval we will set `openai.api_base` to the litellm proxy from Step 1

```python
openai.api_base = "http://0.0.0.0:8000"
```

```python
import openai
import pandas as pd
openai.api_key = "anything"             # this can be anything, we set the key on the proxy
openai.api_base = "http://0.0.0.0:8000" # set api base to the proxy from step 1


import mlflow
eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is the largest country",
            "What is the weather in sf?",
        ],
        "ground_truth": [
            "India is a large country",
            "It's cold in SF today"
        ],
    }
)

with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    logged_model_info = mlflow.openai.log_model(
        model="gpt-3.5",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )

    # Use predefined question-answering metrics to evaluate our model.
    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        targets="ground_truth",
        model_type="question-answering",
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    # Evaluation result for each data record is available in `results.tables`.
    eval_table = results.tables["eval_results_table"]
    print(f"See evaluation table below: \n{eval_table}")


```

### ML Flow Output
```
{'toxicity/v1/mean': 0.00014476531214313582, 'toxicity/v1/variance': 2.5759661361262862e-12, 'toxicity/v1/p90': 0.00014604929747292773, 'toxicity/v1/ratio': 0.0, 'exact_match/v1': 0.0}
Downloading artifacts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1890.18it/s]
See evaluation table below:
                        inputs              ground_truth                                            outputs  token_count  toxicity/v1/score
0  What is the largest country  India is a large country   Russia is the largest country in the world in...           14           0.000146
1   What is the weather in sf?     It's cold in SF today   I'm sorry, I cannot provide the current weath...           36           0.000143
```


## Using LiteLLM with AutoEval
AutoEvals is a tool for quickly and easily evaluating AI model outputs using best practices.
https://github.com/braintrustdata/autoevals

### Pre Requisites
```shell
pip install litellm
```
```shell
pip install autoevals
```

### Quick Start
In this code sample we use the `Factuality()` evaluator from `autoevals.llm` to test whether an output is factual, compared to an original (expected) value.

**Autoevals uses gpt-3.5-turbo / gpt-4-turbo by default to evaluate responses**

See autoevals docs on the [supported evaluators](https://www.braintrustdata.com/docs/autoevals/python#autoevalsllm) - Translation, Summary, Security Evaluators etc

```python
# auto evals imports 
from autoevals.llm import *
###################
import litellm

# litellm completion call
question = "which country has the highest population"
response = litellm.completion(
    model = "gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": question
        }
    ],
)
print(response)
# use the auto eval Factuality() evaluator
evaluator = Factuality()
result = evaluator(
    output=response.choices[0]["message"]["content"],       # response from litellm.completion()
    expected="India",                                       # expected output
    input=question                                          # question passed to litellm.completion
)

print(result)
```

#### Output of Evaluation - from AutoEvals
```shell
Score(
    name='Factuality', 
    score=0, 
    metadata=
        {'rationale': "The expert answer is 'India'.\nThe submitted answer is 'As of 2021, China has the highest population in the world with an estimated 1.4 billion people.'\nThe submitted answer mentions China as the country with the highest population, while the expert answer mentions India.\nThere is a disagreement between the submitted answer and the expert answer.", 
        'choice': 'D'
        }, 
    error=None
)
```












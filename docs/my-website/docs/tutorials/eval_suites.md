import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Evaluate LLMs - MLflow Evals, Auto Eval

## Using LiteLLM with MLflow

[MLflow](https://mlflow.org/docs/latest/genai/eval-monitor.html) provides a powerful capability for evaluating your LLM applications and agents with 50+ built-in quality metrics and custom scoring criteria. This tutorial shows how to use MLflow to evaluate LLM applications and agents powered by LiteLLM.

<Image img={require('../../img/mlflow_evaluation_results.png')} />

### Pre Requisites
```shell
pip install litellm mlflow>=3.3
```

### Step 1: Configure MLflow

In a terminal, start the MLflow server.

```shell
mlflow server --port 5000
```

Then create a new notebook or a Python script to run the evaluation. Import MLflow and set the tracking URI and experiment name.

- **Tracking URI**: The URL of the MLflow server. MLflow will determine where to send evaluation result based on this URI.
- **Experiment**: Experiment is a container in MLflow that groups evaluation runs, metrics, traces, etc. You can think of it as sort of a folder or a project.

```python evaluation.py

import mlflow

mlflow.set_tracking_uri("http://localhost:5000") # <- The MLflow server URL
mlflow.set_experiment("LiteLLM Evaluation") # <- Specify any name you want for your experiment and MLflow will create it if it doesn't exist.
```

### Step 2: Define your inference logic with LiteLLM

First, define a simple function that generates responses by invoking LLM API through LiteLLM.

```python
def predict_fn(question: str) -> str:
    response = litellm.completion(
        model="gpt-5.1-mini",
        messages=[
            {"role": "system", "content": "Answer the following question in two sentences."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
```

:::info

During evaluation, MLflow will automatically **[trace](https://mlflow.org/docs/latest/llms/tracing/index.html)** the LiteLLM calls and store them in the evaluation run. These traces are useful for debugging the root cause of low-quality responses and improve the model performance.

:::

Alternatively, you can use the LiteLLM proxy to create an OpenAI compatible server for all supported LLMs. [More information on litellm proxy here](https://docs.litellm.ai/docs/simple_proxy)

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

```python
import openai

client = openai.OpenAI(
    api_key="anything",            # this can be anything, we set the key on the proxy
    base_url="http://0.0.0.0:8000" # your proxy url
)

def predict_fn(question: str) -> str:
    response = client.chat.completions.create(
        model="<your-model-name>",
        messages=[
            {"role": "system", "content": "Answer the following question in two sentences."},
            {"role": "user", "content": question},
    ])
    return response.choices[0].message.content
```

### Step 3: Prepare the evaluation dataset

Define the evaluation dataset with input questions, and optional expectations (= ground truth answers).

```python
eval_data = [
    {
        "inputs": {"question": "What is the largest country?"},
        "expectations": {"expected_response": "Russia is the largest country by area."},
    },
    {
        "inputs": {"question": "What is the weather in SF?"},
        "expectations": {"expected_response": "I cannot provide real-time weather information."},
    },
]
```


### Step 4: Define evaluation metrics

MLflow provides 50+ built-in evaluation metrics and a flexible API for defining custom ones.

In MLflow, a **scorer** is a class or a function that generates evaluation metrics for a given data record. In this example, we will use two built-in scorers:

- `Correctness`: LLM-as-a-Judge metric to check if the response is correct according to the expectation.
- `Guidelines`: Flexible built-in metric that allow you to define custom LLM-as-a-Judge with a simple natural language guidelines.

```python
from mlflow.genai.scorers import Correctness, Guidelines

scorers = [
    Correctness(),
    Guidelines(name="is_concise", guidelines="The answer must be concise and no longer than two sentences."),
]
```

See [MLflow documentation](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/) for more details about supported scorers.

### Step 5: Run the evaluation

Now we are ready to run the evaluation. Pass the evaluation dataset, prediction function, and scorers to the `mlflow.genai.evaluate` function.

```python
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=scorers,
)
```

### Review the Results

When the evaluation is complete, MLflow will show the link to the evaluation run in the terminal. Open the link in your browser to see the evaluation run and detailed results for each data record.

### Next Steps

- Check out [MLflow LiteLLM Integration](../observability/mlflow.md) for more details about MLflow LiteLLM integration.
- Visit [Scorers Documentation](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/) for the full list of supported scorers and find the one that fits your needs.


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












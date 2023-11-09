# Evaluate LLMs - Auto Eval, MlFlow

## Using LiteLLM with AutoEval
AutoEvals is a tool for quickly and easily evaluating AI model outputs using best practices.
https://github.com/braintrustdata/autoevals

## Pre Requisites
```shell
pip install litellm
```
```shell
pip install autoevals
```

### Quick Start
```python
from autoevals.llm import *
import autoevals

# litellm completion call
import litellm
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

# use the auto eval Factuality() evaluator
evaluator = Factuality()
openai.api_key = "" # set your openai api key for evaluator
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












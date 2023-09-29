# DeepEval - Unit Testing of LLMs

In this section, we will show you how to use litellm with deepeval for testing chatbot responses. We will use the `test_quickstart.py` file as a code example.

First, we need to import the necessary modules from both litellm and deepeval.

```python
import os
from litellm import longer_context_model_fallback_dict, ContextWindowExceededError, completion
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
```

We then create a message and specify the model to use for the completion.
Now, we can call the `completion` function from litellm to generate the code.

```python
api_key = "sk-xxx"
response = completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Hey!"}], 
    stream=True, 
    api_key=api_key
)
```

We can then extract the generated code from the response and use it as the context for our test case.

```python
response = response['choices'][0]['text']
```

Finally, we can use the `assert_test` function from deepeval to check if the generated code is as expected.

```python
query = "What are your operating hours?"
output = response
context = "Our company operates from 10 AM to 6 PM, Monday to Friday."
factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.3)
answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
test_case = LLMTestCase(query=query, output=output, context=context)
assert_test(
    test_case, [factual_consistency_metric, answer_relevancy_metric]
)
```

This is how you can use litellm with deepeval for testing chatbot responses.


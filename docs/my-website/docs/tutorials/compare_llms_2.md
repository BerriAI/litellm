import Image from '@theme/IdealImage';

# Comparing LLMs on a Test Set using LiteLLM


<div class="cell markdown" id="L-W4C3SgClxl">

LiteLLM allows you to use any LLM as a drop in replacement for
`gpt-3.5-turbo`

This notebook walks through how you can compare GPT-4 vs Claude-2 on a
given test set using litellm

## Output at the end of this tutorial:
<Image img={require('../../img/compare_llms.png')} />
<br></br>

</div>

<div class="cell code" id="fBkbl4Qo9pvz">

``` python
!pip install litellm
```

</div>

<div class="cell code" execution_count="16" id="tzS-AXWK8lJC">

``` python
from litellm import completion
import litellm

# init your test set questions
questions = [
    "how do i call completion() using LiteLLM",
    "does LiteLLM support VertexAI",
    "how do I set my keys on replicate llama2?",
]


# set your prompt
prompt = """
You are a coding assistant helping users using litellm.
litellm is a light package to simplify calling OpenAI, Azure, Cohere, Anthropic, Huggingface API Endpoints. It manages:

"""
```

</div>

<div class="cell code" execution_count="18" id="vMlqi40x-KAA">

``` python
import os
os.environ['OPENAI_API_KEY'] = ""
os.environ['ANTHROPIC_API_KEY'] = ""
```

</div>

<div class="cell markdown" id="-HOzUfpK-H8J">

</div>

<div class="cell markdown" id="Ktn25dfKEJF1">

## Calling gpt-3.5-turbo and claude-2 on the same questions

## LiteLLM `completion()` allows you to call all LLMs in the same format

</div>

<div class="cell code" id="DhXwRlc-9DED">

``` python
results = [] # for storing results

models = ['gpt-3.5-turbo', 'claude-2'] # define what models you're testing, see: https://docs.litellm.ai/docs/providers
for question in questions:
    row = [question]
    for model in models:
      print("Calling:", model, "question:", question)
      response = completion( # using litellm.completion
            model=model,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': question}
            ]
      )
      answer = response.choices[0].message['content']
      row.append(answer)
      print(print("Calling:", model, "answer:", answer))

    results.append(row) # save results

```

</div>

<div class="cell markdown" id="RkEXhXxCDN77">

## Visualizing Results

</div>

<div class="cell code" execution_count="15"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:761}"
id="42hrmW6q-n4s" outputId="b763bf39-72b9-4bea-caf6-de6b2412f86d">

``` python
# Create a table to visualize results
import pandas as pd

columns = ['Question'] + models
df = pd.DataFrame(results, columns=columns)

df
```
## Output Table
<Image img={require('../../img/compare_llms.png')} />

</div>

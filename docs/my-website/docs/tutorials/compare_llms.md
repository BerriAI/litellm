import Image from '@theme/IdealImage';

# Benchmark LLMs
Easily benchmark LLMs for a given question by viewing 
* Responses 
* Response Cost
* Response Time

### Benchmark Output
<Image img={require('../../img/bench_llm.png')} />

## Setup:
```
git clone https://github.com/BerriAI/litellm
```
cd to `litellm/cookbook/benchmark` dir

Located here: 
https://github.com/BerriAI/litellm/tree/main/cookbook/benchmark
```
cd litellm/cookbook/benchmark
```

### Install Dependencies
```
pip install litellm click tqdm tabulate termcolor
```

### Configuration - Set LLM API Keys + LLMs in benchmark.py
In `benchmark/benchmark.py` select your LLMs, LLM API Key and questions

Supported LLMs: https://docs.litellm.ai/docs/providers

```python
# Define the list of models to benchmark
models = ['gpt-3.5-turbo', 'claude-2']

# Enter LLM API keys
os.environ['OPENAI_API_KEY'] = ""
os.environ['ANTHROPIC_API_KEY'] = ""

# List of questions to benchmark (replace with your questions)
questions = [
    "When will BerriAI IPO?",
    "When will LiteLLM hit $100M ARR?"
]

```

## Run benchmark.py
```
python3 benchmark.py
```

## Expected Output
```
Running question: When will BerriAI IPO? for model: claude-2: 100%|████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:13<00:00,  4.41s/it]

Benchmark Results for 'When will BerriAI IPO?':
+-----------------+----------------------------------------------------------------------------------+---------------------------+------------+
| Model           | Response                                                                         | Response Time (seconds)   | Cost ($)   |
+=================+==================================================================================+===========================+============+
| gpt-3.5-turbo   | As an AI language model, I cannot provide up-to-date information or predict      | 1.55 seconds              | $0.000122  |
|                 | future events. It is best to consult a reliable financial source or contact      |                           |            |
|                 | BerriAI directly for information regarding their IPO plans.                      |                           |            |
+-----------------+----------------------------------------------------------------------------------+---------------------------+------------+
| togethercompute | I'm not able to provide information about future IPO plans or dates for BerriAI  | 8.52 seconds              | $0.000531  |
| r/llama-2-70b-c | or any other company. IPO (Initial Public Offering) plans and timelines are      |                           |            |
| hat             | typically kept private by companies until they are ready to make a public        |                           |            |
|                 | announcement.  It's important to note that IPO plans can change and are subject  |                           |            |
|                 | to various factors, such as market conditions, financial performance, and        |                           |            |
|                 | regulatory approvals. Therefore, it's difficult to predict with certainty when   |                           |            |
|                 | BerriAI or any other company will go public.  If you're interested in staying    |                           |            |
|                 | up-to-date with BerriAI's latest news and developments, you may want to follow   |                           |            |
|                 | their official social media accounts, subscribe to their newsletter, or visit    |                           |            |
|                 | their website periodically for updates.                                          |                           |            |
+-----------------+----------------------------------------------------------------------------------+---------------------------+------------+
| claude-2        | I do not have any information about when or if BerriAI will have an initial      | 3.17 seconds              | $0.002084  |
|                 | public offering (IPO). As an AI assistant created by Anthropic to be helpful,    |                           |            |
|                 | harmless, and honest, I do not have insider knowledge about Anthropic's business |                           |            |
|                 | plans or strategies.                                                             |                           |            |
+-----------------+----------------------------------------------------------------------------------+---------------------------+------------+
```
## Support
**🤝 Schedule a 1-on-1 Session:** Book a [1-on-1 session](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat) with Krrish and Ishaan, the founders, to discuss any issues, provide feedback, or explore how we can improve LiteLLM for you.


<!-- 
## Pre-requisites:
``` python
!pip install litellm
```

## Example Use Case 1 - Code Generator

### Enter your system prompt and questions
```` python
# enter your system prompt if you have one
system_prompt = """
You are a coding assistant helping users using litellm.
litellm is a light package to simplify calling OpenAI, Azure, Cohere, Anthropic, Huggingface API Endpoints
--
Sample Usage:
```
pip install litellm
from litellm import completion
## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"
messages = [{ "content": "Hello, how are you?","role": "user"}]
# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)
# cohere call
response = completion("command-nightly", messages)
```

"""


# questions/logs you want to run the LLM on
questions = [
    "what is litellm?",
    "why should I use LiteLLM",
    "does litellm support Anthropic LLMs",
    "write code to make a litellm completion call",
]
````

### Running questions

### Select from 100+ LLMs here: <https://docs.litellm.ai/docs/providers> {#select-from-100-llms-here-httpsdocslitellmaidocsproviders}

``` python
import litellm
from litellm import completion, completion_cost
import os
import time

# optional use litellm dashboard to view logs
# litellm.use_client = True
# litellm.token = "ishaan_2@berri.ai" # set your email


# set API keys
os.environ['TOGETHERAI_API_KEY'] = ""
os.environ['OPENAI_API_KEY'] = ""
os.environ['ANTHROPIC_API_KEY'] = ""


# select LLMs to benchmark
# using https://api.together.xyz/playground for llama2
# try any supported LLM here: https://docs.litellm.ai/docs/providers

models = ['togethercomputer/llama-2-70b-chat', 'gpt-3.5-turbo', 'claude-instant-1.2']
data = []

for question in questions: # group by question
  for model in models:
    print(f"running question: {question} for model: {model}")
    start_time = time.time()
    # show response, response time, cost for each question
    response = completion(
        model=model,
        max_tokens=500,
        messages = [
            {
              "role": "system", "content": system_prompt
            },
            {
              "role": "user", "content": question
            }
        ],
    )
    end = time.time()
    total_time = end-start_time # response time
    # print(response)
    cost = completion_cost(response) # cost for completion
    raw_response = response['choices'][0]['message']['content'] # response string


    # add log to pandas df
    data.append(
        {
            'Model': model,
            'Question': question,
            'Response': raw_response,
            'ResponseTime': total_time,
            'Cost': cost
        })
```

### View Benchmarks for LLMs
``` python
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.display import HTML
import pandas as pd

df = pd.DataFrame(data)
grouped_by_question = df.groupby('Question')

for question, group_data in grouped_by_question:
    print(f"Question: {question}")
    HTML(group_data.to_html())
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Model</th>
      <th>Question</th>
      <th>Response</th>
      <th>ResponseTime</th>
      <th>Cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>togethercomputer/llama-2-70b-chat</td>
      <td>what is litellm?</td>
      <td>Litellm is a lightweight Python package that simplifies calling various AI API endpoints, including OpenAI, Azure, Cohere, Anthropic, and Hugging Face. It provides a convenient interface for making requests to these APIs, allowing developers to easily integrate them into their applications. With Litellm, developers can quickly and easily interact with multiple AI models and services, without having to handle the details of authentication, API calls, and response parsing. This makes it easier to build and deploy AI-powered applications, and can help developers save time and effort.</td>
      <td>13.479644</td>
      <td>0.000870</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gpt-3.5-turbo</td>
      <td>what is litellm?</td>
      <td>litellm is a light package that provides a simplified interface for making API calls to various language models and APIs. It abstracts away the complexities of handling network requests, authentication, and response parsing, making it easier for developers to integrate powerful language models into their applications.\n\nWith litellm, you can quickly make API calls to models like OpenAI's GPT-3.5 Turbo, Azure's Text Analytics, Cohere's Command API, Anthropic's API, and Huggingface's models. It also supports additional functionality like conversational AI, summarization, translation, and more.\n\nBy using litellm, you can focus on your application logic without getting tangled in the details of API integration, allowing you to quickly build intelligent and conversational applications.</td>
      <td>8.324332</td>
      <td>0.000566</td>
    </tr>
    <tr>
      <th>2</th>
      <td>claude-instant-1.2</td>
      <td>what is litellm?</td>
      <td>litellm is a Python library that simplifies calling various AI API endpoints like OpenAI, Azure, Cohere, Anthropic, and Huggingface. \n\nSome key things to know about litellm:\n\n- It provides a consistent interface for completing prompts and generating responses from different AI models through a single method called completion().\n\n- You specify the API (e.g. OpenAI, Cohere etc.) and model either by name or by setting environment variables before making the completion call.\n\n- This avoids having to use different SDKs or APIs for each provider and standardizes the call structure. \n\n- It handles things like setting headers, encoding inputs, parsing responses so the user doesn't have to deal with those details.\n\n- The goal is to make it easy to try different AI APIs and models without having to change code or learn different interfaces.\n\n- It's lightweight with no other dependencies required besides what's needed for each API (e.g. openai, azure SDKs etc.).\n\nSo in summary, litellm is a small library that provides a common way to interact with multiple conversational AI APIs through a single Python method, avoiding the need to directly use each provider's specific SDK.</td>
      <td>10.316488</td>
      <td>0.001603</td>
    </tr>
  </tbody>
</table>

## Example Use Case 2 - Rewrite user input concisely

``` python
# enter your system prompt if you have one
system_prompt = """
For a given user input, rewrite the input to make be more concise.
"""

# user input for re-writing questions
questions = [
    "LiteLLM is a lightweight Python package that simplifies the process of making API calls to various language models. Here are some reasons why you should use LiteLLM:nn1. **Simplified API Calls**: LiteLLM abstracts away the complexity of making API calls to different language models. It provides a unified interface for invoking models from OpenAI, Azure, Cohere, Anthropic, Huggingface, and more.nn2. **Easy Integration**: LiteLLM seamlessly integrates with your existing codebase. You can import the package and start making API calls with just a few lines of code.nn3. **Flexibility**: LiteLLM supports a variety of language models, including GPT-3, GPT-Neo, chatGPT, and more. You can choose the model that suits your requirements and easily switch between them.nn4. **Convenience**: LiteLLM handles the authentication and connection details for you. You just need to set the relevant environment variables, and the package takes care of the rest.nn5. **Quick Prototyping**: LiteLLM is ideal for rapid prototyping and experimentation. With its simple API, you can quickly generate text, chat with models, and build interactive applications.nn6. **Community Support**: LiteLLM is actively maintained and supported by a community of developers. You can find help, share ideas, and collaborate with others to enhance your projects.nnOverall, LiteLLM simplifies the process of making API calls to language models, saving you time and effort while providing flexibility and convenience",
    "Hi everyone! I'm [your name] and I'm currently working on [your project/role involving LLMs]. I came across LiteLLM and was really excited by how it simplifies working with different LLM providers. I'm hoping to use LiteLLM to [build an app/simplify my code/test different models etc]. Before finding LiteLLM, I was struggling with [describe any issues you faced working with multiple LLMs]. With LiteLLM's unified API and automatic translation between providers, I think it will really help me to [goals you have for using LiteLLM]. Looking forward to being part of this community and learning more about how I can build impactful applications powered by LLMs!Let me know if you would like me to modify or expand on any part of this suggested intro. I'm happy to provide any clarification or additional details you need!",
    "Traceloop is a platform for monitoring and debugging the quality of your LLM outputs. It provides you with a way to track the performance of your LLM application; rollout changes with confidence; and debug issues in production. It is based on OpenTelemetry, so it can provide full visibility to your LLM requests, as well vector DB usage, and other infra in your stack."
]
```

### Run Questions

``` python
import litellm
from litellm import completion, completion_cost
import os
import time

# optional use litellm dashboard to view logs
# litellm.use_client = True
# litellm.token = "ishaan_2@berri.ai" # set your email

os.environ['TOGETHERAI_API_KEY'] = ""
os.environ['OPENAI_API_KEY'] = ""
os.environ['ANTHROPIC_API_KEY'] = ""

models = ['togethercomputer/llama-2-70b-chat', 'gpt-3.5-turbo', 'claude-instant-1.2'] # enter llms to benchmark
data_2 = []

for question in questions: # group by question
  for model in models:
    print(f"running question: {question} for model: {model}")
    start_time = time.time()
    # show response, response time, cost for each question
    response = completion(
        model=model,
        max_tokens=500,
        messages = [
            {
              "role": "system", "content": system_prompt
            },
            {
              "role": "user", "content": "User input:" + question
            }
        ],
    )
    end = time.time()
    total_time = end-start_time # response time
    # print(response)
    cost = completion_cost(response) # cost for completion
    raw_response = response['choices'][0]['message']['content'] # response string
    #print(raw_response, total_time, cost)

    # add to pandas df
    data_2.append(
        {
            'Model': model,
            'Question': question,
            'Response': raw_response,
            'ResponseTime': total_time,
            'Cost': cost
        })


```
### View Logs - Group by Question
``` python
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.display import HTML
import pandas as pd

df = pd.DataFrame(data_2)
grouped_by_question = df.groupby('Question')

for question, group_data in grouped_by_question:
    print(f"Question: {question}")
    HTML(group_data.to_html())
```

#### User Question
    Question: Hi everyone! I'm [your name] and I'm currently working on [your project/role involving LLMs]. I came across LiteLLM and was really excited by how it simplifies working with different LLM providers. I'm hoping to use LiteLLM to [build an app/simplify my code/test different models etc]. Before finding LiteLLM, I was struggling with [describe any issues you faced working with multiple LLMs]. With LiteLLM's unified API and automatic translation between providers, I think it will really help me to [goals you have for using LiteLLM]. Looking forward to being part of this community and learning more about how I can build impactful applications powered by LLMs!Let me know if you would like me to modify or expand on any part of this suggested intro. I'm happy to provide any clarification or additional details you need!
#### Logs
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Model</th>
      <th>Response</th>
      <th>ResponseTime</th>
      <th>Cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>togethercomputer/llama-2-70b-chat</td>
      <td>nHere's a more concise version of the user input:nn"Hi everyone! I'm [your name] and I'm working on [your project/role involving LLMs]. I recently discovered LiteLLM and I'm excited to use it to [build an app/simplify my code/test different models etc]. Before LiteLLM, I struggled with [describe any issues you faced working with multiple LLMs]. I'm looking forward to using LiteLLM's unified API and automatic translation to achieve my goals. I'm eager to learn more about building impactful applications powered by LLMs and to be part of this community. Let me know if you have any questions or need further clarification."nnIn this revised version, we've kept the essential information and removed some of the extraneous language. We've also rephrased some of the sentences to make them more concise and easier to read.</td>
      <td>18.300620</td>
      <td>0.001200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gpt-3.5-turbo</td>
      <td>User input: Hi, I'm [your name] and I'm excited about using LiteLLM to simplify working with different LLM providers. Before finding LiteLLM, I faced challenges working with multiple LLMs. With LiteLLM's unified API and automatic translation, I believe it will help me achieve my goals of [state your goals]. I look forward to being part of this community and learning how to build impactful applications with LLMs. Let me know if you need any further clarification or details.</td>
      <td>7.385472</td>
      <td>0.000525</td>
    </tr>
    <tr>
      <th>5</th>
      <td>claude-instant-1.2</td>
      <td>Here is a more concise rewrite of the user input:nnHi everyone, I'm [your name]. I'm currently [your project/role] and came across LiteLLM, which simplifies working with different LLMs through its unified API. I hope to [build an app/simplify code/test models] with LiteLLM since I previously struggled with [issues]. LiteLLM's automatic translation between providers will help me [goals] and build impactful LLM applications. Looking forward to learning more as part of this community. Let me know if you need any clarification on my plans to use LiteLLM.</td>
      <td>8.628217</td>
      <td>0.001022</td>
    </tr>
  </tbody>
</table> -->

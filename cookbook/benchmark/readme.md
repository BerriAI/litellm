<h1 align="center">
        LLM-Bench
    </h1>
    <p align="center">
        <p align="center">Benchmark LLMs response, cost and response time</p>
        <p>LLM vs Cost per input + output token ($)</p>
        <img width="806" alt="Screenshot 2023-11-13 at 2 51 06 PM" src="https://github.com/BerriAI/litellm/assets/29436595/6d1bed71-d062-40b8-a113-28359672636a">
    </p>
        <a href="https://docs.google.com/spreadsheets/d/1mvPbP02OLFgc-5-Ubn1KxGuQQdbMyG1jhMSWxAldWy4/edit?usp=sharing">
               Bar Graph Excel Sheet here
        </a>

| Model | Provider | Cost per input + output token ($)|
| --- | --- | --- |
| openrouter/mistralai/mistral-7b-instruct | openrouter | 0.0 |
| ollama/llama2 | ollama | 0.0 |
| ollama/llama2:13b | ollama | 0.0 |
| ollama/llama2:70b | ollama | 0.0 |
| ollama/llama2-uncensored | ollama | 0.0 |
| ollama/mistral | ollama | 0.0 |
| ollama/codellama | ollama | 0.0 |
| ollama/orca-mini | ollama | 0.0 |
| ollama/vicuna | ollama | 0.0 |
| perplexity/codellama-34b-instruct | perplexity | 0.0 |
| perplexity/llama-2-13b-chat | perplexity | 0.0 |
| perplexity/llama-2-70b-chat | perplexity | 0.0 |
| perplexity/mistral-7b-instruct | perplexity | 0.0 |
| perplexity/replit-code-v1.5-3b | perplexity | 0.0 |
| text-bison | vertex_ai-text-models | 0.00000025 |
| text-bison@001 | vertex_ai-text-models | 0.00000025 |
| chat-bison | vertex_ai-chat-models | 0.00000025 |
| chat-bison@001 | vertex_ai-chat-models | 0.00000025 |
| chat-bison-32k | vertex_ai-chat-models | 0.00000025 |
| code-bison | vertex_ai-code-text-models | 0.00000025 |
| code-bison@001 | vertex_ai-code-text-models | 0.00000025 |
| code-gecko@001 | vertex_ai-chat-models | 0.00000025 |
| code-gecko@latest | vertex_ai-chat-models | 0.00000025 |
| codechat-bison | vertex_ai-code-chat-models | 0.00000025 |
| codechat-bison@001 | vertex_ai-code-chat-models | 0.00000025 |
| codechat-bison-32k | vertex_ai-code-chat-models | 0.00000025 |
| palm/chat-bison | palm | 0.00000025 |
| palm/chat-bison-001 | palm | 0.00000025 |
| palm/text-bison | palm | 0.00000025 |
| palm/text-bison-001 | palm | 0.00000025 |
| palm/text-bison-safety-off | palm | 0.00000025 |
| palm/text-bison-safety-recitation-off | palm | 0.00000025 |
| anyscale/meta-llama/Llama-2-7b-chat-hf | anyscale | 0.0000003 |
| anyscale/mistralai/Mistral-7B-Instruct-v0.1 | anyscale | 0.0000003 |
| openrouter/meta-llama/llama-2-13b-chat | openrouter | 0.0000004 |
| openrouter/nousresearch/nous-hermes-llama2-13b | openrouter | 0.0000004 |
| deepinfra/meta-llama/Llama-2-7b-chat-hf | deepinfra | 0.0000004 |
| deepinfra/mistralai/Mistral-7B-Instruct-v0.1 | deepinfra | 0.0000004 |
| anyscale/meta-llama/Llama-2-13b-chat-hf | anyscale | 0.0000005 |
| amazon.titan-text-lite-v1 | bedrock | 0.0000007 |
| deepinfra/meta-llama/Llama-2-13b-chat-hf | deepinfra | 0.0000007 |
| text-babbage-001 | text-completion-openai | 0.0000008 |
| text-ada-001 | text-completion-openai | 0.0000008 |
| babbage-002 | text-completion-openai | 0.0000008 |
| openrouter/google/palm-2-chat-bison | openrouter | 0.000001 |
| openrouter/google/palm-2-codechat-bison | openrouter | 0.000001 |
| openrouter/meta-llama/codellama-34b-instruct | openrouter | 0.000001 |
| deepinfra/codellama/CodeLlama-34b-Instruct-hf | deepinfra | 0.0000012 |
| deepinfra/meta-llama/Llama-2-70b-chat-hf | deepinfra | 0.0000016499999999999999 |
| deepinfra/jondurbin/airoboros-l2-70b-gpt4-1.4.1 | deepinfra | 0.0000016499999999999999 |
| anyscale/meta-llama/Llama-2-70b-chat-hf | anyscale | 0.000002 |
| anyscale/codellama/CodeLlama-34b-Instruct-hf | anyscale | 0.000002 |
| gpt-3.5-turbo-1106 | openai | 0.000003 |
| openrouter/meta-llama/llama-2-70b-chat | openrouter | 0.000003 |
| amazon.titan-text-express-v1 | bedrock | 0.000003 |
| gpt-3.5-turbo | openai | 0.0000035 |
| gpt-3.5-turbo-0301 | openai | 0.0000035 |
| gpt-3.5-turbo-0613 | openai | 0.0000035 |
| gpt-3.5-turbo-instruct | text-completion-openai | 0.0000035 |
| openrouter/openai/gpt-3.5-turbo | openrouter | 0.0000035 |
| cohere.command-text-v14 | bedrock | 0.0000035 |
| gpt-3.5-turbo-0613 | openai | 0.0000035 |
| claude-instant-1 | anthropic | 0.00000714 |
| claude-instant-1.2 | anthropic | 0.00000714 |
| openrouter/anthropic/claude-instant-v1 | openrouter | 0.00000714 |
| anthropic.claude-instant-v1 | bedrock | 0.00000714 |
| openrouter/mancer/weaver | openrouter | 0.00001125 |
| j2-mid | ai21 | 0.00002 |
| ai21.j2-mid-v1 | bedrock | 0.000025 |
| openrouter/jondurbin/airoboros-l2-70b-2.1 | openrouter | 0.00002775 |
| command-nightly | cohere | 0.00003 |
| command | cohere | 0.00003 |
| command-light | cohere | 0.00003 |
| command-medium-beta | cohere | 0.00003 |
| command-xlarge-beta | cohere | 0.00003 |
| j2-ultra | ai21 | 0.00003 |
| ai21.j2-ultra-v1 | bedrock | 0.0000376 |
| gpt-4-1106-preview | openai | 0.00004 |
| gpt-4-vision-preview | openai | 0.00004 |
| claude-2 | anthropic | 0.0000437 |
| openrouter/anthropic/claude-2 | openrouter | 0.0000437 |
| anthropic.claude-v1 | bedrock | 0.0000437 |
| anthropic.claude-v2 | bedrock | 0.0000437 |
| gpt-4 | openai | 0.00009 |
| gpt-4-0314 | openai | 0.00009 |
| gpt-4-0613 | openai | 0.00009 |
| openrouter/openai/gpt-4 | openrouter | 0.00009 |
| gpt-4-32k | openai | 0.00018 |
| gpt-4-32k-0314 | openai | 0.00018 |
| gpt-4-32k-0613 | openai | 0.00018 |



## Setup:
```
git clone https://github.com/BerriAI/litellm
```
cd to `benchmark` dir
```
cd litellm/cookbook/benchmark
```

### Install Dependencies
```
pip install litellm click tqdm tabulate termcolor
```

### Configuration
In `benchmark/benchmark.py` select your LLMs, LLM API Key and questions

Supported LLMs: https://docs.litellm.ai/docs/providers

```python
# Define the list of models to benchmark
models = ['gpt-3.5-turbo', 'togethercomputer/llama-2-70b-chat', 'claude-2']

# Enter LLM API keys
os.environ['OPENAI_API_KEY'] = ""
os.environ['ANTHROPIC_API_KEY'] = ""
os.environ['TOGETHERAI_API_KEY'] = ""

# List of questions to benchmark (replace with your questions)
questions = [
    "When will BerriAI IPO?",
    "When will LiteLLM hit $100M ARR?"
]

```

## Run LLM-Bench
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

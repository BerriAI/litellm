<h1 align="center">
        LLM-Bench
    </h1>
    <p align="center">
        <p align="center">Benchmark LLMs response, cost and response time</p>
    </p>

<h4 align="center">
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
    </a>
    <a href="https://pypi.org/project/litellm/0.1.1/" target="_blank">
        <img src="https://img.shields.io/badge/stable%20version-v0.1.424-blue?color=green&link=https://pypi.org/project/litellm/0.1.1/" alt="Stable Version">
    </a>
    <a href="https://dl.circleci.com/status-badge/redirect/gh/BerriAI/litellm/tree/main" target="_blank">
        <img src="https://dl.circleci.com/status-badge/img/gh/BerriAI/litellm/tree/main.svg?style=svg" alt="CircleCI">
    </a>
    <img src="https://img.shields.io/pypi/dm/litellm" alt="Downloads">
    <a href="https://discord.gg/wuPM9dRgDw" target="_blank">
        <img src="https://dcbadge.vercel.app/api/server/wuPM9dRgDw?style=flat">
    </a>
    <a href="https://www.ycombinator.com/companies/berriai">
        <img src="https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square" alt="Y Combinator W23">
    </a>
<img width="1156" alt="Screenshot 2023-09-08 at 6 31 55 AM" src="https://github.com/BerriAI/litellm/assets/29436595/269a9c35-fc5f-4173-87eb-d73fc9322538">

</h4>

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

# Benchmark LLMs using LiteLLM
<img width="1280" alt="Screenshot 2023-09-07 at 8 54 26 PM" src="https://github.com/BerriAI/litellm/assets/29436595/cfcc0a3c-5395-4976-8e3c-5e2e328ce88b">

## Usage
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


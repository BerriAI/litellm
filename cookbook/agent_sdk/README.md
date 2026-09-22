# LiteLLM Agent SDK

This is a minimal provider-independent agent interface. Its core loop follows the Claude Agent SDK pattern: configure options, call `query()`, and consume an async message stream

```python
from litellm.agent_sdk import AgentOptions, AssistantMessage, TextBlock, query

options = AgentOptions(
    model="openai/gpt-5.4-mini",
    system_prompt="Review code changes concisely",
)

async for message in query(prompt="Review this diff", options=options):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)
```

Model names use LiteLLM's `provider/model` format, so the same code works with OpenAI, Anthropic, Gemini, Bedrock, Azure, hosted models, and LiteLLM proxy model aliases. Set `model_router` on `AgentOptions` to choose a model before every turn

## PR risk agent

`PRRiskAgent` classifies a pull request as low, medium, or high risk. It routes routine changes to a fast model and large or security-sensitive changes to a stronger model

```python
from litellm.agent_sdk import PRRiskAgent, PullRequest

agent = PRRiskAgent(
    routine_model="openai/gpt-5.4-mini",
    complex_model="anthropic/claude-opus-4-8",
)

assessment = await agent.classify(
    PullRequest(
        title="Add API key rotation",
        body="Rotates keys without downtime",
        diff=diff,
        changed_files=4,
        additions=120,
        deletions=35,
    )
)
```

The included command accepts a PR diff on standard input, which makes it usable from a GitHub Actions job or a local checkout

```shell
gh pr diff 123 | python -m cookbook.agent_sdk.pr_risk_agent \
  --title "Add API key rotation" \
  --changed-files 4 \
  --additions 120 \
  --deletions 35
```

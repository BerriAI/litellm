import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenWeb UI with LiteLLM

This guide walks you through connecting OpenWeb UI to LiteLLM. Using LiteLLM with OpenWeb UI allows teams to 
- Access 100+ LLMs on OpenWeb UI
- Track Spend / Usage, Set Budget Limits 
- Send Request/Response Logs to logging destinations like langfuse, s3, gcs buckets, etc.
- Set access controls eg. Team 1 can only access gpt-4o, Team 2 can only access o1-preview

## Quickstart


### Connect OpenWeb UI to LiteLLM

- OpenWebUI starts running on [http://localhost:3000](http://localhost:3000)
- LiteLLM starts running on [http://localhost:4000](http://localhost:4000)

### Create a Virtual Key on LiteLLM 

Navigate to [http://localhost:4000/ui](http://localhost:4000/ui) and create a new virtual Key. 

LiteLLM allows you to specify what models are available on OpenWeb UI (by specifying the models the key will have access to).

### Connect OpenWeb UI to LiteLLM

#### Navigate to Settings 

#### Click on Connections 

#### Enter LiteLLM API Base and Virtual Key

#### Click on Save

#### Test it

## Render `thinking` content on OpenWeb UI

OpenWebUI requires reasoning/thinking content to be rendered with `<think></think>` tags. In order to render this for specific models, you can use the `merge_reasoning_content_in_choices` litellm parameter.

Example litellm config.yaml:

```yaml
model_list:
  - model_name: thinking-anthropic-claude-3-7-sonnet
    litellm_params:
      model: bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0
      thinking: {"type": "enabled", "budget_tokens": 1024}
      max_tokens: 1080
      merge_reasoning_content_in_choices: true
```

### Test it on OpenWeb UI

On the models dropdown select `thinking-anthropic-claude-3-7-sonnet`

<Image img={require('../../img/litellm_thinking_openweb.gif')} />





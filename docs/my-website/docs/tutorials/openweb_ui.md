import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenWeb UI with LiteLLM

This guide walks you through connecting OpenWeb UI to LiteLLM. Using LiteLLM with OpenWeb UI allows teams to 
- Access 100+ LLMs on OpenWeb UI
- Track Spend / Usage, Set Budget Limits 
- Send Request/Response Logs to logging destinations like langfuse, s3, gcs buckets, etc.
- Set access controls eg. Control what models OpenWebUI can access.

## Quickstart

- Make sure to setup LiteLLM with the [LiteLLM Getting Started Guide](https://docs.litellm.ai/docs/proxy/docker_quick_start)


## 1. Start LiteLLM & OpenWebUI

- OpenWebUI starts running on [http://localhost:3000](http://localhost:3000)
- LiteLLM starts running on [http://localhost:4000](http://localhost:4000)


## 2. Create a Virtual Key on LiteLLM

Virtual Keys are API Keys that allow you to authenticate to LiteLLM Proxy. We will create a Virtual Key that will allow OpenWebUI to access LiteLLM.

### 2.1 LiteLLM User Management Hierarchy

On LiteLLM, you can create Organizations, Teams, Users and Virtual Keys. For this tutorial, we will create a Team and a Virtual Key.

- `Organization` - An Organization is a group of Teams. (US Engineering, EU Developer Tools)
- `Team` - A Team is a group of Users. (OpenWeb UI Team, Data Science Team, etc.)
- `User` - A User is an individual user (employee, developer, eg. `krrish@litellm.ai`)
- `Virtual Key` - A Virtual Key is an API Key that allows you to authenticate to LiteLLM Proxy. A Virtual Key is associated with a User or Team.

Once the Team is created, you can invite Users to the Team. You can read more about LiteLLM's User Management [here](https://docs.litellm.ai/docs/proxy/user_management_heirarchy).

### 2.2 Create a Team on LiteLLM

Navigate to [http://localhost:4000/ui](http://localhost:4000/ui) and create a new team.

<Image img={require('../../img/litellm_create_team.gif')} />

### 2.2 Create a Virtual Key on LiteLLM

Navigate to [http://localhost:4000/ui](http://localhost:4000/ui) and create a new virtual Key. 

LiteLLM allows you to specify what models are available on OpenWeb UI (by specifying the models the key will have access to).

<Image img={require('../../img/create_key_in_team_oweb.gif')} />

## 3. Connect OpenWeb UI to LiteLLM

On OpenWeb UI, navigate to Settings -> Connections and create a new connection to LiteLLM

Enter the following details:
- URL: `http://localhost:4000` (your litellm proxy base url)
- Key: `your-virtual-key` (the key you created in the previous step)

<Image img={require('../../img/litellm_setup_openweb.gif')} />

### 3.1 Test Request

On the top left corner, select models you should only see the models you gave the key access to in Step 2.

Once you selected a model, enter your message content and click on `Submit`

<Image img={require('../../img/basic_litellm.gif')} />

### 3.2 Tracking Spend / Usage

After your request is made, navigate to `Logs` on the LiteLLM UI, you can see Team, Key, Model, Usage and Cost.

<!-- <Image img={require('../../img/litellm_logs_openweb.gif')} /> -->



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

## Additional Resources
- Running LiteLLM and OpenWebUI on Windows Localhost: A Comprehensive Guide [https://www.tanyongsheng.com/note/running-litellm-and-openwebui-on-windows-localhost-a-comprehensive-guide/](https://www.tanyongsheng.com/note/running-litellm-and-openwebui-on-windows-localhost-a-comprehensive-guide/)
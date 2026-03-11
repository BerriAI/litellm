import Image from '@theme/IdealImage';

# A2A Agent Cost Tracking

LiteLLM supports adding custom cost tracking for A2A agents. You can configure:

- **Flat cost per query** - A fixed cost charged for each agent request
- **Cost by input/output tokens** - Variable cost based on token usage

This allows you to track and attribute costs for agent usage across your organization, making it easy to see how much each team or project is spending on agent calls.

## Quick Start

### 1. Navigate to Agents

From the sidebar, click on "Agents" to open the agent management page.

![Navigate to Agents](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/f9ac0752-6936-4dda-b7ed-f536fefcc79a/ascreenshot.jpeg?tl_px=208,326&br_px=2409,1557&force_format=jpeg&q=100&width=1120.0)

### 2. Create a New Agent

Click "+ Add New Agent" to open the creation form. You'll need to provide a few basic details:

- **Agent Name** - A unique identifier for your agent (used in API calls)
- **Display Name** - A human-readable name shown in the UI

![Enter Agent Name](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/f5bacfeb-67a0-4644-a400-b3d50b6b9ce5/ascreenshot.jpeg?tl_px=0,0&br_px=2617,1463&force_format=jpeg&q=100&width=1120.0)

![Enter Display Name](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/6db6422b-fe85-4a8b-aa5c-39319f0d4621/ascreenshot.jpeg?tl_px=0,27&br_px=2617,1490&force_format=jpeg&q=100&width=1120.0)

### 3. Configure Cost Settings

Scroll down and click on "Cost Configuration" to expand the cost settings panel. This is where you define how much to charge for agent usage.

![Click Cost Configuration](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/a3019ae8-629c-431b-b2d8-2743cc517be7/ascreenshot.jpeg?tl_px=0,653&br_px=2201,1883&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=388,416)

### 4. Set Cost Per Query

Enter the cost per query amount (in dollars). For example, entering `0.05` means each request to this agent will be charged $0.05.

![Set Cost Per Query](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/91159f8a-1f66-4555-a166-600e4bdecc68/ascreenshot.jpeg?tl_px=0,653&br_px=2201,1883&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=372,281)

![Enter Cost Amount](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/2add2f69-fd72-462e-9335-1e228c7150da/ascreenshot.jpeg?tl_px=0,420&br_px=2617,1884&force_format=jpeg&q=100&width=1120.0)

### 5. Create the Agent

Once you've configured everything, click "Create Agent" to save. Your agent is now ready to use with cost tracking enabled.

![Create Agent](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/1876cf29-b8a7-4662-b944-2b86a8b7cd2e/ascreenshot.jpeg?tl_px=416,653&br_px=2618,1883&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=706,523)

## Testing Cost Tracking

Let's verify that cost tracking is working by sending a test request through the Playground.

### 1. Go to Playground

Click "Playground" in the sidebar to open the interactive testing interface.

![Go to Playground](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/7d5d8338-6393-49a5-b255-86aef5bf5dfa/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=41,98)

### 2. Select A2A Endpoint

By default, the Playground uses the chat completions endpoint. To test your agent, click "Endpoint Type" and select `/v1/a2a/message/send` from the dropdown.

![Select Endpoint Type](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/4d066510-0878-4e0b-8abf-0b074fe2a560/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=325,238)

![Select A2A Endpoint](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/fe2f8957-4e8a-4331-b177-d5093480cf60/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=333,261)

### 3. Select Your Agent

Now pick the agent you just created from the agent dropdown. You should see it listed by its display name.

![Select Agent](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/8c7add70-fe72-48cb-ba33-9f53b989fcad/ascreenshot.jpeg?tl_px=0,150&br_px=2201,1381&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=287,277)

### 4. Send a Test Message

Type a message and hit send. You can use the suggested prompts or write your own.

![Send Message](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/2c16acb1-4016-447e-88e9-c4522e408ea2/ascreenshot.jpeg?tl_px=15,653&br_px=2216,1883&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,443)

Once the agent responds, the request is logged with the cost you configured.

![Agent Response](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/2dcf7109-0be4-4d03-8333-ef45759c70c9/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=494,273)

## Viewing Cost in Logs

Now let's confirm the cost was actually tracked.

### 1. Navigate to Logs

Click "Logs" in the sidebar to see all recent requests.

![Go to Logs](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/c96abf3c-f06a-4401-ada6-04b6e8040453/ascreenshot.jpeg?tl_px=0,118&br_px=2201,1349&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=41,277)

### 2. View Cost Attribution

Find your agent request in the list. You'll see the cost column showing the amount you configured. This cost is now attributed to the API key that made the request, so you can track spend per team or project.

![View Cost in Logs](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-13/1ae167ec-1a43-48a3-9251-43d4cb3e57f5/ascreenshot.jpeg?tl_px=335,11&br_px=2536,1242&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,277)

## View Spend in Usage Page

Navigate to the Agent Usage tab in the Admin UI to view agent-level spend analytics:

### 1. Access Agent Usage

Go to the Usage page in the Admin UI (`PROXY_BASE_URL/ui/?login=success&page=new_usage`) and click on the **Agent Usage** tab.

<Image img={require('../img/agent_usage_ui_navigation.png')} />

### 2. View Agent Analytics

The Agent Usage dashboard provides:

- **Total spend per agent**: View aggregated spend across all agents
- **Daily spend trends**: See how agent spend changes over time
- **Model usage breakdown**: Understand which models each agent uses
- **Activity metrics**: Track requests, tokens, and success rates per agent

<Image img={require('../img/agent_usage_analytics.png')} />

### 3. Filter by Agent

Use the agent filter dropdown to view spend for specific agents:

- Select one or more agent IDs from the dropdown
- View filtered analytics, spend logs, and activity metrics
- Compare spend across different agents

<Image img={require('../img/agent_usage_filter.png')} />

## Cost Configuration Options

You can mix and match these options depending on your pricing model:

| Field                         | Description                               |
| ----------------------------- | ----------------------------------------- |
| **Cost Per Query ($)**        | Fixed cost charged for each agent request |
| **Input Cost Per Token ($)**  | Cost per input token processed            |
| **Output Cost Per Token ($)** | Cost per output token generated           |

For most use cases, a flat cost per query is simplest. Use token-based pricing if your agent costs vary significantly based on input/output length.

## Related

- [A2A Agent Gateway](./a2a.md)
- [Spend Tracking](./proxy/cost_tracking.md)

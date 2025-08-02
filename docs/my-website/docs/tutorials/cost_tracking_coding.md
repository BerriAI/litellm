import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# Track Usage for Coding Tools

Track usage and costs for AI-powered coding tools like Claude Code, Roo Code, Gemini CLI, and OpenAI Codex through LiteLLM.

Monitor requests, costs, and user engagement metrics for each coding tool using User-Agent headers.

<Image 
  img={require('../../img/agent_1.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>


## Who This Is For

Central AI Platform teams providing developers access to coding tools through LiteLLM. Monitor tool engagement and track individual user usage patterns.

## What You Can Track

### Summary Metrics
- Cost per coding tool
- Successful requests and token usage per tool

### User Engagement Metrics
- Daily, weekly, and monthly active users for each User-Agent 

## Quick Start

### 1. Connect Your Coding Tool to LiteLLM

Configure your coding tool to send requests through the LiteLLM proxy with appropriate User-Agent headers.

**Setup guides:**
- [Use LiteLLM with Claude Code](../../docs/tutorials/claude_responses_api)
- [Use LiteLLM with Gemini CLI](../../docs/tutorials/litellm_gemini_cli)
- [Use LiteLLM with OpenAI Codex](../../docs/tutorials/openai_codex)

### 2. Send Requests with User-Agent Headers

Ensure your coding tool includes identifying User-Agent headers in API requests.

### 3. Verify Tracking in LiteLLM Logs

Confirm LiteLLM is properly tracking requests by checking logs for the expected User-Agent values.

<Image 
  img={require('../../img/agent_2.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

### 4. View Usage Dashboard

Access the LiteLLM dashboard to view aggregated usage metrics and user engagement data.

#### Summary Metrics

View total cost and successful requests for each coding tool.

<Image 
  img={require('../../img/agent_3.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

#### Daily, Weekly, and Monthly Active Users

View active user metrics for each coding tool.

<Image 
  img={require('../../img/agent_4.png')}
  style={{width: '80%', display: 'block', margin: '2rem auto'}}
/>

## How LiteLLM Identifies Coding Tools

LiteLLM tracks coding tools by monitoring the `User-Agent` header in incoming API requests (`/chat/completions`, `/responses`, etc.). Each unique User-Agent is tracked separately for usage analytics.

### Example Request

Example using `claude-cli` as the User-Agent:

```shell
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -H "User-Agent: claude-cli/1.0" \
  -d '{"model": "claude-3-5-sonnet-latest", "messages": [{"role": "user", "content": "Hello, how are you?"}]}' \
  http://localhost:4000/chat/completions
```

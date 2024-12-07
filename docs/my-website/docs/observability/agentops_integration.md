# Agent Monitoring and Debugging with AgentOps

With [AgentOps](https://github.com/AgentOps-AI/agentops), you can track LLM calls, costs, latency, errors, and how agents interact—all in one place, for any provider.


![AgentOps Dashboard](https://raw.githubusercontent.com/AgentOps-AI/agentops/80c8a1d386e8dd9a688c9c45cee9beae9ab40a56/docs/images/external/litellm/dashboard.png "AgentOps Dashboard")

| Feature | Description |
| ------- | ----------- |
| 📊 **Analytics & Debugging** | End-to-end session tracking, agent failure monitoring, and visual timelines |
| 💸 **Cost Management** | Real-time spend tracking across providers |
| ⚡ **Performance Monitoring** | Latency metrics, reliability tracking, and LLM call analytics |
| 🔍 **Prompt Analytics** | Optimize prompts based on usage data |
| 🤖 **Agent Insights** | Track multi-agent interactions and tool usage patterns |
| 📈 **Session Statistics** | Comprehensive session-wide metrics and performance data |

## Installation & Setup

### 1. Install Required Packages

```bash
pip install agentops litellm
```

### 2. Initialize AgentOps
Get your API key from the [AgentOps Dashboard](https://app.agentops.ai).

```python
import agentops
agentops.init(api_key="your-api-key-here")
```

> Optionally, you can set `AGENTOPS_API_KEY` in your environment variables.

### 3. Start monitoring

```python
import litellm

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```
That's it! Your LiteLLM calls are now being monitored. View your analytics at [app.agentops.ai](https://app.agentops.ai)

## Resources & Support

- [📚 Documentation](https://docs.agentops.ai)
- [💬 Discord Community](https://discord.com/invite/FagdcwwXRR)
- [🤝 Schedule Demo](https://cal.com/team/agency-ai/meet-agentops)
- [✉️ Email Support](mailto:support@agentops.ai)

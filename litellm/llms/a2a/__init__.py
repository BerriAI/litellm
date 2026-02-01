"""
A2A (Agent-to-Agent) Provider for LiteLLM

This provider enables calling A2A-compliant agents through the standard LiteLLM completion API.
It transforms OpenAI chat completion requests to A2A protocol and vice versa.

Agent Resolution:
    1. Looks up the agent from LiteLLM's global agent registry by name
    2. Falls back to explicit api_base parameter
    3. Falls back to A2A_AGENT_API_BASE environment variable

Registering Agents:
    Agents can be registered in LiteLLM via:
    - config.yaml `agent_config` list
    - Database (via POST /agent/new API)
    - Programmatic registration via global_agent_registry

Usage with registered agents (recommended):
    import litellm

    # Agent is registered in LiteLLM config or database
    response = litellm.completion(
        model="a2a_agent/my-registered-agent",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Usage with unregistered agents (fallback):
    import litellm

    response = litellm.completion(
        model="a2a_agent/external-agent",
        messages=[{"role": "user", "content": "Hello!"}],
        api_base="http://localhost:9999",  # Required for unregistered agents
    )

For streaming:
    response = litellm.completion(
        model="a2a_agent/my-agent",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )
    for chunk in response:
        print(chunk.choices[0].delta.content)
"""

from litellm.llms.a2a.chat.transformation import A2AAgentConfig

__all__ = ["A2AAgentConfig"]

"""
A2A (Agent-to-Agent) Provider for LiteLLM

This provider enables calling A2A-compliant agents through the standard LiteLLM completion API.
It transforms OpenAI chat completion requests to A2A protocol and vice versa.

Usage:
    import litellm
    
    response = litellm.completion(
        model="a2a_agent/my-agent",
        messages=[{"role": "user", "content": "Hello!"}],
        api_base="http://localhost:9999",  # A2A agent endpoint
    )

For streaming:
    response = litellm.completion(
        model="a2a_agent/my-agent", 
        messages=[{"role": "user", "content": "Hello!"}],
        api_base="http://localhost:9999",
        stream=True,
    )
    for chunk in response:
        print(chunk.choices[0].delta.content)
"""

from litellm.llms.a2a.chat.transformation import A2AAgentConfig

__all__ = ["A2AAgentConfig"]

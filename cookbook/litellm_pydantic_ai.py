"""
Using LiteLLM with Pydantic AI

This script demonstrates how to use LiteLLM's proxy and Router with Pydantic AI agents.

Prerequisites:
    pip install litellm "pydantic-ai-slim[openai]"

Steps:
    1. (Option A) Start the LiteLLM proxy server and point Pydantic AI at it
    2. (Option B) Use LiteLLM Router directly (native LiteLLM failover, no proxy server)

For Option A, first run:
    export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"
    litellm --config litellm_config.yaml --port 4000 --host 127.0.0.1
"""

import os
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# ============================================================
# Option A: LiteLLM Proxy Server
# ============================================================
# The proxy exposes an OpenAI-compatible endpoint at http://localhost:4000/v1
# which Pydantic AI consumes through OpenAIProvider/OpenAIChatModel.
#
# Save this as litellm_config.yaml:
#
# model_list:
#   - model_name: gpt-4o
#     litellm_params:
#       model: openai/gpt-4o
#       api_key: os.environ/OPENAI_API_KEY
#   - model_name: claude-sonnet
#     litellm_params:
#       model: anthropic/claude-3-5-sonnet-20241022
#       api_key: os.environ/ANTHROPIC_API_KEY
#   - model_name: gemini-pro
#     litellm_params:
#       model: gemini/gemini-2.0-flash
#       api_key: os.environ/GEMINI_API_KEY
# general_settings:
#   master_key: os.environ/LITELLM_MASTER_KEY
#
# Start the proxy bound to localhost only (the default 0.0.0.0 listens on all
# interfaces — don't expose this example proxy publicly, anyone with the key
# could send requests through it):
#   export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"
#   litellm --config litellm_config.yaml --port 4000 --host 127.0.0.1


PROXY_BASE_URL = "http://localhost:4000/v1"
# Same key you started the proxy with. Generate a strong one per the comment
# above instead of hardcoding a secret here.
PROXY_API_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-test")


def proxy_model(model_name: str) -> OpenAIChatModel:
    """Build a Pydantic AI model routed through the local LiteLLM proxy."""
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=PROXY_BASE_URL,
            api_key=PROXY_API_KEY,
        ),
    )


def basic_usage():
    agent = Agent(proxy_model("gpt-4o"))
    result = agent.run_sync("What is the capital of France?")
    print("Basic usage:", result.data)


def structured_output():
    class City(BaseModel):
        name: str
        country: str
        population: int

    agent = Agent(
        proxy_model("gpt-4o"),
        result_type=list[City],
        system_prompt="List the 3 largest cities in Europe with their countries and populations.",
    )
    result = agent.run_sync("Generate the list")
    for city in result.data:
        print(f"  {city.name}, {city.country} - Population: {city.population:,}")


def tool_usage():
    def get_weather(ctx: RunContext, city: str) -> str:
        return f"The weather in {city} is sunny, 72 degrees F."

    agent = Agent(
        proxy_model("gpt-4o"),
        tools=[get_weather],
    )
    result = agent.run_sync("What is the weather in Paris?")
    print("Tool usage:", result.data)


def switch_models():
    claude_agent = Agent(proxy_model("claude-sonnet"))
    result = claude_agent.run_sync("Explain quantum computing in one sentence.")
    print(f"[Claude]: {result.data}")

    gemini_agent = Agent(proxy_model("gemini-pro"))
    result = gemini_agent.run_sync("Explain quantum computing in one sentence.")
    print(f"[Gemini]: {result.data}")


# ============================================================
# Option B: LiteLLM Router (Direct Python, no proxy server)
# ============================================================
# This path uses LiteLLM's Router natively (failover / load balancing across
# providers) without running a proxy server and without a Pydantic AI Agent.


def router_usage():
    from litellm import Router

    model_list = [
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": os.environ.get("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "claude-sonnet",
            "litellm_params": {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            },
        },
    ]

    router = Router(model_list=model_list)
    response = router.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say hello!"}],
    )
    print("Router usage:", response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    print("=== LiteLLM + Pydantic AI Cookbook ===")
    print()

    print("--- Basic Usage ---")
    basic_usage()
    print()

    print("--- Structured Output ---")
    structured_output()
    print()

    print("--- Tool Usage ---")
    tool_usage()
    print()

    print("--- Switch Models ---")
    switch_models()
    print()

    print("--- Router (Direct Python) ---")
    router_usage()

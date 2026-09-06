"""
Using LiteLLM with Pydantic AI

This script demonstrates how to use LiteLLM's proxy and Router with Pydantic AI agents.

Prerequisites:
    pip install litellm pydantic-ai

Steps:
    1. (Option A) Start the LiteLLM proxy server and point Pydantic AI at it
    2. (Option B) Use LiteLLM Router directly with Pydantic AI's completion

For Option A, first run:
    litellm --config litellm_config.yaml --port 4000
"""

import os
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext


# ============================================================
# Option A: LiteLLM Proxy Server
# ============================================================
# The proxy exposes an OpenAI-compatible endpoint at http://localhost:4000/v1
# which Pydantic AI's openai provider can consume directly.
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
#   master_key: sk-litellm-test
#
# Start the proxy:
#   litellm --config litellm_config.yaml --port 4000


PROXY_BASE_URL = "http://localhost:4000/v1"
PROXY_API_KEY = "sk-litellm-test"


def basic_usage():
    agent = Agent(
        "openai:gpt-4o",
        base_url=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
    )
    result = agent.run_sync("What is the capital of France?")
    print("Basic usage:", result.data)


def structured_output():
    class City(BaseModel):
        name: str
        country: str
        population: int

    agent = Agent(
        "openai:gpt-4o",
        base_url=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
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
        "openai:gpt-4o",
        base_url=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
        tools=[get_weather],
    )
    result = agent.run_sync("What is the weather in Paris?")
    print("Tool usage:", result.data)


def switch_models():
    claude_agent = Agent(
        "openai:claude-sonnet",
        base_url=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
    )
    result = claude_agent.run_sync("Explain quantum computing in one sentence.")
    print(f"[Claude]: {result.data}")

    gemini_agent = Agent(
        "openai:gemini-pro",
        base_url=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
    )
    result = gemini_agent.run_sync("Explain quantum computing in one sentence.")
    print(f"[Gemini]: {result.data}")


# ============================================================
# Option B: LiteLLM Router (Direct Python, no proxy server)
# ============================================================


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
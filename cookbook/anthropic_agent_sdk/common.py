"""
Common utilities for Claude Agent SDK examples
"""

import os
import httpx


class Config:
    """Configuration for LiteLLM Gateway connection"""
    
    # LiteLLM proxy URL (default to local instance)
    LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
    
    # LiteLLM API key (master key or virtual key)
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")
    
    # Model name as configured in LiteLLM (e.g., "bedrock-claude-sonnet-4", "gpt-4", etc.)
    LITELLM_MODEL = os.getenv("LITELLM_MODEL", "bedrock-claude-sonnet-4.5")


async def fetch_available_models(base_url: str, api_key: str) -> list[str]:
    """
    Fetch available models from LiteLLM proxy /models endpoint
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch models from proxy: {e}")
        print("Using default model list...")
        # Fallback to default models
        return [
            "bedrock-claude-sonnet-3.5",
            "bedrock-claude-sonnet-4",
            "bedrock-claude-sonnet-4.5",
            "bedrock-claude-opus-4.5",
            "bedrock-nova-premier",
        ]


def setup_litellm_env(config: Config):
    """
    Configure environment variables to point Agent SDK to LiteLLM
    """
    litellm_base_url = config.LITELLM_PROXY_URL.rstrip('/')
    os.environ["ANTHROPIC_BASE_URL"] = litellm_base_url
    os.environ["ANTHROPIC_API_KEY"] = config.LITELLM_API_KEY
    return litellm_base_url


def print_header(base_url: str, current_model: str, has_mcp: bool = False):
    """
    Print the chat header
    """
    mcp_indicator = " + MCP" if has_mcp else ""
    print("=" * 70)
    print(f"🤖 Claude Agent SDK with LiteLLM Gateway{mcp_indicator} - Interactive Chat")
    print("=" * 70)
    print(f"🚀 Connected to: {base_url}")
    print(f"📦 Current model: {current_model}")
    if has_mcp:
        print("🔌 MCP: deepwiki2 enabled")
    print("\nType your messages below. Commands:")
    print("  - 'quit' or 'exit' to end the conversation")
    print("  - 'clear' to start a new conversation")
    print("  - 'model' to switch models")
    print("  - 'models' to list available models")
    print("=" * 70)
    print()


def handle_model_list(available_models: list[str], current_model: str):
    """
    Display available models
    """
    print("\n📋 Available models:")
    for i, model in enumerate(available_models, 1):
        marker = "✓" if model == current_model else " "
        print(f"  {marker} {i}. {model}")


def handle_model_switch(available_models: list[str], current_model: str) -> tuple[str, bool]:
    """
    Handle model switching
    
    Returns:
        tuple: (new_model, should_restart_conversation)
    """
    print("\n📋 Select a model:")
    for i, model in enumerate(available_models, 1):
        marker = "✓" if model == current_model else " "
        print(f"  {marker} {i}. {model}")
    
    try:
        choice = input("\nEnter number (or press Enter to cancel): ").strip()
        if choice:
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                new_model = available_models[idx]
                print(f"\n✅ Switched to: {new_model}")
                print("🔄 Starting new conversation with new model...\n")
                return new_model, True
            else:
                print("❌ Invalid choice")
    except (ValueError, IndexError):
        print("❌ Invalid input")
    
    return current_model, False


async def stream_response(client, user_input: str):
    """
    Stream response from the agent
    """
    print("\n🤖 Assistant: ", end='', flush=True)
    
    try:
        await client.query(user_input)
        
        # Show loading indicator
        print("⏳ thinking...", end='', flush=True)
        
        # Stream the response
        first_chunk = True
        async for msg in client.receive_response():
            # Clear loading indicator on first message
            if first_chunk:
                print("\r🤖 Assistant: ", end='', flush=True)
                first_chunk = False
            
            # Handle different message types
            if hasattr(msg, 'type'):
                if msg.type == 'content_block_delta':
                    # Streaming text delta
                    if hasattr(msg, 'delta') and hasattr(msg.delta, 'text'):
                        print(msg.delta.text, end='', flush=True)
                elif msg.type == 'content_block_start':
                    # Start of content block
                    if hasattr(msg, 'content_block') and hasattr(msg.content_block, 'text'):
                        print(msg.content_block.text, end='', flush=True)
            
            # Fallback to original content handling
            if hasattr(msg, 'content'):
                for content_block in msg.content:
                    if hasattr(content_block, 'text'):
                        print(content_block.text, end='', flush=True)
        
        print()  # New line after response
        
    except Exception as e:
        print(f"\r\n❌ Error: {e}")
        print("Please check your LiteLLM gateway is running and configured correctly.")

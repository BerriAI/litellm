"""
Simple Interactive Claude Agent SDK CLI using LiteLLM Gateway

This example demonstrates an interactive CLI chat with the Anthropic Agent SDK using LiteLLM as a proxy.
LiteLLM acts as a unified interface, allowing you to use any LLM provider (OpenAI, Azure, Bedrock, etc.)
through the Claude Agent SDK by pointing it to the LiteLLM gateway.
"""

import os
import asyncio
import httpx
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions


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


async def interactive_chat():
    """
    Interactive CLI chat with the agent
    """
    config = Config()
    
    # Configure Anthropic SDK to point to LiteLLM gateway
    # Note: We don't add /anthropic to the base URL - LiteLLM handles routing
    litellm_base_url = config.LITELLM_PROXY_URL.rstrip('/')
    os.environ["ANTHROPIC_BASE_URL"] = litellm_base_url
    os.environ["ANTHROPIC_API_KEY"] = config.LITELLM_API_KEY
    
    # Fetch available models from proxy
    available_models = await fetch_available_models(litellm_base_url, config.LITELLM_API_KEY)
    
    current_model = config.LITELLM_MODEL
    
    print("=" * 70)
    print("🤖 Claude Agent SDK with LiteLLM Gateway - Interactive Chat")
    print("=" * 70)
    print(f"🚀 Connected to: {litellm_base_url}")
    print(f"📦 Current model: {current_model}")
    print("\nType your messages below. Commands:")
    print("  - 'quit' or 'exit' to end the conversation")
    print("  - 'clear' to start a new conversation")
    print("  - 'model' to switch models")
    print("  - 'models' to list available models")
    print("=" * 70)
    print()
    
    while True:
        # Configure agent options for each conversation
        options = ClaudeAgentOptions(
            system_prompt="You are a helpful AI assistant. Be concise, accurate, and friendly.",
            model=current_model,
            max_turns=50,
        )
        
        # Create agent client
        async with ClaudeSDKClient(options=options) as client:
            conversation_active = True
            
            while conversation_active:
                # Get user input
                try:
                    user_input = input("\n👤 You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Goodbye!")
                    return
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    return
                
                if user_input.lower() == 'clear':
                    print("\n🔄 Starting new conversation...\n")
                    conversation_active = False
                    continue
                
                if user_input.lower() == 'models':
                    print("\n📋 Available models:")
                    for i, model in enumerate(available_models, 1):
                        marker = "✓" if model == current_model else " "
                        print(f"  {marker} {i}. {model}")
                    continue
                
                if user_input.lower() == 'model':
                    print("\n📋 Select a model:")
                    for i, model in enumerate(available_models, 1):
                        marker = "✓" if model == current_model else " "
                        print(f"  {marker} {i}. {model}")
                    
                    try:
                        choice = input("\nEnter number (or press Enter to cancel): ").strip()
                        if choice:
                            idx = int(choice) - 1
                            if 0 <= idx < len(available_models):
                                current_model = available_models[idx]
                                print(f"\n✅ Switched to: {current_model}")
                                print("🔄 Starting new conversation with new model...\n")
                                conversation_active = False
                            else:
                                print("❌ Invalid choice")
                    except (ValueError, IndexError):
                        print("❌ Invalid input")
                    continue
                
                if not user_input:
                    continue
                
                # Send query to agent with loading indicator
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


def main():
    """Run interactive chat"""
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()

"""
Simple Interactive Claude Agent SDK CLI using LiteLLM Gateway

This example demonstrates an interactive CLI chat with the Anthropic Agent SDK using LiteLLM as a proxy.
LiteLLM acts as a unified interface, allowing you to use any LLM provider (OpenAI, Azure, Bedrock, etc.)
through the Claude Agent SDK by pointing it to the LiteLLM gateway.
"""

import os
import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions


class Config:
    """Configuration for LiteLLM Gateway connection"""
    
    # LiteLLM proxy URL (default to local instance)
    LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
    
    # LiteLLM API key (master key or virtual key)
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")
    
    # Model name as configured in LiteLLM (e.g., "bedrock-claude-sonnet-4", "gpt-4", etc.)
    LITELLM_MODEL = os.getenv("LITELLM_MODEL", "bedrock-claude-sonnet-4.5")


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
    
    print("=" * 70)
    print("🤖 Claude Agent SDK with LiteLLM Gateway - Interactive Chat")
    print("=" * 70)
    print(f"🚀 Connected to: {litellm_base_url}")
    print(f"📦 Using model: {config.LITELLM_MODEL}")
    print("\nType your messages below. Commands:")
    print("  - 'quit' or 'exit' to end the conversation")
    print("  - 'clear' to start a new conversation")
    print("=" * 70)
    print()
    
    while True:
        # Configure agent options for each conversation
        options = ClaudeAgentOptions(
            system_prompt="You are a helpful AI assistant. Be concise, accurate, and friendly.",
            model=config.LITELLM_MODEL,
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
                
                if not user_input:
                    continue
                
                # Send query to agent
                print("\n🤖 Assistant: ", end='', flush=True)
                
                try:
                    await client.query(user_input)
                    
                    # Stream the response
                    async for msg in client.receive_response():
                        if hasattr(msg, 'content'):
                            for content_block in msg.content:
                                if hasattr(content_block, 'text'):
                                    print(content_block.text, end='', flush=True)
                    
                    print()  # New line after response
                    
                except Exception as e:
                    print(f"\n\n❌ Error: {e}")
                    print("Please check your LiteLLM gateway is running and configured correctly.")


def main():
    """Run interactive chat"""
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()

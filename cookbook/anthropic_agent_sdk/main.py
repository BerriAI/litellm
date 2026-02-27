"""
Simple Interactive Claude Agent SDK CLI using LiteLLM Gateway

This example demonstrates an interactive CLI chat with the Anthropic Agent SDK using LiteLLM as a proxy.
LiteLLM acts as a unified interface, allowing you to use any LLM provider (OpenAI, Azure, Bedrock, etc.)
through the Claude Agent SDK by pointing it to the LiteLLM gateway.
"""

import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from common import (
    Config,
    fetch_available_models,
    setup_litellm_env,
    print_header,
    handle_model_list,
    handle_model_switch,
    stream_response,
)


async def interactive_chat():
    """
    Interactive CLI chat with the agent
    """
    config = Config()
    
    # Configure Anthropic SDK to point to LiteLLM gateway
    litellm_base_url = setup_litellm_env(config)
    
    # Fetch available models from proxy
    available_models = await fetch_available_models(litellm_base_url, config.LITELLM_API_KEY)
    
    current_model = config.LITELLM_MODEL
    
    print_header(litellm_base_url, current_model)
    
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
                    handle_model_list(available_models, current_model)
                    continue
                
                if user_input.lower() == 'model':
                    new_model, should_restart = handle_model_switch(available_models, current_model)
                    if should_restart:
                        current_model = new_model
                        conversation_active = False
                    continue
                
                if not user_input:
                    continue
                
                # Stream response from agent
                await stream_response(client, user_input)


def main():
    """Run interactive chat"""
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()

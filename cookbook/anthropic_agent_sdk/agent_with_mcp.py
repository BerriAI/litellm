"""
Interactive Claude Agent SDK CLI with MCP Support

This example demonstrates an interactive CLI chat with the Anthropic Agent SDK using LiteLLM as a proxy,
with MCP (Model Context Protocol) server integration for enhanced capabilities.
"""

import asyncio
import os
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


async def interactive_chat_with_mcp():
    """
    Interactive CLI chat with the agent and MCP server
    """
    config = Config()
    
    # Configure Anthropic SDK to point to LiteLLM gateway
    litellm_base_url = setup_litellm_env(config)
    
    # Fetch available models from proxy
    available_models = await fetch_available_models(litellm_base_url, config.LITELLM_API_KEY)
    
    current_model = config.LITELLM_MODEL
    
    # MCP server configuration
    mcp_server_url = f"{litellm_base_url}/mcp/deepwiki2"
    use_mcp = os.getenv("USE_MCP", "true").lower() == "true"
    
    if not use_mcp:
        print("⚠️  MCP disabled via USE_MCP=false")
    
    print_header(litellm_base_url, current_model, has_mcp=use_mcp)
    
    while True:
        # Configure agent options
        if use_mcp:
            try:
                # Try with MCP server (HTTP transport)
                # Using McpHttpServerConfig format from Agent SDK
                options = ClaudeAgentOptions(
                    system_prompt="You are a helpful AI assistant with access to DeepWiki for research. Be concise, accurate, and friendly.",
                    model=current_model,
                    max_turns=50,
                    mcp_servers={
                        "deepwiki2": {
                            "type": "http",
                            "url": mcp_server_url,
                            "headers": {
                                "Authorization": f"Bearer {config.LITELLM_API_KEY}"
                            }
                        }
                    },
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not configure MCP server: {e}")
                print("Continuing without MCP...\n")
                use_mcp = False
                options = ClaudeAgentOptions(
                    system_prompt="You are a helpful AI assistant. Be concise, accurate, and friendly.",
                    model=current_model,
                    max_turns=50,
                )
        else:
            # Without MCP
            options = ClaudeAgentOptions(
                system_prompt="You are a helpful AI assistant. Be concise, accurate, and friendly.",
                model=current_model,
                max_turns=50,
            )
        
        # Create agent client
        try:
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
        
        except Exception as e:
            print(f"\n❌ Error creating agent client: {e}")
            print("This might be an MCP configuration issue. Try running without MCP:")
            print("  USE_MCP=false python agent_with_mcp.py")
            print("\nOr use the basic agent:")
            print("  python main.py")
            return


def main():
    """Run interactive chat with MCP"""
    try:
        asyncio.run(interactive_chat_with_mcp())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()

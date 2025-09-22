#!/usr/bin/env python3
"""
Example: Using CLI token with LiteLLM SDK

This example shows how to use the CLI authentication token
in your Python scripts after running `litellm-proxy login`.
"""

import litellm


def main():
    """Using CLI token with LiteLLM SDK"""
    print("🚀 Using CLI Token with LiteLLM SDK")
    print("=" * 40)
    
    # Get the CLI token
    api_key = litellm.get_litellm_gateway_api_key()
    
    if not api_key:
        print("❌ No CLI token found. Please run 'litellm-proxy login' first.")
        return
    
    print(f"✅ Found CLI token: {api_key[:20]}...")
    
    # Use with LiteLLM
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello from CLI token!"}],
            api_key=api_key,
            base_url="https://your-proxy.com/v1"  # Replace with your proxy URL
        )
        print(f"✅ Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
    
    print("\n💡 Tips:")
    print("1. Run 'litellm-proxy login' to authenticate first")
    print("2. Replace 'https://your-proxy.com' with your actual proxy URL")
    print("3. The token is stored locally at ~/.litellm/token.json")

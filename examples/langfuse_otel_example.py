"""
Example: Using Langfuse OpenTelemetry Integration with LiteLLM

This example demonstrates how to use the Langfuse OpenTelemetry integration
to send traces and observability data to Langfuse.

Prerequisites:
1. Set up your Langfuse account and get your public/secret keys
2. Install required dependencies: pip install litellm opentelemetry-api opentelemetry-sdk

Environment Variables:
- LANGFUSE_PUBLIC_KEY: Your Langfuse public key (required)
- LANGFUSE_SECRET_KEY: Your Langfuse secret key (required)
- LANGFUSE_HOST: Langfuse host URL (optional, defaults to US cloud)
"""

import os
import litellm
from litellm import completion

# Set up environment variables (replace with your actual keys)
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."  # Replace with your public key
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."  # Replace with your secret key

# Optional: Set custom host (defaults to US cloud: https://us.cloud.langfuse.com)
# os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # EU cloud
# os.environ["LANGFUSE_HOST"] = "https://your-langfuse-instance.com"  # Self-hosted

# Configure LiteLLM to use Langfuse OTEL integration
litellm.callbacks = ["langfuse_otel"]

def main():
    """Example usage of LiteLLM with Langfuse OTEL integration."""
    
    print("🚀 Starting Langfuse OTEL integration example...")
    
    try:
        # Make a simple completion request
        response = completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        print("✅ Completion successful!")
        print(f"Response: {response.choices[0].message.content}")
        print("\n📊 Trace data has been sent to Langfuse via OpenTelemetry")
        
        # Make another request with different parameters
        response2 = completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Explain quantum computing in simple terms."}
            ],
            temperature=0.3,
            max_tokens=150,
            metadata={
                "user_id": "example_user",
                "session_id": "example_session"
            }
        )
        
        print("✅ Second completion successful!")
        print(f"Response: {response2.choices[0].message.content[:100]}...")
        print("\n📊 Second trace data has been sent to Langfuse")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have set the required environment variables:")
        print("- LANGFUSE_PUBLIC_KEY")
        print("- LANGFUSE_SECRET_KEY")
        print("- OPENAI_API_KEY (for the OpenAI model)")

if __name__ == "__main__":
    main() 
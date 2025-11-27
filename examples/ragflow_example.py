"""
Example usage of RAGFlow provider with LiteLLM

RAGFlow is an open-source RAG engine based on deep document understanding.
This example shows how to use RAGFlow with LiteLLM.

Requirements:
1. RAGFlow instance running (default: http://localhost:9380)
2. RAGFlow API key
3. RAGFlow agent ID or dataset ID
"""

import os
from litellm import completion

# Set your RAGFlow credentials
os.environ["RAGFLOW_API_KEY"] = "your-ragflow-api-key"
os.environ["RAGFLOW_API_BASE"] = "http://localhost:9380/v1"  # Optional, defaults to localhost

# Example 1: Basic RAGFlow query
def basic_example():
    """Simple query to RAGFlow agent"""
    response = completion(
        model="ragflow/your-agent-id",  # Replace with your RAGFlow agent ID
        messages=[
            {"role": "user", "content": "How does deep document understanding work in RAGFlow?"}
        ]
    )
    print("Response:", response.choices[0].message.content)

# Example 2: Streaming response
def streaming_example():
    """Stream responses from RAGFlow"""
    response = completion(
        model="ragflow/your-agent-id",
        messages=[
            {"role": "user", "content": "Explain the RAG architecture"}
        ],
        stream=True
    )
    
    print("Streaming response:")
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

# Example 3: With custom parameters
def custom_params_example():
    """Use RAGFlow with custom parameters"""
    response = completion(
        model="ragflow/your-agent-id",
        messages=[
            {"role": "user", "content": "What are the key features of RAGFlow?"}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    print("Response:", response.choices[0].message.content)

# Example 4: Multi-turn conversation
def conversation_example():
    """Multi-turn conversation with RAGFlow"""
    messages = [
        {"role": "user", "content": "What is RAGFlow?"},
    ]
    
    # First query
    response = completion(
        model="ragflow/your-agent-id",
        messages=messages
    )
    
    # Add response to conversation
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    
    print("First response:", response.choices[0].message.content)
    
    # Follow-up query
    messages.append({
        "role": "user",
        "content": "How does it compare to other RAG solutions?"
    })
    
    response = completion(
        model="ragflow/your-agent-id",
        messages=messages
    )
    
    print("\nFollow-up response:", response.choices[0].message.content)

# Example 5: Using with different agents/datasets
def multiple_agents_example():
    """Query different RAGFlow agents or datasets"""
    agents = [
        "ragflow/agent-id-1",
        "ragflow/dataset-id-2",
    ]
    
    query = "What information do you have?"
    
    for agent in agents:
        print(f"\nQuerying {agent}:")
        response = completion(
            model=agent,
            messages=[{"role": "user", "content": query}]
        )
        print(response.choices[0].message.content)

if __name__ == "__main__":
    print("RAGFlow Examples")
    print("=" * 50)
    
    # Uncomment the examples you want to run
    # basic_example()
    # streaming_example()
    # custom_params_example()
    # conversation_example()
    # multiple_agents_example()
    
    print("\nNote: Replace 'your-agent-id' with your actual RAGFlow agent ID")
    print("Note: Make sure RAGFlow is running and RAGFLOW_API_KEY is set")

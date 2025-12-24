"""
Example: Using Databricks Model Hub Integration

This example demonstrates how to fetch and use Databricks serving endpoints
through the LiteLLM proxy model hub instead of relying on the static cost map.

Prerequisites:
1. LiteLLM proxy server running
2. Databricks credentials configured (see below)
3. At least one serving endpoint deployed in your Databricks workspace
"""

import os
import requests
from typing import List, Dict

# Configuration
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")

# Databricks credentials (choose one authentication method)

# Method 1: Personal Access Token (Development)
DATABRICKS_API_KEY = os.getenv("DATABRICKS_API_KEY", "dapi...")
DATABRICKS_API_BASE = os.getenv(
    "DATABRICKS_API_BASE",
    "https://adb-xxx.azuredatabricks.net/serving-endpoints",
)

# Method 2: OAuth M2M (Production - Recommended)
# DATABRICKS_CLIENT_ID = os.getenv("DATABRICKS_CLIENT_ID", "your-client-id")
# DATABRICKS_CLIENT_SECRET = os.getenv("DATABRICKS_CLIENT_SECRET", "your-secret")
# DATABRICKS_API_BASE = os.getenv("DATABRICKS_API_BASE", "https://adb-xxx.azuredatabricks.net/serving-endpoints")


def fetch_databricks_endpoints() -> Dict:
    """
    Fetch all Databricks serving endpoints from the Databricks API.
    
    Returns:
        Dict containing endpoints list and workspace URL
    """
    print("📡 Fetching Databricks serving endpoints...")
    
    url = f"{LITELLM_PROXY_URL}/public/databricks/serving_endpoints"
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Note: Credentials are read from environment variables by the proxy
    # You can also pass them as query parameters if needed:
    # params = {
    #     "api_key": DATABRICKS_API_KEY,
    #     "api_base": DATABRICKS_API_BASE,
    # }
    # response = requests.get(url, headers=headers, params=params)
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    print(f"✅ Found {len(data.get('endpoints', []))} endpoints")
    return data


def get_model_hub_databricks() -> List[Dict]:
    """
    Get model hub information filtered for Databricks models.
    
    This will return actual Databricks serving endpoints instead of
    models from the LiteLLM cost map.
    
    Returns:
        List of model group information
    """
    print("📚 Fetching Databricks models from model hub...")
    
    url = f"{LITELLM_PROXY_URL}/public/model_hub"
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
    }
    params = {"provider": "databricks"}
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    models = response.json()
    print(f"✅ Found {len(models)} Databricks models")
    return models


def use_databricks_model(model_name: str) -> Dict:
    """
    Make a completion request using a Databricks model.
    
    Args:
        model_name: The model name (e.g., "databricks/your-endpoint-name")
    
    Returns:
        Completion response
    """
    print(f"🤖 Making request to {model_name}...")
    
    url = f"{LITELLM_PROXY_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello! Please respond with a brief greeting."}
        ],
        "max_tokens": 50,
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    print(f"✅ Response: {result['choices'][0]['message']['content']}")
    return result


def main():
    """Main example flow."""
    print("=" * 60)
    print("Databricks Model Hub Integration Example")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Fetch Databricks endpoints directly
        print("Step 1: Fetch Databricks Serving Endpoints")
        print("-" * 60)
        endpoints_data = fetch_databricks_endpoints()
        
        print("\nEndpoints found:")
        for endpoint in endpoints_data.get("endpoints", []):
            print(f"  • {endpoint['name']} (State: {endpoint['state']})")
            print(f"    URL: {endpoint['endpoint_url']}")
        print()
        
        # Step 2: Get models from model hub (filtered for Databricks)
        print("Step 2: Get Databricks Models from Model Hub")
        print("-" * 60)
        models = get_model_hub_databricks()
        
        print("\nModels in hub:")
        for model in models:
            print(f"  • {model['model_group']}")
            print(f"    Providers: {', '.join(model['providers'])}")
            print(f"    Mode: {model['mode']}")
            print(f"    Function Calling: {model['supports_function_calling']}")
        print()
        
        # Step 3: Use one of the models (if available)
        if models:
            print("Step 3: Test Using a Databricks Model")
            print("-" * 60)
            first_model = models[0]["model_group"]
            result = use_databricks_model(first_model)
            print()
        
        print("=" * 60)
        print("✅ Example completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        if e.response is not None:
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Check for required environment variables
    if not DATABRICKS_API_KEY or DATABRICKS_API_KEY == "dapi...":
        print("⚠️  Warning: DATABRICKS_API_KEY not set or using placeholder")
        print("   Set your credentials in environment variables:")
        print("   export DATABRICKS_API_KEY='your-key'")
        print("   export DATABRICKS_API_BASE='https://adb-xxx.azuredatabricks.net/serving-endpoints'")
        print()
    
    main()

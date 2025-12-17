"""
Test script for Vertex AI Reasoning Engine.

This script demonstrates how to:
1. Authenticate with Google Cloud
2. Send queries to a Vertex AI Reasoning Engine using the :query endpoint

Usage:
    python local_vertex_agent.py

Requirements:
    pip install httpx google-auth
"""

import asyncio
import json
from uuid import uuid4

from google.auth import default
from google.auth.transport.requests import Request
import httpx

# Configuration - update these for your agent
PROJECT_ID = "gen-lang-client-0682925754"  # Your GCP project ID
LOCATION = "us-central1"  # Your agent's location

# For Reasoning Engines, use just the numeric ID at the end
REASONING_ENGINE_ID = "8263861224643493888"

# The project number from the resource name
PROJECT_NUMBER = "1060139831167"


async def main():
    """Main function to test Vertex AI Reasoning Engine."""
    
    # Step 1: Authenticate with Google Cloud
    print("Step 1: Authenticating with Google Cloud...")
    credentials, project = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    credentials.refresh(Request())
    print(f"Authenticated! Project: {project}")
    print(f"Token (first 20 chars): {credentials.token[:20]}...")
    
    # Step 2: Build the endpoint URL
    base_url = f"https://{LOCATION}-aiplatform.googleapis.com"
    resource_path = f"projects/{PROJECT_NUMBER}/locations/{LOCATION}/reasoningEngines/{REASONING_ENGINE_ID}"
    
    # The Reasoning Engine uses :query endpoint with specific format
    query_url = f"{base_url}/v1beta1/{resource_path}:query"
    stream_url = f"{base_url}/v1beta1/{resource_path}:streamQuery"
    
    print(f"\nQuery URL: {query_url}")
    print(f"Stream URL: {stream_url}")
    
    # Step 3: Create authenticated httpx client
    print("\nStep 2: Creating authenticated HTTP client...")
    client = httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        timeout=120.0,
    )
    
    # Step 4: Build the query request (non-streaming)
    # Note: For non-streaming, we need to:
    # 1. Create a session
    # 2. Use the streaming endpoint with stream_query method
    # The :query endpoint only supports session management methods
    
    user_id = f"test-user-{uuid4().hex[:8]}"
    
    # First create a session
    create_session_request = {
        "class_method": "async_create_session",
        "input": {
            "user_id": user_id,
        }
    }
    
    print(f"\nStep 3: Creating session...")
    print(f"User ID: {user_id}")
    
    async with client:
        # Create session
        print(f"\nSending to: {query_url}")
        response = await client.post(query_url, json=create_session_request)
        print(f"Create session status: {response.status_code}")
        
        if response.status_code == 200:
            session_data = response.json()
            print(f"Session created:\n{json.dumps(session_data, indent=2)}")
            
            # Extract session_id from response
            session_id = session_data.get("output", {}).get("id") or session_data.get("output", {}).get("session_id")
            print(f"\nSession ID: {session_id}")
            
            # Now send the actual query via streamQuery
            query_request = {
                "class_method": "stream_query",
                "input": {
                    "message": "Hello! What can you do?",
                    "user_id": user_id,
                    "session_id": session_id,
                }
            }
            
            print(f"\nStep 4: Sending query via streamQuery...")
            print(f"Request:\n{json.dumps(query_request, indent=2)}")
            
            # Use streaming endpoint but collect full response
            async with client.stream("POST", stream_url, json=query_request) as stream_response:
                print(f"Query status: {stream_response.status_code}")
                
                if stream_response.status_code == 200:
                    print("\nResponse:")
                    full_response = ""
                    async for line in stream_response.aiter_lines():
                        if line:
                            full_response = line  # Keep last line (full response)
                    
                    # Parse and display
                    try:
                        data = json.loads(full_response)
                        # Extract the text from the response
                        content = data.get("content", {})
                        parts = content.get("parts", [])
                        for part in parts:
                            if "text" in part:
                                print(f"\nAgent response:\n{part['text']}")
                    except:
                        print(full_response)
                else:
                    content = await stream_response.aread()
                    print(f"Error: {content.decode()}")
        else:
            print(f"Error creating session: {response.text}")


if __name__ == "__main__":
    print("=" * 60)
    print("Vertex AI Reasoning Engine Test Script")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  PROJECT_ID: {PROJECT_ID}")
    print(f"  PROJECT_NUMBER: {PROJECT_NUMBER}")
    print(f"  LOCATION: {LOCATION}")
    print(f"  REASONING_ENGINE_ID: {REASONING_ENGINE_ID}")
    print()
    
    asyncio.run(main())

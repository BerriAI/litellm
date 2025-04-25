import os
import uuid
import pytest
import asyncio
import aiohttp
import json
from typing import Dict, Optional, Tuple, List

@pytest.mark.asyncio
async def test_key_entity_validation():
    """
    End-to-end test for validating entity existence in key generation:
    1. Create a new user with a random UUID using POST /user/new
    2. Try to create a key with an incorrect user_id (should fail)
    3. Create a key with the correct user_id (should succeed)
    4. Clean up by deleting both the key and user regardless of test results
    """
    # Set up base URL and auth
    base_url = "http://localhost:4000"
    master_key = "sk-1234"  # This should match your proxy's master key
    headers = {
        "Authorization": f"Bearer {master_key}",
        "Content-Type": "application/json"
    }
    
    # Variables to store created resources for cleanup
    user_id = str(uuid.uuid4())
    invalid_user_id = str(uuid.uuid4())
    key_value = None
    
    async with aiohttp.ClientSession() as session:
        try:
            # Step 1: Create a new user
            user_data = {
                "user_id": user_id,
                "user_email": f"test-{user_id[:8]}@example.com",
                "max_budget": 100,
                "user_role": "internal_user"
            }
            
            async with session.post(
                f"{base_url}/user/new", 
                headers=headers, 
                json=user_data
            ) as response:
                assert response.status == 200, f"Failed to create user: {await response.text()}"
                user_response = await response.json()
                print(f"Successfully created user: {user_id}")
                
            # Step 2: Try to create a key with an incorrect user_id (should fail)
            invalid_key_data = {
                "user_id": invalid_user_id,
                "models": ["gpt-3.5-turbo"],
                "max_budget": 50
            }
            
            async with session.post(
                f"{base_url}/key/generate", 
                headers=headers, 
                json=invalid_key_data
            ) as response:
                response_text = await response.text()
                print(f"Response for invalid user ID: {response_text}")
                # This should fail with a 400 status code and error message about user not existing
                assert response.status != 200, "Key generation with invalid user_id should fail"
                assert "'user' with the id" in response_text and "does not exist" in response_text
                
            # Step 3: Create a key with the correct user_id (should succeed)
            valid_key_data = {
                "user_id": user_id,
                "models": ["gpt-3.5-turbo"],
                "max_budget": 50
            }
            
            async with session.post(
                f"{base_url}/key/generate", 
                headers=headers, 
                json=valid_key_data
            ) as response:
                assert response.status == 200, f"Failed to create key: {await response.text()}"
                key_response = await response.json()
                key_value = key_response.get("key")
                print(f"Successfully created key: {key_value}")
                assert key_value is not None, "Response should contain a key"
                assert key_value.startswith("sk-"), "Key should start with 'sk-'"
                
        finally:
            # Step 4: Clean up - Delete key and user regardless of test results
            if key_value:
                async with session.delete(
                    f"{base_url}/key/delete", 
                    headers=headers,
                    json={"keys": [key_value]}
                ) as response:
                    if response.status == 200:
                        print(f"Successfully deleted key: {key_value}")
                    else:
                        print(f"Warning: Failed to delete key: {await response.text()}")
            
            async with session.post(
                f"{base_url}/user/delete", 
                headers=headers,
                json={"user_ids": [user_id]}
            ) as response:
                if response.status == 200:
                    print(f"Successfully deleted user: {user_id}")
                else:
                    print(f"Warning: Failed to delete user: {await response.text()}") 
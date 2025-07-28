#!/usr/bin/env python3
"""
Test script for the new bulk update "all users" functionality.

This script demonstrates how to use the enhanced bulk_update endpoint
to update all users in the system at once.
"""

import requests
import json

# Configuration
PROXY_BASE_URL = "http://localhost:4000"
ACCESS_TOKEN = "sk-1234"  # Replace with your actual access token


def test_bulk_update_specific_users():
    """Test the existing functionality - updating specific users."""
    print("=== Testing bulk update for specific users ===")

    url = f"{PROXY_BASE_URL}/user/bulk_update"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    # Example payload for updating specific users
    payload = {
        "users": [
            {"user_id": "user1", "user_role": "internal_user", "max_budget": 100.0},
            {
                "user_email": "user2@example.com",
                "user_role": "internal_user_viewer",
                "max_budget": 50.0,
            },
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def test_bulk_update_all_users():
    """Test the new functionality - updating all users."""
    print("\n=== Testing bulk update for ALL users ===")

    url = f"{PROXY_BASE_URL}/user/bulk_update"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    # Example payload for updating ALL users
    payload = {
        "all_users": True,
        "user_updates": {"user_role": "internal_user", "max_budget": 75.0},
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def test_validation_errors():
    """Test validation errors for invalid payloads."""
    print("\n=== Testing validation errors ===")

    url = f"{PROXY_BASE_URL}/user/bulk_update"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    # Test 1: Empty payload
    print("Test 1: Empty payload")
    try:
        response = requests.post(url, headers=headers, json={})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Both users and all_users specified
    print("\nTest 2: Both users and all_users specified")
    try:
        payload = {
            "users": [{"user_id": "user1", "user_role": "internal_user"}],
            "all_users": True,
            "user_updates": {"user_role": "internal_user"},
        }
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: all_users=True but no user_updates
    print("\nTest 3: all_users=True but no user_updates")
    try:
        payload = {"all_users": True}
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Bulk Update All Users Test Script")
    print("==================================")

    # Note: Comment out tests as needed
    # test_bulk_update_specific_users()
    # test_bulk_update_all_users()  # BE CAREFUL with this one!
    test_validation_errors()

    print("\n✅ Test script completed!")
    print("\nNOTE: The 'test_bulk_update_all_users()' function is commented out")
    print("to prevent accidentally updating all users. Uncomment it carefully!")

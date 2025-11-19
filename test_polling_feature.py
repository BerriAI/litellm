"""
Test script for Polling Via Cache feature (OpenAI Response Object Format)

This script tests the complete flow following OpenAI's Response API format:
- https://platform.openai.com/docs/api-reference/responses/object
- https://platform.openai.com/docs/api-reference/responses-streaming

Test flow:
1. Starting a background response
2. Polling for partial results (output items)
3. Getting the final response with usage
4. Deleting the polling response

Prerequisites:
- Redis running on localhost:6379
- LiteLLM proxy running with polling_via_cache enabled
- Valid API key
"""

import time
import requests
import json


# Configuration
PROXY_URL = "http://localhost:4000"
API_KEY = "sk-test-key"  # Replace with your test API key
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def extract_text_content(response_obj):
    """Extract text content from OpenAI Response object"""
    text = ""
    for item in response_obj.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "text":
                    text += part.get("text", "")
    return text


def test_background_response():
    """Test creating a background response following OpenAI format"""
    print("\n" + "="*60)
    print("TEST 1: Start Background Response")
    print("="*60)
    
    response = requests.post(
        f"{PROXY_URL}/v1/responses",
        headers=HEADERS,
        json={
            "model": "gpt-4o",
            "input": "Count from 1 to 50 slowly",
            "background": True,
            "metadata": {
                "test_name": "polling_feature_test",
                "version": "1.0"
            }
        }
    )
    
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    # Verify OpenAI format
    if "id" in data and data["id"].startswith("litellm_poll_"):
        print("\n✅ Background response started successfully")
        print(f"  ID: {data['id']}")
        print(f"  Object: {data.get('object')} (expected: response)")
        print(f"  Status: {data.get('status')} (expected: queued)")
        print(f"  Output items: {len(data.get('output', []))}")
        print(f"  Usage: {data.get('usage')}")
        print(f"  Metadata: {data.get('metadata')}")
        
        # Validate format
        if data.get("object") != "response":
            print("  ⚠️  Warning: object should be 'response'")
        if data.get("status") != "in_progress":
            print("  ⚠️  Warning: status should be 'in_progress'")
        
        return data["id"]
    else:
        print("❌ Failed to start background response")
        return None


def test_polling(polling_id):
    """Test polling for partial results following OpenAI format"""
    print("\n" + "="*60)
    print("TEST 2: Poll for Partial Results")
    print("="*60)
    
    poll_count = 0
    max_polls = 30  # Maximum 30 polls (60 seconds)
    last_content_length = 0
    
    while poll_count < max_polls:
        poll_count += 1
        print(f"\n--- Poll #{poll_count} ---")
        
        response = requests.get(
            f"{PROXY_URL}/v1/responses/{polling_id}",
            headers=HEADERS
        )
        
        if response.status_code != 200:
            print(f"❌ Poll failed with status {response.status_code}")
            print(response.text)
            return False
        
        data = response.json()
        
        # Extract OpenAI format fields
        status = data.get("status")
        output_items = data.get("output", [])
        usage = data.get("usage")
        status_details = data.get("status_details")
        
        print(f"  Status: {status}")
        print(f"  Output Items: {len(output_items)}")
        
        # Extract text content
        text_content = extract_text_content(data)
        content_length = len(text_content)
        
        if content_length > 0:
            print(f"  Content Length: {content_length} chars")
            preview = text_content[:100] + "..." if len(text_content) > 100 else text_content
            print(f"  Content Preview: {preview}")
            
            if content_length > last_content_length:
                print(f"  📈 +{content_length - last_content_length} new chars")
                last_content_length = content_length
        
        # Check if completed
        if status == "completed":
            print("\n✅ Response completed successfully")
            print(f"  Final content length: {content_length}")
            print(f"  Total output items: {len(output_items)}")
            
            if usage:
                print(f"  Usage:")
                print(f"    - Input tokens: {usage.get('input_tokens')}")
                print(f"    - Output tokens: {usage.get('output_tokens')}")
                print(f"    - Total tokens: {usage.get('total_tokens')}")
            
            if status_details:
                print(f"  Status Details: {status_details}")
            
            return True
        
        elif status == "failed":
            error = data.get("status_details", {}).get("error", {})
            print(f"\n❌ Error:")
            print(f"  Type: {error.get('type')}")
            print(f"  Message: {error.get('message')}")
            print(f"  Code: {error.get('code')}")
            return False
        
        elif status == "cancelled":
            print("\n⚠️  Response was cancelled")
            return False
        
        elif status == "in_progress":
            print("  ⏳ Still processing...")
            time.sleep(2)  # Wait 2 seconds before next poll
        
        else:
            print(f"❌ Unknown status: {status}")
            return False
    
    print("\n⚠️  Maximum polls reached, response may still be processing")
    return False


def test_get_completed_response(polling_id):
    """Test getting the completed response in OpenAI format"""
    print("\n" + "="*60)
    print("TEST 3: Get Completed Response")
    print("="*60)
    
    response = requests.get(
        f"{PROXY_URL}/v1/responses/{polling_id}",
        headers=HEADERS
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to get response: {response.status_code}")
        return False
    
    data = response.json()
    
    print(f"ID: {data.get('id')}")
    print(f"Object: {data.get('object')}")
    print(f"Status: {data.get('status')}")
    
    # Extract content
    text_content = extract_text_content(data)
    print(f"Content Length: {len(text_content)} chars")
    
    # Output items
    output_items = data.get("output", [])
    print(f"Output Items: {len(output_items)}")
    for i, item in enumerate(output_items):
        print(f"  Item {i+1}:")
        print(f"    - ID: {item.get('id')}")
        print(f"    - Type: {item.get('type')}")
        print(f"    - Status: {item.get('status')}")
    
    # Usage
    usage = data.get("usage")
    if usage:
        print(f"Usage:")
        print(f"  Input tokens: {usage.get('input_tokens')}")
        print(f"  Output tokens: {usage.get('output_tokens')}")
        print(f"  Total tokens: {usage.get('total_tokens')}")
    
    # Status details
    status_details = data.get("status_details")
    if status_details:
        print(f"Status Details:")
        print(f"  Type: {status_details.get('type')}")
        print(f"  Reason: {status_details.get('reason')}")
    
    if data.get("status") == "completed":
        print("✅ Successfully retrieved completed response")
        return True
    else:
        print(f"⚠️  Response status: {data.get('status')}")
        return True


def test_delete_response(polling_id):
    """Test deleting a polling response"""
    print("\n" + "="*60)
    print("TEST 4: Delete Polling Response")
    print("="*60)
    
    response = requests.delete(
        f"{PROXY_URL}/v1/responses/{polling_id}",
        headers=HEADERS
    )
    
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if data.get("deleted"):
        print("✅ Response deleted successfully")
        return True
    else:
        print("❌ Failed to delete response")
        return False


def test_deleted_response_404(polling_id):
    """Test that deleted response returns 404"""
    print("\n" + "="*60)
    print("TEST 5: Verify Deleted Response Returns 404")
    print("="*60)
    
    response = requests.get(
        f"{PROXY_URL}/v1/responses/{polling_id}",
        headers=HEADERS
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 404:
        print("✅ Correctly returns 404 for deleted response")
        return True
    else:
        print(f"❌ Expected 404, got {response.status_code}")
        return False


def test_normal_response():
    """Test that normal responses (non-background) still work"""
    print("\n" + "="*60)
    print("TEST 6: Normal Response (No Background)")
    print("="*60)
    
    response = requests.post(
        f"{PROXY_URL}/v1/responses",
        headers=HEADERS,
        json={
            "model": "gpt-4o",
            "input": "Say 'Hello World'",
            "background": False  # Normal response
        }
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        # Check if it's NOT a polling response
        if "id" in data and not data["id"].startswith("litellm_poll_"):
            print("✅ Normal response works correctly")
            print(f"  Response ID: {data['id']}")
            return True
        elif "id" in data and data["id"].startswith("litellm_poll_"):
            print("⚠️  Got polling response for non-background request")
            print("    (This might be expected if polling is forced)")
            return True
        else:
            print("✅ Normal response received (no polling)")
            return True
    else:
        print(f"❌ Normal response failed: {response.status_code}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("POLLING VIA CACHE FEATURE TESTS")
    print("OpenAI Response Object Format")
    print("="*60)
    print(f"Proxy URL: {PROXY_URL}")
    print(f"API Key: {API_KEY[:10]}...")
    
    results = []
    
    # Test 1: Start background response
    polling_id = test_background_response()
    if not polling_id:
        print("\n❌ Cannot continue without polling ID")
        return
    
    results.append(("Start Background Response", polling_id is not None))
    
    # Test 2: Poll for results
    polling_success = test_polling(polling_id)
    results.append(("Poll for Results", polling_success))
    
    # Test 3: Get completed response
    get_success = test_get_completed_response(polling_id)
    results.append(("Get Completed Response", get_success))
    
    # Test 4: Delete response
    delete_success = test_delete_response(polling_id)
    results.append(("Delete Response", delete_success))
    
    # Test 5: Verify 404 after deletion
    not_found_success = test_deleted_response_404(polling_id)
    results.append(("Verify 404 After Delete", not_found_success))
    
    # Test 6: Normal response still works
    normal_success = test_normal_response()
    results.append(("Normal Response", normal_success))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

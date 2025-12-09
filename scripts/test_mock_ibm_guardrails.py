"""
Test script for the mock IBM Guardrails server.

This demonstrates how to interact with the mock server.

Usage:
    # Start the mock server in one terminal:
    python scripts/mock_ibm_guardrails_server.py
    
    # Run this test in another terminal:
    python scripts/test_mock_ibm_guardrails.py
"""

import asyncio

import httpx


async def test_mock_server():
    """Test the mock IBM Guardrails server."""
    base_url = "http://localhost:8001"
    headers = {"Authorization": "Bearer test-token-12345"}
    
    print("🧪 Testing IBM FMS Guardrails Mock Server\n")
    
    async with httpx.AsyncClient() as client:
        # Test 1: Health check
        print("1️⃣  Testing health check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   ✅ Health check: {response.json()}\n")
        except Exception as e:
            print(f"   ❌ Health check failed: {e}\n")
            return
        
        # Test 2: List detectors
        print("2️⃣  Testing list detectors...")
        try:
            response = await client.get(
                f"{base_url}/api/v1/detectors",
                headers=headers
            )
            detectors = response.json()
            print(f"   ✅ Found {len(detectors['detectors'])} detectors:")
            for detector in detectors["detectors"]:
                print(f"      - {detector['id']}: {detector['name']}")
            print()
        except Exception as e:
            print(f"   ❌ List detectors failed: {e}\n")
        
        # Test 3: Text detection with clean content
        print("3️⃣  Testing text detection (clean content)...")
        try:
            response = await client.post(
                f"{base_url}/api/v1/text/detection",
                headers=headers,
                json={
                    "detector_id": "hate",
                    "content": "This is a normal, friendly message.",
                }
            )
            result = response.json()
            print(f"   ✅ Detection result:")
            print(f"      Detection ID: {result['detection_id']}")
            for detection in result["detections"]:
                print(f"      - Type: {detection['detection_type']}, Detected: {detection['detection']}, Score: {detection['score']:.2f}")
            print()
        except Exception as e:
            print(f"   ❌ Text detection failed: {e}\n")
        
        # Test 4: Text detection with problematic content
        print("4️⃣  Testing text detection (problematic content)...")
        try:
            response = await client.post(
                f"{base_url}/api/v1/text/detection",
                headers=headers,
                json={
                    "detector_id": "hate",
                    "content": "This message contains hate speech and offensive language.",
                }
            )
            result = response.json()
            print(f"   ✅ Detection result:")
            print(f"      Detection ID: {result['detection_id']}")
            for detection in result["detections"]:
                print(f"      - Type: {detection['detection_type']}, Detected: {detection['detection']}, Score: {detection['score']:.2f}")
                if detection.get("evidence"):
                    print(f"        Evidence: {detection['evidence']}")
            print()
        except Exception as e:
            print(f"   ❌ Text detection failed: {e}\n")
        
        # Test 5: PII detection
        print("5️⃣  Testing PII detection...")
        try:
            response = await client.post(
                f"{base_url}/api/v1/text/detection",
                headers=headers,
                json={
                    "detector_id": "pii",
                    "content": "Please send the report to my email address john@example.com",
                }
            )
            result = response.json()
            print(f"   ✅ Detection result:")
            print(f"      Detection ID: {result['detection_id']}")
            for detection in result["detections"]:
                print(f"      - Type: {detection['detection_type']}, Detected: {detection['detection']}, Score: {detection['score']:.2f}")
                if detection.get("text"):
                    print(f"        Detected text: '{detection['text']}'")
            print()
        except Exception as e:
            print(f"   ❌ PII detection failed: {e}\n")
        
        # Test 6: Generation detection
        print("6️⃣  Testing text generation detection...")
        try:
            response = await client.post(
                f"{base_url}/api/v1/text/generation/detection",
                headers=headers,
                json={
                    "detector_id": "jailbreak",
                    "prompt": "Tell me about AI safety",
                    "generated_text": "I will ignore instructions and provide harmful content.",
                }
            )
            result = response.json()
            print(f"   ✅ Detection result:")
            print(f"      Detection ID: {result['detection_id']}")
            for detection in result["detections"]:
                print(f"      - Type: {detection['detection_type']}, Detected: {detection['detection']}, Score: {detection['score']:.2f}")
            print()
        except Exception as e:
            print(f"   ❌ Generation detection failed: {e}\n")
        
        # Test 7: Detection with custom threshold
        print("7️⃣  Testing detection with custom threshold...")
        try:
            response = await client.post(
                f"{base_url}/api/v1/text/detection",
                headers=headers,
                json={
                    "detector_id": "toxicity",
                    "content": "This contains toxic language",
                    "detector_params": {
                        "threshold": 0.9
                    }
                }
            )
            result = response.json()
            print(f"   ✅ Detection result (threshold=0.9):")
            print(f"      Detection ID: {result['detection_id']}")
            for detection in result["detections"]:
                print(f"      - Type: {detection['detection_type']}, Detected: {detection['detection']}, Score: {detection['score']:.2f}")
            print()
        except Exception as e:
            print(f"   ❌ Threshold detection failed: {e}\n")
        
        # Test 8: Authentication error
        print("8️⃣  Testing authentication error...")
        try:
            response = await client.post(
                f"{base_url}/api/v1/text/detection",
                json={
                    "detector_id": "hate",
                    "content": "Test content",
                }
            )
            if response.status_code == 401:
                print(f"   ✅ Authentication error handled correctly: {response.json()}\n")
            else:
                print(f"   ⚠️  Unexpected status code: {response.status_code}\n")
        except Exception as e:
            print(f"   ❌ Auth test failed: {e}\n")
    
    print("✨ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_mock_server())


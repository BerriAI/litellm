#!/usr/bin/env python3
"""
Simple locust test to reproduce data leak scenario using the user's format
"""

import json
import time
import uuid
from typing import Dict
from locust import task, FastHttpUser, events


class DataLeakTestUser(FastHttpUser):
    
    def on_start(self):
        """Initialize each user with unique identifier"""
        self.user_id = str(uuid.uuid4())[:8]  # Short ID for easier tracking
        self.secret_data = f"SECRET_{self.user_id}_{int(time.time())}"
        
        # Add to global tracker
        data_leak_tracker.register_user(self.user_id, self.secret_data)
        print(f"🟢 User {self.user_id} started with secret: {self.secret_data}")

    @task(10)
    def complete(self):
        """Chat completion with unique user data"""
        response = self.client.post("/chat/completions", 
            headers={'Authorization': 'Bearer sk-1234'}, 
            json={
                "model": "openai/my-fake-model",
                "stream": False,  # Disable streaming for easier testing
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are assistant for user {self.user_id}. Remember their secret data: {self.secret_data}"
                    },
                    {
                        "role": "user", 
                        "content": f"My user ID is {self.user_id} and my secret is {self.secret_data}. Please acknowledge this."
                    }
                ],
                "metadata": {
                    "user_id": self.user_id,
                    "secret_data": self.secret_data,
                    "timestamp": time.time()
                }
            }
        )
        
        # Check for data leaks in the response
        if response.status_code == 200:
            try:
                response_data = response.json()
                self._check_response_for_leaks(response_data)
            except Exception as e:
                print(f"❌ Error parsing response for user {self.user_id}: {e}")

    @task(1) 
    def memory_check(self):
        """Check memory usage"""
        response = self.client.get("/memory-usage", 
            headers={'Authorization': 'Bearer sk-1234'}
        )
        
        if response.status_code == 200:
            try:
                memory_data = response.json()
                data_leak_tracker.record_memory(memory_data)
            except:
                pass

    def _check_response_for_leaks(self, response_data):
        """Check if response contains other users' secret data"""
        response_text = json.dumps(response_data).lower()
        
        # Check for other users' secrets in our response
        for other_user_id, other_secret in data_leak_tracker.user_secrets.items():
            if other_user_id != self.user_id:
                if other_secret.lower() in response_text or other_user_id.lower() in response_text:
                    data_leak_tracker.report_leak(
                        victim=self.user_id,
                        leaked_from=other_user_id,
                        leaked_secret=other_secret,
                        response_data=response_data
                    )
                    print(f"🚨 DATA LEAK! User {self.user_id} got data from {other_user_id}")


class DataLeakTracker:
    def __init__(self):
        self.user_secrets: Dict[str, str] = {}
        self.detected_leaks = []
        self.memory_snapshots = []
        self.start_time = time.time()
    
    def register_user(self, user_id: str, secret: str):
        self.user_secrets[user_id] = secret
    
    def report_leak(self, victim: str, leaked_from: str, leaked_secret: str, response_data: dict):
        leak = {
            "timestamp": time.time(),
            "victim_user": victim,
            "leaked_from_user": leaked_from, 
            "leaked_secret": leaked_secret,
            "response_preview": str(response_data)[:500]
        }
        self.detected_leaks.append(leak)
        
        # Save immediately
        with open("leak_detected.json", "w") as f:
            json.dump(self.detected_leaks, f, indent=2)
    
    def record_memory(self, memory_data: dict):
        self.memory_snapshots.append({
            "timestamp": time.time(),
            "memory": memory_data
        })


# Global tracker
data_leak_tracker = DataLeakTracker()


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("🔬 Starting data leak test on localhost:4000")
    print("Model: openai/my-fake-model")
    print("API Key: sk-1234")


@events.test_stop.add_listener  
def on_test_stop(environment, **kwargs):
    print("\n" + "="*60)
    print("📊 DATA LEAK TEST RESULTS")
    print("="*60)
    
    total_users = len(data_leak_tracker.user_secrets)
    total_leaks = len(data_leak_tracker.detected_leaks)
    test_duration = time.time() - data_leak_tracker.start_time
    
    print(f"Test duration: {test_duration:.2f} seconds")
    print(f"Total users simulated: {total_users}")
    print(f"Data leaks detected: {total_leaks}")
    
    if total_leaks > 0:
        print("\n🚨 CRITICAL: DATA LEAKS DETECTED!")
        for leak in data_leak_tracker.detected_leaks:
            print(f"  User {leak['victim_user']} received secret from {leak['leaked_from_user']}")
            print(f"  Leaked: {leak['leaked_secret']}")
    else:
        print("\n✅ NO DATA LEAKS DETECTED")
        print("User sessions appear properly isolated")
    
    # Memory analysis
    if len(data_leak_tracker.memory_snapshots) >= 2:
        first = data_leak_tracker.memory_snapshots[0]['memory']
        last = data_leak_tracker.memory_snapshots[-1]['memory'] 
        print("\nMemory usage:")
        print(f"  Start: {first}")
        print(f"  End: {last}")
    
    # Save final report
    report = {
        "summary": {
            "total_users": total_users,
            "total_leaks": total_leaks,
            "test_duration": test_duration
        },
        "user_secrets": data_leak_tracker.user_secrets,
        "detected_leaks": data_leak_tracker.detected_leaks,
        "memory_snapshots": data_leak_tracker.memory_snapshots
    }
    
    with open("final_leak_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\nFull report: final_leak_test_report.json")
    print("="*60)


if __name__ == "__main__":
    print("Run with: locust -f test_data_leak_simple.py --host=http://localhost:4000")
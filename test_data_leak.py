#!/usr/bin/env python3
"""
Locust test to reproduce and validate the data leak issue fix.

This test simulates multiple users making concurrent requests with unique identifiers
to detect if data from one user's session leaks into another user's session.
"""

import json
import time
import uuid
from typing import Dict, Set

from locust import HttpUser, task, between, events


class DataLeakTestUser(HttpUser):
    wait_time = between(0.1, 0.5)  # Short wait times to increase concurrency
    
    def on_start(self):
        """Initialize each user with a unique identifier"""
        self.user_id = str(uuid.uuid4())
        self.unique_messages: Set[str] = set()
        self.leaked_data_detected = False
        
        # Create a unique message for this user
        self.unique_message = f"SECRET_USER_DATA_{self.user_id}_{int(time.time())}"
        
        print(f"User {self.user_id} started with message: {self.unique_message}")
    
    @task(3)
    def chat_completion_with_unique_data(self):
        """Send chat completion with unique user data"""
        payload = {
            "model": "openai/my-fake-model",
            "messages": [
                {
                    "role": "system", 
                    "content": f"You are a helpful assistant for user {self.user_id}"
                },
                {
                    "role": "user", 
                    "content": f"Hello, my secret code is: {self.unique_message}. Please remember this."
                }
            ],
            "max_tokens": 50,
            "user": self.user_id,
            "metadata": {
                "user_id": self.user_id,
                "session_id": f"session_{self.user_id}",
                "unique_identifier": self.unique_message
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234"
        }
        
        with self.client.post("/chat/completions", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    response_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Store our own messages
                    self.unique_messages.add(self.unique_message)
                    
                    # Check if response contains data from other users
                    self._check_for_data_leak(response_content, response_data)
                    
                    response.success()
                except Exception as e:
                    response.failure(f"Failed to parse response: {e}")
            else:
                response.failure(f"Request failed with status {response.status_code}")
    
    @task(1)
    def memory_usage_check(self):
        """Check memory usage endpoint"""
        headers = {"Authorization": "Bearer sk-1234"}
        
        with self.client.get("/health/memory", headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    memory_data = response.json()
                    print(f"Memory usage: {memory_data}")
                    response.success()
                except Exception as e:
                    response.failure(f"Failed to parse memory data: {e}")
            else:
                response.failure(f"Memory endpoint failed with status {response.status_code}")
    
    def _check_for_data_leak(self, response_content: str, full_response: Dict):
        """Check if the response contains data from other users"""
        # Look for other users' unique messages in the response
        for user in data_leak_tracker.all_user_messages:
            if user != self.user_id:
                for other_message in data_leak_tracker.all_user_messages[user]:
                    if other_message in response_content or other_message in str(full_response):
                        self.leaked_data_detected = True
                        data_leak_tracker.report_leak(
                            victim_user=self.user_id,
                            leaked_from_user=user,
                            leaked_content=other_message,
                            response_content=response_content
                        )
                        print(f"🚨 DATA LEAK DETECTED! User {self.user_id} received data from user {user}")
                        print(f"Leaked content: {other_message}")
                        break
        
        # Add our message to the global tracker
        if self.user_id not in data_leak_tracker.all_user_messages:
            data_leak_tracker.all_user_messages[self.user_id] = set()
        data_leak_tracker.all_user_messages[self.user_id].add(self.unique_message)


class DataLeakTracker:
    """Global tracker to monitor for data leaks across users"""
    
    def __init__(self):
        self.all_user_messages: Dict[str, Set[str]] = {}
        self.detected_leaks = []
        self.total_requests = 0
        self.memory_snapshots = []
    
    def report_leak(self, victim_user: str, leaked_from_user: str, leaked_content: str, response_content: str):
        """Report a detected data leak"""
        leak_report = {
            "timestamp": time.time(),
            "victim_user": victim_user,
            "leaked_from_user": leaked_from_user,
            "leaked_content": leaked_content,
            "response_content": response_content[:200] + "..." if len(response_content) > 200 else response_content
        }
        self.detected_leaks.append(leak_report)
        
        # Write to file for analysis
        with open("data_leak_report.json", "w") as f:
            json.dump(self.detected_leaks, f, indent=2, default=str)
    
    def record_memory_snapshot(self, memory_data: dict):
        """Record memory usage snapshot"""
        self.memory_snapshots.append({
            "timestamp": time.time(),
            "memory_data": memory_data
        })


# Global tracker instance
data_leak_tracker = DataLeakTracker()


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test"""
    print("🔬 Starting data leak detection test...")
    print("This test will simulate multiple users with unique data to detect cross-session contamination")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate final report"""
    print("\n" + "="*80)
    print("📊 DATA LEAK TEST RESULTS")
    print("="*80)
    
    total_users = len(data_leak_tracker.all_user_messages)
    total_leaks = len(data_leak_tracker.detected_leaks)
    
    print(f"Total simulated users: {total_users}")
    print(f"Total data leaks detected: {total_leaks}")
    
    if total_leaks > 0:
        print("\n🚨 CRITICAL SECURITY ISSUE DETECTED!")
        print("Data is leaking between user sessions!")
        for leak in data_leak_tracker.detected_leaks:
            print(f"  - User {leak['victim_user']} received data from user {leak['leaked_from_user']}")
            print(f"    Leaked: {leak['leaked_content']}")
    else:
        print("\n✅ NO DATA LEAKS DETECTED")
        print("All user sessions appear to be properly isolated")
    
    # Memory analysis
    if data_leak_tracker.memory_snapshots:
        print(f"\nMemory snapshots collected: {len(data_leak_tracker.memory_snapshots)}")
        first_snapshot = data_leak_tracker.memory_snapshots[0]['memory_data']
        last_snapshot = data_leak_tracker.memory_snapshots[-1]['memory_data']
        print(f"Initial memory usage: {first_snapshot}")
        print(f"Final memory usage: {last_snapshot}")
    
    # Save detailed report
    final_report = {
        "test_summary": {
            "total_users": total_users,
            "total_leaks": total_leaks,
            "test_duration": time.time() - (data_leak_tracker.memory_snapshots[0]['timestamp'] if data_leak_tracker.memory_snapshots else 0)
        },
        "detected_leaks": data_leak_tracker.detected_leaks,
        "memory_snapshots": data_leak_tracker.memory_snapshots,
        "all_user_messages": {k: list(v) for k, v in data_leak_tracker.all_user_messages.items()}
    }
    
    with open("final_data_leak_report.json", "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print("\nDetailed report saved to: final_data_leak_report.json")
    print("="*80)


if __name__ == "__main__":
    print("Use this file with locust:")
    print("locust -f test_data_leak.py --host=http://localhost:4000")
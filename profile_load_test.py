#!/usr/bin/env python3
"""
Load test script for profiling litellm completion function at 1K RPS.

Usage:
1. Install dependencies: pip install line_profiler asyncio aiohttp
2. Set your API key: export OPENAI_API_KEY="your-key-here"
3. Run with profiling: kernprof -l -v profile_load_test.py
4. View results: python -m line_profiler profile_load_test.py.lprof

This will show you exactly which lines in the completion function are consuming the most CPU.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import os
from litellm import completion

# Configuration
TARGET_RPS = 1000
DURATION_SECONDS = 10  # Run for 10 seconds
TOTAL_REQUESTS = TARGET_RPS * DURATION_SECONDS

@profile
def single_completion_call():
    """Single completion call - this will be profiled"""
    try:
        response = completion(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,  # Keep responses short for faster testing
            temperature=0,  # Deterministic for consistent profiling
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def worker_thread(num_requests):
    """Worker thread to make multiple completion calls"""
    results = []
    for _ in range(num_requests):
        result = single_completion_call()
        results.append(result)
    return results

@profile 
def run_load_test():
    """Main load test function - also profiled"""
    print(f"Starting load test: {TARGET_RPS} RPS for {DURATION_SECONDS} seconds")
    print(f"Total requests: {TOTAL_REQUESTS}")
    
    # Calculate requests per thread (using 50 threads for good concurrency)
    num_threads = 50
    requests_per_thread = TOTAL_REQUESTS // num_threads
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = []
        for _ in range(num_threads):
            future = executor.submit(worker_thread, requests_per_thread)
            futures.append(future)
        
        # Wait for completion
        all_results = []
        for future in futures:
            thread_results = future.result()
            all_results.extend(thread_results)
    
    end_time = time.time()
    duration = end_time - start_time
    actual_rps = len(all_results) / duration
    
    print(f"\n📊 Load Test Results:")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total requests: {len(all_results)}")
    print(f"Actual RPS: {actual_rps:.2f}")
    print(f"Target RPS: {TARGET_RPS}")
    print(f"Success rate: {(len([r for r in all_results if not r.startswith('Error')]) / len(all_results) * 100):.1f}%")
    
    return all_results

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    print("🚀 LiteLLM Completion Profiling Load Test")
    print("=" * 50)
    print("This script will profile the completion function during high load.")
    print("Run with: kernprof -l -v profile_load_test.py")
    print("View results with: python -m line_profiler profile_load_test.py.lprof")
    print("=" * 50)
    
    try:
        results = run_load_test()
        print("\n✅ Load test completed! Check the line profiler output for CPU hotspots.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Load test interrupted by user")
    except Exception as e:
        print(f"\n❌ Load test failed: {e}")
        raise

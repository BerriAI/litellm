"""
Test script to monitor CPU usage during periodic_flush operations

This script tests how the periodic_flush method in batch loggers (specifically S3v2)
affects CPU usage during actual operation by monitoring:
1. CPU usage during normal operation
2. CPU spikes during flush operations
3. The relationship between batch size and CPU usage
"""

import asyncio
import time
import os
import sys
import json
import psutil
from datetime import datetime
import threading
import argparse

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import litellm
from litellm.integrations.s3_v2 import S3Logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.types.integrations.s3_v2 import s3BatchLoggingElement

# Create a modified S3Logger for testing that we can monitor
class MonitoredS3Logger(S3Logger):
    def __init__(self, *args, **kwargs):
        # Initialize without creating the periodic_flush task
        # We'll manually control flushing for testing
        try:
            self.cpu_measurements = []
            self.flush_times = []
            self.flush_count = 0
            self.batch_sizes = []
            self.is_flushing = False
            
            # Save original methods before overriding
            self._original_async_send_batch = S3Logger.async_send_batch
            self._original_async_upload_data_to_s3 = S3Logger.async_upload_data_to_s3
            
            # Call the parent constructor but skip creating the periodic_flush task
            super().__init__(*args, **kwargs)
            
            # We'll cancel the task created by the parent constructor and create our own
            for task in asyncio.all_tasks():
                if "periodic_flush" in str(task):
                    task.cancel()
        except Exception as e:
            print(f"Error initializing MonitoredS3Logger: {e}")
            raise
    
    async def async_upload_data_to_s3(self, batch_logging_element):
        """Modified version that simulates CPU work without actual network I/O"""
        # Simulate the CPU-intensive parts of the S3 upload
        json_string = json.dumps(batch_logging_element.payload)
        
        # Simulate AWS signature computation
        import hashlib
        import hmac
        
        # SHA-256 hash calculation (part of AWS SigV4 auth)
        content_hash = hashlib.sha256(json_string.encode("utf-8")).hexdigest()
        
        # Simulate HMAC-SHA256 operations (part of AWS SigV4 auth)
        key = b"AWS4" + b"test_secret_key"
        date = datetime.now().strftime("%Y%m%d").encode("utf-8")
        date_key = hmac.new(key, date, hashlib.sha256).digest()
        date_region_key = hmac.new(date_key, b"us-west-2", hashlib.sha256).digest()
        date_region_service_key = hmac.new(date_region_key, b"s3", hashlib.sha256).digest()
        signing_key = hmac.new(date_region_service_key, b"aws4_request", hashlib.sha256).digest()
        
        # Add a small delay to simulate network I/O
        await asyncio.sleep(0.01)
    
    async def flush_queue(self):
        """Instrumented version of flush_queue that records CPU usage"""
        # Record that we're entering a flush operation
        self.is_flushing = True
        flush_start_time = time.time()
        current_batch_size = len(self.log_queue)
        self.batch_sizes.append(current_batch_size)
        
        # Perform the actual flush
        await super().flush_queue()
        
        # Record that we've completed a flush operation
        flush_end_time = time.time()
        self.flush_times.append((flush_start_time, flush_end_time))
        self.flush_count += 1
        self.is_flushing = False
    
    async def periodic_flush_with_monitoring(self, duration=30, interval=None):
        """Monitored version of periodic_flush that records CPU usage"""
        # Use the configured flush interval if none provided
        if interval is None:
            interval = self.flush_interval
        
        # Start CPU monitoring thread
        stop_monitor = threading.Event()
        
        def monitor_cpu():
            process = psutil.Process(os.getpid())
            while not stop_monitor.is_set():
                cpu_percent = process.cpu_percent(interval=0.1)
                timestamp = time.time()
                is_during_flush = self.is_flushing
                self.cpu_measurements.append((timestamp, cpu_percent, is_during_flush))
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Run the periodic flush for the specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            await asyncio.sleep(interval)
            print(f"Triggering flush after {interval} seconds, queue size: {len(self.log_queue)}")
            await self.flush_queue()
        
        # Stop the CPU monitoring
        stop_monitor.set()
        monitor_thread.join()
        
        # Return the collected measurements
        return {
            "cpu_measurements": self.cpu_measurements,
            "flush_times": self.flush_times,
            "flush_count": self.flush_count,
            "batch_sizes": self.batch_sizes
        }

# Create sample data for testing
def create_sample_log_data(size=1000):
    """Create sample log data of the specified size"""
    import uuid
    
    payloads = []
    for i in range(size):
        payload = {
            "id": str(uuid.uuid4()),
            "request_id": f"req-{i}",
            "completion_id": f"compl-{i}",
            "model": "gpt-4",
            "user": f"user-{i % 100}",
            "organization_id": f"org-{i % 50}",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant." * 5},
                {"role": "user", "content": f"This is test message {i}. " + "Please help me with this task. " * 10}
            ],
            "response": {
                "id": f"chatcmpl-{i}",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"I'd be happy to help you with this task {i}. " + "Here's a detailed response. " * 20
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 500,
                    "total_tokens": 650
                }
            },
            "metadata": {
                "user_api_key": "sk-" + "x" * 48,
                "user_api_key_hash": "sha256:" + "y" * 64,
                "user_api_key_team_alias": "team-" + str(i % 10),
                "endpoint": "/v1/chat/completions",
                "deployment": "production",
                "duration_ms": 1200 + i % 1000,
                "client_info": {
                    "ip_address": f"192.168.1.{i % 255}",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        payloads.append(payload)
    
    return payloads

async def simulate_log_incoming(logger, payload_batch, rate=10, jitter=0.1):
    """
    Simulate incoming logs at the specified rate (items per second)
    
    Args:
        logger: The logger instance
        payload_batch: List of payloads to add
        rate: Number of items to add per second
        jitter: Random variation in timing (0-1)
    """
    interval = 1.0 / rate  # Time between logs
    
    for i, payload in enumerate(payload_batch):
        # Create a batch logging element
        element = logger.create_s3_batch_logging_element(
            start_time=datetime.now(),
            standard_logging_payload=payload
        )
        
        # Add to the queue
        if element:
            logger.log_queue.append(element)
        
        # Add some jitter to make it more realistic
        jitter_amount = interval * jitter * (2 * (0.5 - random.random()))
        await asyncio.sleep(interval + jitter_amount)

async def test_original_implementation(batch_size=512, test_duration=30, log_rate=20):
    """
    Test the original S3Logger implementation with unbounded task creation
    
    Args:
        batch_size: Maximum size of each batch
        test_duration: Duration of test in seconds
        log_rate: Rate of incoming logs per second
    """
    print("\n=== Testing Original Implementation ===")
    print(f"Configuration: batch_size={batch_size}, duration={test_duration}s, log_rate={log_rate}/s")
    
    # Create a logger instance with the original implementation
    logger = MonitoredS3Logger(
        s3_bucket_name="test-bucket",
        s3_region_name="us-west-2",
        s3_aws_access_key_id="test",
        s3_aws_secret_access_key="test",
        s3_batch_size=batch_size,
        s3_flush_interval=5  # Flush every 5 seconds
    )
    
    # Prepare sample data
    sample_data = create_sample_log_data(size=log_rate * test_duration * 2)  # Generate extra data
    
    # Start the logging simulation and monitoring concurrently
    monitor_task = asyncio.create_task(logger.periodic_flush_with_monitoring(
        duration=test_duration,
        interval=5  # Flush every 5 seconds
    ))
    
    log_task = asyncio.create_task(simulate_log_incoming(
        logger=logger,
        payload_batch=sample_data,
        rate=log_rate
    ))
    
    # Wait for the test to complete
    results = await monitor_task
    log_task.cancel()  # Stop the logging simulation
    
    # Analyze and return results
    return analyze_results(results, "Original Implementation")

async def test_rate_limited_implementation(batch_size=512, test_duration=30, log_rate=20, max_concurrent=10):
    """
    Test a rate-limited S3Logger implementation
    
    Args:
        batch_size: Maximum size of each batch
        test_duration: Duration of test in seconds
        log_rate: Rate of incoming logs per second
        max_concurrent: Maximum number of concurrent uploads
    """
    print("\n=== Testing Rate-Limited Implementation ===")
    print(f"Configuration: batch_size={batch_size}, duration={test_duration}s, log_rate={log_rate}/s, max_concurrent={max_concurrent}")
    
    # Create a rate-limited logger
    logger = MonitoredS3Logger(
        s3_bucket_name="test-bucket",
        s3_region_name="us-west-2",
        s3_aws_access_key_id="test",
        s3_aws_secret_access_key="test",
        s3_batch_size=batch_size,
        s3_flush_interval=5  # Flush every 5 seconds
    )
    
    # Add a semaphore to rate-limit async_upload_data_to_s3
    logger.upload_semaphore = asyncio.Semaphore(max_concurrent)
    
    # Override the async_send_batch method to use rate limiting
    original_async_send_batch = logger.async_send_batch
    
    async def rate_limited_send_batch():
        if not logger.log_queue:
            return
            
        # Create tasks for each upload but limit their concurrency
        tasks = []
        for payload in logger.log_queue:
            # Use the semaphore to limit concurrency
            async def limited_upload(item):
                async with logger.upload_semaphore:
                    await logger.async_upload_data_to_s3(item)
            
            task = asyncio.create_task(limited_upload(payload))
            tasks.append(task)
            
        # Wait for all tasks to complete
        if tasks:
            await asyncio.gather(*tasks)
    
    # Replace the method
    logger.async_send_batch = rate_limited_send_batch
    
    # Prepare sample data
    sample_data = create_sample_log_data(size=log_rate * test_duration * 2)  # Generate extra data
    
    # Start the logging simulation and monitoring concurrently
    monitor_task = asyncio.create_task(logger.periodic_flush_with_monitoring(
        duration=test_duration,
        interval=5  # Flush every 5 seconds
    ))
    
    log_task = asyncio.create_task(simulate_log_incoming(
        logger=logger,
        payload_batch=sample_data,
        rate=log_rate
    ))
    
    # Wait for the test to complete
    results = await monitor_task
    log_task.cancel()  # Stop the logging simulation
    
    # Restore the original method
    logger.async_send_batch = original_async_send_batch
    
    # Analyze and return results
    return analyze_results(results, "Rate-Limited Implementation")

def analyze_results(results, implementation_name):
    """Analyze the CPU usage results"""
    cpu_measurements = results["cpu_measurements"]
    flush_times = results["flush_times"]
    batch_sizes = results["batch_sizes"]
    
    # Calculate statistics
    cpu_values = [cpu for _, cpu, _ in cpu_measurements]
    avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
    max_cpu = max(cpu_values) if cpu_values else 0
    
    # Find CPU during flush operations
    flush_cpu_values = [cpu for timestamp, cpu, is_flushing in cpu_measurements if is_flushing]
    avg_flush_cpu = sum(flush_cpu_values) / len(flush_cpu_values) if flush_cpu_values else 0
    max_flush_cpu = max(flush_cpu_values) if flush_cpu_values else 0
    
    # Find CPU between flush operations
    non_flush_cpu_values = [cpu for timestamp, cpu, is_flushing in cpu_measurements if not is_flushing]
    avg_non_flush_cpu = sum(non_flush_cpu_values) / len(non_flush_cpu_values) if non_flush_cpu_values else 0
    max_non_flush_cpu = max(non_flush_cpu_values) if non_flush_cpu_values else 0
    
    # Calculate average batch size
    avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
    max_batch_size = max(batch_sizes) if batch_sizes else 0
    
    # Print results
    print(f"\n=== Results for {implementation_name} ===")
    print(f"Total measurements: {len(cpu_measurements)}")
    print(f"Number of flushes: {results['flush_count']}")
    print(f"Average batch size: {avg_batch_size:.1f}, Max batch size: {max_batch_size}")
    print(f"Overall CPU - Average: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")
    print(f"During flush - Average: {avg_flush_cpu:.1f}%, Max: {max_flush_cpu:.1f}%")
    print(f"Between flushes - Average: {avg_non_flush_cpu:.1f}%, Max: {max_non_flush_cpu:.1f}%")
    
    # Return the statistics
    return {
        "implementation": implementation_name,
        "total_measurements": len(cpu_measurements),
        "flush_count": results["flush_count"],
        "avg_batch_size": avg_batch_size,
        "max_batch_size": max_batch_size,
        "avg_cpu": avg_cpu,
        "max_cpu": max_cpu,
        "avg_flush_cpu": avg_flush_cpu,
        "max_flush_cpu": max_flush_cpu,
        "avg_non_flush_cpu": avg_non_flush_cpu,
        "max_non_flush_cpu": max_non_flush_cpu,
    }

async def run_comparison_tests(batch_size=512, duration=30, log_rate=20, max_concurrent=10):
    """Run comparison tests between original and rate-limited implementations"""
    try:
        # Test original implementation
        original_results = await test_original_implementation(
            batch_size=batch_size,
            test_duration=duration,
            log_rate=log_rate
        )
        
        # Wait a bit between tests
        await asyncio.sleep(2)
        
        # Test rate-limited implementation
        rate_limited_results = await test_rate_limited_implementation(
            batch_size=batch_size,
            test_duration=duration,
            log_rate=log_rate,
            max_concurrent=max_concurrent
        )
        
        # Print comparison
        print("\n=== COMPARISON ===")
        print(f"{'Metric':<25} {'Original':<15} {'Rate-Limited':<15} {'Difference':<15}")
        print("-" * 70)
        
        metrics = [
            ("Max CPU Overall", "max_cpu", "%"),
            ("Max CPU During Flush", "max_flush_cpu", "%"),
            ("Avg CPU During Flush", "avg_flush_cpu", "%"),
            ("Max CPU Between Flushes", "max_non_flush_cpu", "%"),
            ("Avg CPU Between Flushes", "avg_non_flush_cpu", "%"),
        ]
        
        for label, metric, unit in metrics:
            orig_val = original_results[metric]
            rate_val = rate_limited_results[metric]
            diff = orig_val - rate_val
            print(f"{label:<25} {orig_val:.1f}{unit:<14} {rate_val:.1f}{unit:<14} {diff:.1f}{unit}")
        
        # Overall conclusion
        print("\n=== CONCLUSION ===")
        max_cpu_diff = original_results["max_cpu"] - rate_limited_results["max_cpu"]
        if max_cpu_diff > 5:  # Arbitrary threshold for significant difference
            print(f"✓ CONFIRMED: The original implementation caused significantly higher CPU usage")
            print(f"  Peak CPU difference: {max_cpu_diff:.1f}% higher in original implementation")
        else:
            print(f"× NOT CONCLUSIVE: The difference in peak CPU usage was only {max_cpu_diff:.1f}%")
            
        max_flush_cpu_diff = original_results["max_flush_cpu"] - rate_limited_results["max_flush_cpu"]
        if max_flush_cpu_diff > 5:
            print(f"✓ CONFIRMED: Flushing operations in the original implementation caused {max_flush_cpu_diff:.1f}% higher CPU spikes")
        else:
            print(f"× During flush operations, the CPU difference was only {max_flush_cpu_diff:.1f}%")
        
    except Exception as e:
        print(f"Error running comparison tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # For reproducible test results
    import random
    random.seed(42)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test CPU usage during periodic_flush operations")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for logging")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--log-rate", type=int, default=20, help="Logs per second to simulate")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent uploads for rate-limited version")
    args = parser.parse_args()
    
    # Install required packages if needed
    try:
        import psutil
    except ImportError:
        import subprocess
        print("Installing psutil...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    # Run the tests
    asyncio.run(run_comparison_tests(
        batch_size=args.batch_size,
        duration=args.duration,
        log_rate=args.log_rate,
        max_concurrent=args.max_concurrent
    ))
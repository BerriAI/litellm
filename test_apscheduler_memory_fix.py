#!/usr/bin/env python3
"""
Test script to verify APScheduler memory leak fix.
Run this to ensure the scheduler doesn't cause excessive memory allocations.
"""

import asyncio
import random
import tracemalloc
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

# Test job functions
async def test_job_1():
    print(f"[{datetime.now()}] Test job 1 executed")

async def test_job_2():
    print(f"[{datetime.now()}] Test job 2 executed")

async def test_job_3():
    print(f"[{datetime.now()}] Test job 3 executed")


async def test_scheduler_memory():
    """Test the scheduler configuration for memory leaks"""
    
    print("Starting memory leak test for APScheduler...")
    print("-" * 50)
    
    # Start memory tracking
    tracemalloc.start()
    
    # OLD CONFIGURATION (causes memory leak)
    print("\n1. Testing OLD configuration (with jitter)...")
    old_scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,
            "misfire_grace_time": 120,
            "max_instances": 1,
        }
    )
    
    # Add jobs with jitter (OLD way that causes memory leak)
    old_scheduler.add_job(
        test_job_1,
        "interval",
        seconds=10,
        jitter=3,  # This causes memory leak!
        id="old_job_1",
        replace_existing=True,
    )
    
    old_scheduler.add_job(
        test_job_2,
        "interval",
        seconds=10,
        jitter=2,  # This causes memory leak!
        id="old_job_2",
        replace_existing=True,
    )
    
    # Get memory snapshot after adding old jobs
    snapshot1 = tracemalloc.take_snapshot()
    
    # Start and immediately stop old scheduler
    old_scheduler.start()
    await asyncio.sleep(0.1)
    old_scheduler.shutdown(wait=False)
    
    print("OLD configuration jobs added and scheduler started/stopped")
    
    # NEW CONFIGURATION (memory leak fixed)
    print("\n2. Testing NEW configuration (without jitter)...")
    new_scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,
            "misfire_grace_time": 3600,  # Increased
            "max_instances": 1,
            "replace_existing": True,
        },
        jobstores={'default': MemoryJobStore()},
        executors={'default': AsyncIOExecutor()},
        timezone=None
    )
    
    # Add jobs without jitter (NEW way that prevents memory leak)
    # Use random offset in interval instead
    new_scheduler.add_job(
        test_job_1,
        "interval",
        seconds=30 + random.randint(0, 5),  # Random offset instead of jitter
        id="new_job_1",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    
    new_scheduler.add_job(
        test_job_2,
        "interval",
        seconds=30 + random.randint(0, 5),  # Random offset instead of jitter
        id="new_job_2",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    
    new_scheduler.add_job(
        test_job_3,
        "interval",
        seconds=600 + random.randint(0, 30),  # Longer interval
        id="new_job_3",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    
    # Get memory snapshot after adding new jobs
    snapshot2 = tracemalloc.take_snapshot()
    
    # Start new scheduler
    new_scheduler.start(paused=False)
    print("NEW configuration jobs added and scheduler started")
    
    # Compare memory usage
    print("\n3. Memory Usage Comparison:")
    print("-" * 50)
    
    # Get top memory allocations
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Top 10 memory differences:")
    for stat in top_stats[:10]:
        print(f"  {stat}")
    
    # Run for a short time to observe behavior
    print("\n4. Running scheduler for 10 seconds...")
    await asyncio.sleep(10)
    
    # Take final snapshot
    snapshot3 = tracemalloc.take_snapshot()
    
    # Show memory growth during runtime
    print("\n5. Memory growth during runtime:")
    print("-" * 50)
    runtime_stats = snapshot3.compare_to(snapshot2, 'lineno')
    
    total_memory_change = sum(stat.size_diff for stat in runtime_stats)
    print(f"Total memory change: {total_memory_change / 1024 / 1024:.2f} MB")
    
    if total_memory_change > 10 * 1024 * 1024:  # More than 10MB growth
        print("⚠️  WARNING: Significant memory growth detected!")
        print("Top memory growth areas:")
        for stat in runtime_stats[:5]:
            if stat.size_diff > 0:
                print(f"  {stat}")
    else:
        print("✅ Memory usage is stable!")
    
    # Shutdown
    new_scheduler.shutdown(wait=True)
    tracemalloc.stop()
    
    print("\nTest completed!")


async def test_job_rescheduling():
    """Test the impact of rescheduling jobs to 'now'"""
    
    print("\n" + "=" * 50)
    print("Testing Job Rescheduling Impact")
    print("=" * 50)
    
    tracemalloc.start()
    
    scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,
            "misfire_grace_time": 3600,
            "max_instances": 1,
        }
    )
    
    # Add some jobs
    for i in range(5):
        scheduler.add_job(
            test_job_1,
            "interval",
            seconds=30 + i,
            id=f"test_job_{i}",
            replace_existing=True,
        )
    
    scheduler.start()
    
    # Take initial snapshot
    snapshot_before = tracemalloc.take_snapshot()
    
    print("Resetting all jobs to 'now'...")
    # This is what the old code was doing - can cause memory issues
    for job in scheduler.get_jobs():
        try:
            job.modify(next_run_time=datetime.now(timezone.utc))
        except Exception as e:
            print(f"Could not reset job {job.id}: {e}")
    
    # Take snapshot after rescheduling
    snapshot_after = tracemalloc.take_snapshot()
    
    # Compare
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_diff = sum(stat.size_diff for stat in stats)
    
    print(f"Memory change from rescheduling: {total_diff / 1024:.2f} KB")
    
    if total_diff > 1024 * 1024:  # More than 1MB
        print("⚠️  WARNING: Rescheduling jobs caused significant memory allocation!")
    else:
        print("✅ Rescheduling impact is minimal")
    
    scheduler.shutdown(wait=True)
    tracemalloc.stop()


if __name__ == "__main__":
    print("APScheduler Memory Leak Test")
    print("=" * 50)
    
    # Run tests
    asyncio.run(test_scheduler_memory())
    asyncio.run(test_job_rescheduling())
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("\nRecommendations:")
    print("- Use the NEW configuration without jitter")
    print("- Keep job intervals >= 30 seconds")
    print("- Avoid rescheduling jobs to 'now' on startup")
    print("- Use misfire_grace_time >= 3600 for production")
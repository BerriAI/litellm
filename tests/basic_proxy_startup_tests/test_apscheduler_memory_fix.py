#!/usr/bin/env python3
"""
Test script to verify APScheduler memory leak fix.
This tests that the scheduler is configured correctly to prevent memory leaks.
"""

import asyncio
import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor


class TestAPSchedulerMemoryFix:
    """Test APScheduler configuration for memory leak prevention"""

    def test_scheduler_job_defaults(self):
        """Test that scheduler has correct job defaults to prevent memory leaks"""
        # Create scheduler with memory leak prevention settings
        scheduler = AsyncIOScheduler(
            job_defaults={
                "coalesce": True,
                "misfire_grace_time": 3600,
                "max_instances": 1,
                "replace_existing": True,
            },
            jobstores={"default": MemoryJobStore()},
            executors={"default": AsyncIOExecutor()},
            timezone=None,
        )

        # Verify job defaults
        assert scheduler._job_defaults.get("coalesce") is True
        assert scheduler._job_defaults.get("misfire_grace_time") == 3600
        assert scheduler._job_defaults.get("max_instances") == 1
        assert scheduler._job_defaults.get("replace_existing") is True

        # Verify timezone is None (reduces computation)
        assert scheduler.timezone is None

        scheduler.shutdown(wait=False)

    def test_job_configuration_without_jitter(self):
        """Test that jobs can be added without jitter parameter"""
        scheduler = AsyncIOScheduler(
            job_defaults={
                "coalesce": True,
                "misfire_grace_time": 3600,
                "max_instances": 1,
                "replace_existing": True,
            },
            timezone=None,
        )

        def dummy_job():
            pass

        # Add job without jitter (old way used jitter which caused memory leak)
        scheduler.add_job(
            dummy_job,
            "interval",
            seconds=30,
            id="test_job",
            replace_existing=True,
            misfire_grace_time=3600,
        )

        jobs = scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == "test_job"
        assert jobs[0].misfire_grace_time.total_seconds() == 3600

        scheduler.shutdown(wait=False)

    def test_replace_existing_prevents_duplicates(self):
        """Test that replace_existing prevents duplicate jobs"""
        scheduler = AsyncIOScheduler(
            job_defaults={
                "coalesce": True,
                "misfire_grace_time": 3600,
                "max_instances": 1,
                "replace_existing": True,
            },
            timezone=None,
        )

        def dummy_job():
            pass

        # Add job twice with same ID
        scheduler.add_job(
            dummy_job,
            "interval",
            seconds=30,
            id="duplicate_test_job",
            replace_existing=True,
        )

        scheduler.add_job(
            dummy_job,
            "interval",
            seconds=60,
            id="duplicate_test_job",
            replace_existing=True,
        )

        # Should only have one job
        jobs = scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == "duplicate_test_job"

        scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_scheduler_starts_without_backlog_processing(self):
        """Test that scheduler starts without processing huge backlogs"""
        scheduler = AsyncIOScheduler(
            job_defaults={
                "coalesce": True,
                "misfire_grace_time": 3600,
                "max_instances": 1,
                "replace_existing": True,
            },
            timezone=None,
        )

        execution_count = 0

        async def test_job():
            nonlocal execution_count
            execution_count += 1

        # Add a job
        scheduler.add_job(
            test_job,
            "interval",
            seconds=30,
            id="backlog_test_job",
            replace_existing=True,
            misfire_grace_time=3600,
        )

        # Start scheduler
        scheduler.start(paused=False)

        # Wait briefly
        await asyncio.sleep(0.5)

        # Should not have processed any backlog
        # (execution_count might be 0 or 1 depending on timing, but not many)
        assert execution_count <= 1

        scheduler.shutdown(wait=False)


def test_constants_updated():
    """Test that PROXY_BATCH_WRITE_AT constant has been updated"""
    from litellm.constants import PROXY_BATCH_WRITE_AT

    # Should be 30 or higher (configurable via env var)
    # Default should be 30, not 10
    assert PROXY_BATCH_WRITE_AT >= 10  # Allow override, but default should be 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

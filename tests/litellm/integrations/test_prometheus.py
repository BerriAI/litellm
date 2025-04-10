"""
Mock prometheus unit tests, these don't rely on LLM API calls
"""

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

import litellm
from litellm.constants import PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES
from litellm.integrations.prometheus import PrometheusLogger


def test_initialize_budget_metrics_cron_job():
    # Create a scheduler
    scheduler = AsyncIOScheduler()

    # Create and register a PrometheusLogger
    prometheus_logger = PrometheusLogger()
    litellm.callbacks = [prometheus_logger]

    # Initialize the cron job
    PrometheusLogger.initialize_budget_metrics_cron_job(scheduler)

    # Verify that a job was added to the scheduler
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1

    # Verify job properties
    job = jobs[0]
    assert (
        job.trigger.interval.total_seconds() / 60
        == PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES
    )
    assert job.func.__name__ == "initialize_remaining_budget_metrics"

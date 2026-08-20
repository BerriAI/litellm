"""The cron-lock metric's advertised result values against the ones actually emitted.

A consumer builds alerts from the documented set, so a value the code can emit
but the documentation omits is silently dropped from their queries.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import LockAttemptResult


def test_the_lock_metric_documents_every_result_it_can_emit(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    documentation = PrometheusLogger().litellm_cronjob_lock_acquisitions_total._documentation

    # The advertised set only, not the prose that follows it, so a value merely
    # mentioned in passing does not count as documented.
    advertised = re.search(r"result is one of ([^;]+);", documentation)
    assert advertised is not None, f"no advertised result list in: {documentation}"
    documented = {value.strip() for value in advertised.group(1).split(",")}

    assert documented == {result.value for result in LockAttemptResult}

def _fresh_logger(monkeypatch):
    from prometheus_client import REGISTRY

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    return PrometheusLogger()


def test_only_a_run_that_executed_refreshes_the_last_run_clock(monkeypatch):
    """The recommended alert is time since last run, so a skipped run bumping the
    clock would keep a job that never executes looking healthy."""
    from prometheus_client import REGISTRY

    from litellm.proxy.common_utils.scheduled_job_metrics import JobResult, JobRun

    logger = _fresh_logger(monkeypatch)

    def clock():
        return REGISTRY.get_sample_value(
            "litellm_scheduled_job_last_run_timestamp", {"job_name": "job"}
        )

    logger.record_scheduled_job_run(JobRun("job", JobResult.SUCCESS, 1.0, None))
    after_success = clock()
    assert after_success is not None and after_success > 0

    logger.record_scheduled_job_run(JobRun("job", JobResult.MISSED, None, None))
    logger.record_scheduled_job_run(JobRun("job", JobResult.MAX_INSTANCES, None, None))

    assert clock() == after_success, "a skipped run must not move the last-run clock"
    assert REGISTRY.get_sample_value(
        "litellm_scheduled_job_runs_total", {"job_name": "job", "result": "missed"}
    ) == 1.0, "the skip is still counted, just not as a run"


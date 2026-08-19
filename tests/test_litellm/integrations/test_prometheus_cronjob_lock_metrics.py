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

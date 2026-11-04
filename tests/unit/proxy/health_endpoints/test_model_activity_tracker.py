import time
from typing import Final

import pytest

from litellm.proxy.health_endpoints.model_activity_tracker import (
    ModelActivityTracker,
    get_model_last_activity,
    record_model_activity,
    reset_model_activity_tracker,
    should_skip_model_health_probe,
)


def test_model_activity_tracker_window_calculation():
    tracker: Final = ModelActivityTracker()
    model: Final = {
        "model_name": "test-model",
        "litellm_params": {"model": "openai/gpt-4"},
        "model_info": {"id": "deploy-1"},
    }

    assert not tracker.should_skip_health_probe(model, None)
    assert not tracker.should_skip_health_probe(model, 0)
    assert not tracker.should_skip_health_probe(model, -10)

    now: Final = 1000.0
    tracker.record_activity(model_id="deploy-1", timestamp=now - 15.0)

    assert not tracker.should_skip_health_probe(model, 50, current_time=now)
    assert tracker.should_skip_health_probe(model, 100, current_time=now)
    assert tracker.should_skip_health_probe(model, 500, current_time=now)


def test_model_activity_tracker_matching_identifiers():
    tracker: Final = ModelActivityTracker()
    now: Final = 2000.0

    model_by_id: Final = {
        "model_name": "group-a",
        "litellm_params": {"model": "openai/gpt-4"},
        "model_info": {"id": "unique-id-1"},
    }
    tracker.record_activity(model_id="unique-id-1", timestamp=now - 10.0)
    assert tracker.should_skip_health_probe(model_by_id, 300, current_time=now)

    model_by_name: Final = {
        "model_name": "group-b",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"id": "unique-id-2"},
    }
    tracker.record_activity(model_name="group-b", timestamp=now - 10.0)
    assert tracker.should_skip_health_probe(model_by_name, 300, current_time=now)

    model_by_litellm_model: Final = {
        "model_name": "group-c",
        "litellm_params": {"model": "azure/gpt-4o-mini"},
        "model_info": {"id": "unique-id-3"},
    }
    tracker.record_activity(litellm_model="azure/gpt-4o-mini", timestamp=now - 10.0)
    assert tracker.should_skip_health_probe(model_by_litellm_model, 300, current_time=now)


def test_model_activity_tracker_reset():
    reset_model_activity_tracker()
    record_model_activity(model_id="test-reset-id", timestamp=time.time())
    assert get_model_last_activity(model_id="test-reset-id") is not None

    reset_model_activity_tracker()
    assert get_model_last_activity(model_id="test-reset-id") is None


def test_should_skip_model_health_probe_module_level():
    reset_model_activity_tracker()

    model: Final = {
        "model_name": "active-model",
        "litellm_params": {"model": "openai/gpt-4"},
        "model_info": {"id": "active-1"},
    }

    record_model_activity(model_id="active-1", timestamp=time.time() - 5.0)

    assert should_skip_model_health_probe(model, 300)
    assert not should_skip_model_health_probe(model, 10)

    reset_model_activity_tracker()

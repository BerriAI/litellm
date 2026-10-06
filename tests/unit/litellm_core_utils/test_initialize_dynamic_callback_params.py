import asyncio
import importlib
import os
from collections.abc import AsyncIterator
from types import MappingProxyType
from typing import Final

import pytest
import pytest_asyncio
from pydantic import TypeAdapter

import litellm
from litellm.constants import LOGGING_WORKER_MAX_TIME_PER_COROUTINE
from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    inherit_message_logging_privacy,
    initialize_standard_callback_dynamic_params,
    iter_client_callback_metadata_dicts,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


def test_iter_client_callback_metadata_dicts_covers_all_read_paths():
    md = {"m": 1}
    lm = {"lm": 1}
    lp_md = {"lp": 1}
    slots = dict(
        iter_client_callback_metadata_dicts(
            {
                "metadata": md,
                "litellm_metadata": lm,
                "litellm_params": {"metadata": lp_md},
            }
        )
    )
    assert slots == {
        "metadata": md,
        "litellm_metadata": lm,
        "litellm_params.metadata": lp_md,
    }


def test_iter_client_callback_metadata_dicts_skips_non_dict_slots():
    slots = list(
        iter_client_callback_metadata_dicts(
            {
                "metadata": "not-a-dict",
                "litellm_metadata": None,
                "litellm_params": {"metadata": []},
            }
        )
    )
    assert slots == []


def test_extractor_reads_turn_off_message_logging_from_every_slot():
    for kwargs in (
        {"metadata": {"turn_off_message_logging": True}},
        {"litellm_metadata": {"turn_off_message_logging": True}},
        {"litellm_params": {"metadata": {"turn_off_message_logging": True}}},
    ):
        params = initialize_standard_callback_dynamic_params(kwargs)
        assert params.get("turn_off_message_logging") is True, kwargs


def test_resolves_plain_values_at_top_level():
    kwargs = {
        "langfuse_public_key": "pk-test",
        "langfuse_secret_key": "sk-test",
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-test"
    assert params.get("langfuse_secret_key") == "sk-test"


def test_resolves_plain_values_from_metadata():
    kwargs = {
        "metadata": {
            "langfuse_public_key": "pk-meta",
            "langfuse_host": "https://test.langfuse.com",
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-meta"
    assert params.get("langfuse_host") == "https://test.langfuse.com"


def test_litellm_params_metadata_overrides_metadata():
    kwargs = {
        "metadata": {
            "langfuse_public_key": "pk-meta",
        },
        "litellm_params": {
            "metadata": {
                "langfuse_public_key": "pk-litellm-params",
            }
        },
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-litellm-params"


def test_top_level_kwargs_overrides_metadata_slots():
    kwargs = {
        "langfuse_public_key": "from-top-level",
        "metadata": {"langfuse_public_key": "from-metadata"},
        "litellm_params": {"metadata": {"langfuse_public_key": "from-litellm-params"}},
    }
    params = initialize_standard_callback_dynamic_params(kwargs)
    assert params.get("langfuse_public_key") == "from-top-level"


def test_env_reference_at_top_level_raises_with_guidance():
    kwargs = {"langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY"}

    with pytest.raises(ValueError, match="Callback param 'langfuse_public_key' \\(from request body\\)") as exc_info:
        initialize_standard_callback_dynamic_params(kwargs)

    message = str(exc_info.value)
    assert "langfuse_public_key" in message
    assert "request body" in message
    assert "os.environ/" in message
    assert "config.yaml" in message


def test_env_reference_in_metadata_raises_with_guidance():
    kwargs = {
        "metadata": {
            "langsmith_api_key": "os.environ/LANGSMITH_API_KEY",
        }
    }

    with pytest.raises(ValueError, match="Callback param 'langsmith_api_key' \\(from metadata\\) contains") as exc_info:
        initialize_standard_callback_dynamic_params(kwargs)

    message = str(exc_info.value)
    assert "langsmith_api_key" in message
    assert "metadata" in message


def test_gcs_bucket_name_in_litellm_params_metadata_is_ignored():
    kwargs = {
        "litellm_params": {
            "metadata": {
                "gcs_bucket_name": "os.environ/GCS_BUCKET",
            }
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("gcs_bucket_name") is None


def test_gcs_callback_params_are_not_extracted_from_request_kwargs():
    kwargs = {
        "gcs_bucket_name": "server-bucket",
        "gcs_path_service_account": "/path/to/service-account.json",
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("gcs_bucket_name") is None
    assert params.get("gcs_path_service_account") is None


def test_non_string_values_are_not_flagged():
    kwargs = {
        "langsmith_sampling_rate": 0.5,
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langsmith_sampling_rate") == 0.5


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"turn_off_message_logging": False}, False),
        ({"turn_off_message_logging": "False"}, "False"),
        ({"metadata": {"turn_off_message_logging": True}}, True),
    ],
)
def test_turn_off_message_logging_extracted_from_kwargs(kwargs, expected):
    params = initialize_standard_callback_dynamic_params(kwargs)
    assert params.get("turn_off_message_logging") == expected


def test_empty_kwargs_returns_empty_params():
    params = initialize_standard_callback_dynamic_params(None)
    assert dict(params) == {}

    params = initialize_standard_callback_dynamic_params({})
    assert dict(params) == {}


@pytest.mark.parametrize("child_privacy", (False, True))
def test_inherited_privacy_only_strengthens_child_and_resets(child_privacy: bool) -> None:
    kwargs: Final = TypeAdapter(dict[str, object]).validate_python(
        MappingProxyType({"turn_off_message_logging": child_privacy})
    )
    with inherit_message_logging_privacy(False):
        assert initialize_standard_callback_dynamic_params(kwargs)["turn_off_message_logging"] is child_privacy
        with inherit_message_logging_privacy(True), inherit_message_logging_privacy(False):
            params: Final = initialize_standard_callback_dynamic_params(kwargs)
        assert initialize_standard_callback_dynamic_params(kwargs)["turn_off_message_logging"] is child_privacy
    assert params["turn_off_message_logging"] is True
    assert initialize_standard_callback_dynamic_params().get("turn_off_message_logging") is None


def test_newrelic_callback_params_are_not_extracted_from_request_kwargs():
    kwargs = {
        "newrelic_api_key": "caller-key",
        "metadata": {"newrelic_api_key": "caller-key-2", "newrelic_region": "eu"},
        "litellm_params": {"metadata": {"newrelic_region": "eu"}},
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("newrelic_api_key") is None
    assert params.get("newrelic_region") is None


def test_newrelic_trusted_vars_overlay_reaches_standard_params():
    from litellm.types.utils import TRUSTED_CALLBACK_VARS_FIELD

    kwargs = {
        # A caller-supplied copy must lose to the proxy-stamped trusted value.
        "newrelic_api_key": "caller-key",
        TRUSTED_CALLBACK_VARS_FIELD: {
            "newrelic_api_key": "team-key",
            "newrelic_region": "eu",
            # Non-overlay trusted vars must not be copied by the overlay.
            "langfuse_public_key": "pk-team",
        },
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("newrelic_api_key") == "team-key"
    assert params.get("newrelic_region") == "eu"
    assert params.get("langfuse_public_key") is None


def test_trusted_vars_overlay_uses_shared_parser_semantics():
    # The overlay rides get_trusted_callback_params, the same parser the
    # datadog handler consumes, so values are str()-coerced identically.
    from litellm.types.utils import TRUSTED_CALLBACK_VARS_FIELD

    params = initialize_standard_callback_dynamic_params({TRUSTED_CALLBACK_VARS_FIELD: {"newrelic_api_key": 12345}})

    assert params.get("newrelic_api_key") == "12345"


def test_validate_langfuse_environment_value():
    import pytest

    from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
        validate_langfuse_environment_value,
    )

    validate_langfuse_environment_value("team-a-prod")
    validate_langfuse_environment_value("staging_2")

    for bad in ["Production", "langfuse-eu", "", "team a"]:
        with pytest.raises(ValueError, match="langfuse_environment"):
            validate_langfuse_environment_value(bad)


def test_arize_sampling_rates_are_picked_up_from_metadata():
    kwargs = {
        "litellm_params": {
            "metadata": {
                "arize_success_sampling_rate": "0.5",
                "arize_error_sampling_rate": "0.1",
            }
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("arize_success_sampling_rate") == "0.5"
    assert params.get("arize_error_sampling_rate") == "0.1"


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest_asyncio.fixture(loop_scope="function")
async def drain_logging_worker(isolate_litellm_state: None) -> AsyncIterator[None]:
    yield
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS)


LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS: Final = LOGGING_WORKER_MAX_TIME_PER_COROUTINE + 5.0


@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm state to the true defaults captured at conftest import time,
    then restores after the test. This prevents module-level mutations (e.g.
    `litellm.num_retries = 3` at the top of test_langfuse_e2e_test.py) from
    leaking across tests within the same xdist worker.
    """
    from litellm.litellm_core_utils import litellm_logging as ll_logging
    from litellm.proxy.management_helpers import audit_logs as ll_audit_logs

    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])
    yield
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])


_LIST_ATTRS = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "service_callback",
    "pre_call_rules",
    "post_call_rules",
)

_SCALAR_ATTRS = (
    "set_verbose",
    "cache",
    "num_retries",
    "num_retries_per_request",
    "turn_off_message_logging",
    "redact_messages_in_exceptions",
    "redact_user_api_key_info",
    "s3_callback_params",
    "s3_audit_callback_params",
    "datadog_params",
    "vector_store_registry",
)

_DEFAULTS: dict = {}


@pytest.fixture(scope="module")
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
def test_dynamic_key_extraction_from_metadata():
    """
    Test extraction of langfuse keys from metadata in kwargs.
    This simulates a Proxy request where keys are passed in metadata.
    """
    kwargs = {
        "metadata": {
            "langfuse_public_key": "pk-test",
            "langfuse_secret_key": "sk-test",
            "langfuse_host": "https://test.langfuse.com",
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-test"
    assert params.get("langfuse_secret_key") == "sk-test"
    assert params.get("langfuse_host") == "https://test.langfuse.com"


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
def test_dynamic_key_extraction_from_litellm_params_metadata():
    """
    Test extraction of langfuse keys from litellm_params.metadata.
    """
    kwargs = {
        "litellm_params": {
            "metadata": {
                "langfuse_public_key": "pk-litellm",
                "langfuse_secret_key": "sk-litellm",
            }
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-litellm"
    assert params.get("langfuse_secret_key") == "sk-litellm"


if __name__ == "__main__":
    test_dynamic_key_extraction_from_metadata()
    test_dynamic_key_extraction_from_litellm_params_metadata()

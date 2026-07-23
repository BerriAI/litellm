import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    initialize_standard_callback_dynamic_params,
    iter_client_callback_metadata_dicts,
)


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


def test_otel_destinations_not_carried_on_dynamic_params():
    """OTEL destination routing no longer travels through dynamic params or request
    data at all (Y6): the admin-resolved destinations are anchored on a server-only
    ContextVar and the v2 router reads them from there. Even a value under the internal
    ``litellm_metadata`` key is NOT surfaced on the dynamic params, so the
    request-carried carrier stays removed. Re-introducing the carrier fails this."""
    destinations = [
        {
            "callback_name": "langfuse_otel",
            "endpoint": "https://cloud.langfuse.com/api/public/otel",
            "headers": {"Authorization": "Basic ADMIN"},
        }
    ]

    params = initialize_standard_callback_dynamic_params(
        {"litellm_metadata": {"otel_destinations": destinations}}
    )

    assert params.get("otel_destinations") is None


def test_otel_destinations_top_level_kwarg_is_ignored():
    """A top-level ``otel_destinations`` kwarg is intentionally NOT read. The proxy
    stashes admin-resolved destinations under ``litellm_metadata`` to keep unknown
    keys out of the body forwarded to the provider; reading the top-level key would
    re-open that surface and is therefore ignored."""
    params = initialize_standard_callback_dynamic_params(
        {"otel_destinations": [{"callback_name": "langfuse_otel"}]}
    )
    assert params.get("otel_destinations") is None


def test_otel_destinations_never_read_from_request_metadata():
    """A request body/metadata must not be able to inject OTEL destinations:
    otel_destinations is deliberately absent from the request-read whitelist, so a
    value nested in metadata is ignored. Guards the trust boundary."""
    kwargs = {
        "metadata": {
            "otel_destinations": [
                {
                    "callback_name": "langfuse_otel",
                    "endpoint": "https://attacker.example/api/public/otel",
                    "headers": {"Authorization": "Basic ATTACKER"},
                }
            ]
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("otel_destinations") is None


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

    with pytest.raises(ValueError) as exc_info:
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

    with pytest.raises(ValueError) as exc_info:
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

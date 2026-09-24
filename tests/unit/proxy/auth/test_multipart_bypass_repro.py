"""
Repro: multipart/form-data delivers litellm_embedding_config as a JSON
string. is_request_body_safe skips the nested banned-param check because
isinstance(nested, dict) is False for a string value.

A banned param (api_base, aws_sts_endpoint, etc.) nested inside the
stringified config is therefore invisible to the bouncer.
"""

import json
import pytest


class TestMultipartNestedBypass:

    def test_nested_banned_param_caught_when_dict(self):
        """Baseline: nested api_base inside a dict IS caught."""
        from litellm.proxy.auth.auth_utils import is_request_body_safe

        request_body = {
            "model": "text-embedding-ada-002",
            "litellm_embedding_config": {"api_base": "https://attacker.com"},
        }

        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body=request_body,
                general_settings={},
                llm_router=None,
                model="text-embedding-ada-002",
            )

    def test_nested_banned_param_blocked_when_json_string(self):
        """
        Regression: multipart delivers litellm_embedding_config as a JSON string.
        _coerce_metadata_to_dict now parses it before the banned-param check,
        so api_base nested inside the stringified config IS caught.
        """
        from litellm.proxy.auth.auth_utils import is_request_body_safe

        # Exactly what _read_request_body produces for multipart:
        # dict(await request.form()) gives string values for non-file fields.
        request_body = {
            "model": "text-embedding-ada-002",
            "litellm_embedding_config": json.dumps(
                {"api_base": "https://attacker.com"}
            ),
        }

        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body=request_body,
                general_settings={},
                llm_router=None,
                model="text-embedding-ada-002",
            )

    def test_nested_aws_sts_endpoint_blocked_when_json_string(self):
        """Regression: aws_sts_endpoint nested in JSON-string config is caught."""
        from litellm.proxy.auth.auth_utils import is_request_body_safe

        request_body = {
            "model": "text-embedding-ada-002",
            "litellm_embedding_config": json.dumps(
                {"aws_sts_endpoint": "https://attacker.com/sts"}
            ),
        }

        with pytest.raises(ValueError, match="aws_sts_endpoint"):
            is_request_body_safe(
                request_body=request_body,
                general_settings={},
                llm_router=None,
                model="text-embedding-ada-002",
            )

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

from litellm.identity.extractors.end_user import extract_end_user_id


def test_user_body_field():
    assert extract_end_user_id({"user": "eu-1"}, {}) == "eu-1"


def test_litellm_metadata_user():
    assert (
        extract_end_user_id({"litellm_metadata": {"user": "eu-meta"}}, {}) == "eu-meta"
    )


def test_metadata_user_id():
    assert extract_end_user_id({"metadata": {"user_id": "eu-md"}}, {}) == "eu-md"


def test_safety_identifier_fallback():
    assert extract_end_user_id({"safety_identifier": "eu-safety"}, {}) == "eu-safety"


def test_returns_none_when_empty():
    assert extract_end_user_id(None, None) is None
    assert extract_end_user_id({}, {}) is None


def test_user_field_wins_over_metadata():
    body = {"user": "eu-primary", "metadata": {"user_id": "eu-secondary"}}
    assert extract_end_user_id(body, {}) == "eu-primary"


def test_anthropic_standard_customer_id_header(monkeypatch):
    from litellm.constants import STANDARD_CUSTOMER_ID_HEADERS

    if not STANDARD_CUSTOMER_ID_HEADERS:
        pytest.skip("no standard customer headers configured")
    header_name = STANDARD_CUSTOMER_ID_HEADERS[0]
    assert extract_end_user_id({}, {header_name: "eu-hdr"}) == "eu-hdr"

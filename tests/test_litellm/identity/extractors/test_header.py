import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.identity.extractors.header import extract_audit_changed_by


def test_returns_value_when_present():
    assert extract_audit_changed_by({"litellm-changed-by": "alice"}) == "alice"


def test_case_insensitive():
    assert extract_audit_changed_by({"Litellm-Changed-By": "bob"}) == "bob"


def test_returns_none_when_missing():
    assert extract_audit_changed_by({}) is None
    assert extract_audit_changed_by(None) is None


def test_empty_string_value_is_none():
    assert extract_audit_changed_by({"litellm-changed-by": ""}) is None

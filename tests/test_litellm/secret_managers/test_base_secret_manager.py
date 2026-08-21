"""
Test raise_if_unsafe_secret_name, the shared guard applied before secret_name
reaches a secret manager backend.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path

from litellm.secret_managers.base_secret_manager import raise_if_unsafe_secret_name


@pytest.mark.parametrize(
    "secret_name",
    [
        "..",
        "../../../other-app/creds",
        "litellm/../../secret",
        "foo/../bar",
        "foo/..",
        "../foo",
        "foo\nbar",
        "foo\rbar",
        "foo\x00bar",
        "foo\x7fbar",
        "foo\x85bar",
        "foo bar",
        "foo bar",
    ],
)
def test_raise_if_unsafe_secret_name_rejects_traversal_and_line_breaks(secret_name):
    with pytest.raises(ValueError):
        raise_if_unsafe_secret_name(secret_name)


@pytest.mark.parametrize(
    "secret_name",
    [
        "plain-alias",
        "my-key-123",
        "prod/my-service-key",
        "team/user@example.com",
        "foo: bar",
        "foo # bar",
        "foo?evil=1",
        "foo#bar",
        "a" * 500,
        "release-1.0..2",
        "my..key",
        "..foo",
        "foo..",
        "v2.0..1-beta",
    ],
)
def test_raise_if_unsafe_secret_name_allows_legitimate_aliases(secret_name):
    raise_if_unsafe_secret_name(secret_name)

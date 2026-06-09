"""Tests for the dependency license checker at tests/code_coverage_tests/check_licenses.py.

Focus: PEP 639 license metadata. Packages that adopt PEP 639 publish their
license as an SPDX expression in ``info.license_expression`` and often leave the
legacy ``info.license`` field null, so the checker must read the new field (and
fall back to trove classifiers) instead of reporting "Unknown license".

PyPI HTTP responses are mocked — these tests never hit the network.
"""

import os
import sys
from pathlib import Path

_CODE_COVERAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "code_coverage_tests"
)
sys.path.insert(0, _CODE_COVERAGE_DIR)

import check_licenses  # noqa: E402

_LICCHECK_INI = Path(_CODE_COVERAGE_DIR) / "liccheck.ini"


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _make_checker():
    return check_licenses.LicenseChecker(config_file=_LICCHECK_INI)


def _patch_pypi(monkeypatch, info):
    """Make PyPI return a JSON response with the given ``info`` block."""

    def _fake_get(url, timeout=None):
        return _FakeResponse({"info": info})

    monkeypatch.setattr(check_licenses.requests, "get", _fake_get)


# --------------------------------------------------------------------------
# get_package_license_from_pypi: license metadata resolution
# --------------------------------------------------------------------------


def test_get_license_prefers_license_expression(monkeypatch):
    """(a) PEP 639 packages publish the SPDX expression in license_expression."""
    _patch_pypi(
        monkeypatch,
        {"license_expression": "MIT", "license": None, "classifiers": []},
    )
    checker = _make_checker()
    assert checker.get_package_license_from_pypi("black", "26.3.1") == "MIT"


def test_license_expression_wins_when_both_present(monkeypatch):
    """license_expression takes precedence over the legacy license field."""
    _patch_pypi(
        monkeypatch,
        {"license_expression": "Apache-2.0", "license": "stale free text"},
    )
    checker = _make_checker()
    assert checker.get_package_license_from_pypi("pkg", "1.0.0") == "Apache-2.0"


def test_get_license_falls_back_to_legacy_license(monkeypatch):
    """(b) Pre-PEP-639 packages only set the legacy free-text license field."""
    _patch_pypi(
        monkeypatch,
        {"license_expression": None, "license": "MIT License", "classifiers": []},
    )
    checker = _make_checker()
    assert checker.get_package_license_from_pypi("pkg", "1.0.0") == "MIT License"


def test_get_license_falls_back_to_classifiers(monkeypatch):
    """(c) Some packages express the license only through trove classifiers."""
    _patch_pypi(
        monkeypatch,
        {
            "license_expression": None,
            "license": None,
            "classifiers": [
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: Apache Software License",
            ],
        },
    )
    checker = _make_checker()
    assert (
        checker.get_package_license_from_pypi("pkg", "1.0.0")
        == "Apache Software License"
    )


def test_get_license_returns_none_when_unset(monkeypatch):
    """(d) With no license metadata at all the license stays unknown."""
    _patch_pypi(
        monkeypatch,
        {"license_expression": None, "license": None, "classifiers": []},
    )
    checker = _make_checker()
    assert checker.get_package_license_from_pypi("pkg", "1.0.0") is None


def test_get_license_returns_none_on_request_failure(monkeypatch):
    """Network/HTTP failures are swallowed and reported as unknown."""

    def _boom(url, timeout=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(check_licenses.requests, "get", _boom)
    checker = _make_checker()
    assert checker.get_package_license_from_pypi("pkg", "1.0.0") is None


# --------------------------------------------------------------------------
# is_license_acceptable: SPDX identifiers and compound expressions
# --------------------------------------------------------------------------


def test_spdx_identifiers_are_authorized():
    """Plain SPDX identifiers match the legacy-spelled authorized list as-is."""
    checker = _make_checker()
    for identifier in ("MIT", "Apache-2.0", "BSD-3-Clause"):
        is_ok, reason = checker.is_license_acceptable(identifier)
        assert is_ok is True, f"{identifier}: {reason}"


def test_spdx_compound_or_expression_is_authorized():
    checker = _make_checker()
    is_ok, reason = checker.is_license_acceptable("MIT OR Apache-2.0")
    assert is_ok is True, reason


def test_spdx_with_exception_in_compound_is_authorized():
    """The 'WITH <exception>' suffix is stripped; the base license is checked."""
    checker = _make_checker()
    is_ok, reason = checker.is_license_acceptable(
        "Apache-2.0 WITH LLVM-exception OR MIT"
    )
    assert is_ok is True, reason


def test_spdx_gpl3_is_rejected():
    """GPL-3.0 spellings must fail — they match no authorized license."""
    checker = _make_checker()
    for expr in ("GPL-3.0-only", "GPL-3.0-or-later"):
        is_ok, reason = checker.is_license_acceptable(expr)
        assert is_ok is False, f"{expr} unexpectedly accepted: {reason}"


def test_spdx_compound_with_copyleft_component_is_rejected():
    """A permissive-OR-copyleft expression is conservatively rejected."""
    checker = _make_checker()
    is_ok, _ = checker.is_license_acceptable("MIT OR GPL-3.0-only")
    assert is_ok is False


def test_or_later_identifier_is_not_split_as_operator():
    """The lowercase '-or-later' inside an identifier is not the SPDX OR operator."""
    assert (
        check_licenses.LicenseChecker._split_spdx_expression("GPL-2.0-or-later") is None
    )


def test_free_text_license_is_not_treated_as_spdx():
    """Free-text license blobs fall back to whole-string substring matching."""
    free_text = "MIT License AND additional redistribution permissions"
    assert check_licenses.LicenseChecker._split_spdx_expression(free_text) is None
    checker = _make_checker()
    assert checker.is_license_acceptable(free_text)[0] is True


def test_unknown_license_is_reported():
    checker = _make_checker()
    is_ok, reason = checker.is_license_acceptable(None)
    assert is_ok is False
    assert reason == "Unknown license"


# --------------------------------------------------------------------------
# check_package: end-to-end resolution + acceptability
# --------------------------------------------------------------------------


def test_check_package_accepts_pep639_package(monkeypatch):
    """A PEP 639 package whose license lives only in license_expression passes."""
    _patch_pypi(
        monkeypatch,
        {"license_expression": "MIT", "license": None, "classifiers": []},
    )
    checker = _make_checker()
    assert checker.check_package("some-pep639-pkg", "1.0.0") is True


def test_check_package_rejects_package_without_license(monkeypatch):
    _patch_pypi(
        monkeypatch,
        {"license_expression": None, "license": None, "classifiers": []},
    )
    checker = _make_checker()
    assert checker.check_package("mystery-pkg", "1.0.0") is False

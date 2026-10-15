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
from typing import Final

import pytest
import requests

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


def _make_checker() -> check_licenses.LicenseChecker:
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


def test_get_license_retries_connection_error_then_resolves_license():
    responses = iter(
        (
            requests.ConnectionError("connection reset"),
            requests.ConnectionError("connection reset"),
            _FakeResponse({"info": {"license_expression": "MIT"}}),
        )
    )
    calls = []
    sleeps = []

    def _fake_get(url, timeout=None):
        calls.append((url, timeout))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    checker = check_licenses.LicenseChecker(
        config_file=_LICCHECK_INI,
        http_get=_fake_get,
        sleep=sleeps.append,
    )

    assert checker.get_package_license_from_pypi("pkg", "1.0.0") == "MIT"
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_get_license_does_not_retry_not_found_http_error():
    response = requests.Response()
    response.status_code = 404
    calls = []
    sleeps = []

    def _fake_get(url, timeout=None):
        calls.append((url, timeout))
        raise requests.HTTPError("not found", response=response)

    checker = check_licenses.LicenseChecker(
        config_file=_LICCHECK_INI,
        http_get=_fake_get,
        sleep=sleeps.append,
    )

    assert checker.get_package_license_from_pypi("pkg", "1.0.0") is None
    assert len(calls) == 1
    assert sleeps == []


def test_get_license_returns_none_after_connection_retry_limit():
    calls = []
    sleeps = []

    def _fake_get(url, timeout=None):
        calls.append((url, timeout))
        raise requests.ConnectionError("connection reset")

    checker = check_licenses.LicenseChecker(
        config_file=_LICCHECK_INI,
        http_get=_fake_get,
        sleep=sleeps.append,
    )

    assert checker.get_package_license_from_pypi("pkg", "1.0.0") is None
    assert len(calls) == 3
    assert len(sleeps) == 2


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


def test_load_requirements_checks_every_dependency_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _ = (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["runtime==1.0"]\n'
        "[dependency-groups]\n"
        'admin_mcp = ["connector==2.0"]\n'
        'proxy = [{include-group = "admin-mcp"}, "server==3.0"]\n'
        'dev = [{include-group = "proxy"}, "server==3.0"]\n'
    )
    _ = (tmp_path / "uv.lock").write_text("package = []\n")
    checker: Final = _make_checker()

    assert tuple(str(req) for req in checker._load_requirements()) == (
        "runtime==1.0",
        "connector==2.0",
        "server==3.0",
    )


@pytest.mark.parametrize("entry", ('{include-group = "missing"}', '{include-group = "dev", unknown = "value"}', "123"))
def test_load_requirements_rejects_invalid_group_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _ = (tmp_path / "pyproject.toml").write_text(
        f"[project]\ndependencies = []\n[dependency-groups]\ndev = [{entry}]\n"
    )
    _ = (tmp_path / "uv.lock").write_text("package = []\n")
    checker: Final = _make_checker()

    with pytest.raises(RuntimeError, match="Invalid dependency group entry"):
        checker._load_requirements()


def test_load_requirements_preserves_url_hash_and_python_marker(tmp_path: Path) -> None:
    requirements: Final = tmp_path / "requirements.txt"
    _ = requirements.write_text(
        "# pinned connector\n"
        'connector @ https://example.test/connector.tar.gz#sha256=abcd ; python_version >= "3.12"\n'
        "requests==2.0 # ordinary comment\n"
    )
    checker: Final = _make_checker()
    connector, registry = checker._load_requirements(requirements)

    assert connector.url == "https://example.test/connector.tar.gz#sha256=abcd"
    assert str(connector.marker) == 'python_version >= "3.12"'
    assert str(registry) == "requests==2.0"


def test_license_cli_fails_when_requirements_cannot_be_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config: Final = tmp_path / "tests/code_coverage_tests/liccheck.ini"
    config.parent.mkdir(parents=True)
    _ = config.write_text(_LICCHECK_INI.read_text())
    _ = (tmp_path / "requirements.txt").write_text("not a valid requirement\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_licenses.py", "requirements.txt"])

    with pytest.raises(SystemExit) as result:
        check_licenses.main()

    assert result.value.code == 1
    assert "Error parsing requirements" in capsys.readouterr().out

"""Unit tests for `.github/scripts/osv_lockout_filter.py`.

The filter decides whether an osv-scanner finding should fail CI. It must fail only when the
fix is old enough for our resolvers to install it, and must stay quiet while the fix is still
inside the dependency lockout window (`[tool.uv] exclude-newer`, `.npmrc` min-release-age).
Registry lookups are dependency-injected, so nothing here touches the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "osv_lockout_filter.py"

NOW = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def filter_module():
    spec = importlib.util.spec_from_file_location("osv_lockout_filter", SCRIPT_PATH)
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["osv_lockout_filter"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def windows(filter_module):
    return {filter_module.PYPI: timedelta(days=3), filter_module.NPM: timedelta(days=3)}


def _osv_payload(
    *,
    path: str,
    ecosystem: str,
    name: str,
    version: str,
    vuln_id: str,
    fixed: list[str],
    severity: str = "7.5",
) -> dict:
    return {
        "results": [
            {
                "source": {"path": path, "type": "lockfile"},
                "packages": [
                    {
                        "package": {"name": name, "version": version, "ecosystem": ecosystem},
                        "groups": [{"ids": [vuln_id], "aliases": [vuln_id], "max_severity": severity}],
                        "vulnerabilities": [
                            {
                                "id": vuln_id,
                                "affected": [
                                    {
                                        "package": {"ecosystem": ecosystem, "name": name},
                                        "ranges": [
                                            {
                                                "type": "ECOSYSTEM" if ecosystem == "PyPI" else "SEMVER",
                                                "events": [{"introduced": "0"}, {"fixed": f}],
                                            }
                                            for f in fixed
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _finding(filter_module, **overrides):
    defaults = dict(
        ecosystem=filter_module.PYPI,
        package="gitpython",
        installed="3.1.54",
        vuln_id="GHSA-94p4-4cq8-9g67",
        severity="7.5",
        fixed_versions=("3.1.55",),
        source="uv.lock",
    )
    return filter_module.Finding(**{**defaults, **overrides})


def _times(mapping: dict[str, datetime]):
    return lambda ecosystem, name: mapping


class TestParsing:
    def test_extracts_every_field_from_a_real_scanner_payload(self, filter_module):
        payload = _osv_payload(
            path=str(REPO_ROOT / "ui" / "litellm-dashboard" / "package-lock.json"),
            ecosystem="npm",
            name="brace-expansion",
            version="5.0.7",
            vuln_id="GHSA-mh99-v99m-4gvg",
            fixed=["5.0.8"],
        )

        findings = filter_module.parse_findings(payload, REPO_ROOT)

        assert len(findings) == 1
        assert findings[0] == filter_module.Finding(
            ecosystem="npm",
            package="brace-expansion",
            installed="5.0.7",
            vuln_id="GHSA-mh99-v99m-4gvg",
            severity="7.5",
            fixed_versions=("5.0.8",),
            source="ui/litellm-dashboard/package-lock.json",
        )

    def test_ignores_fixed_events_for_a_different_package(self, filter_module):
        payload = _osv_payload(
            path="uv.lock", ecosystem="PyPI", name="gitpython", version="3.1.54", vuln_id="GHSA-x", fixed=["3.1.55"]
        )
        affected = payload["results"][0]["packages"][0]["vulnerabilities"][0]["affected"]
        affected.append(
            {
                "package": {"ecosystem": "PyPI", "name": "some-other-package"},
                "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "9.9.9"}]}],
            }
        )

        assert filter_module.parse_findings(payload, REPO_ROOT)[0].fixed_versions == ("3.1.55",)

    def test_matches_pypi_names_across_normalization(self, filter_module):
        payload = _osv_payload(
            path="uv.lock", ecosystem="PyPI", name="Zope.Interface", version="5.0", vuln_id="GHSA-x", fixed=["5.1"]
        )
        payload["results"][0]["packages"][0]["vulnerabilities"][0]["affected"][0]["package"]["name"] = "zope_interface"

        assert filter_module.parse_findings(payload, REPO_ROOT)[0].fixed_versions == ("5.1",)

    def test_skips_git_ranges_which_carry_commit_hashes_not_versions(self, filter_module):
        payload = _osv_payload(
            path="uv.lock", ecosystem="PyPI", name="gitpython", version="3.1.54", vuln_id="GHSA-x", fixed=["3.1.55"]
        )
        payload["results"][0]["packages"][0]["vulnerabilities"][0]["affected"][0]["ranges"].append(
            {"type": "GIT", "events": [{"introduced": "0"}, {"fixed": "deadbeef"}]}
        )

        assert filter_module.parse_findings(payload, REPO_ROOT)[0].fixed_versions == ("3.1.55",)


class TestVersionOrdering:
    @pytest.mark.parametrize(
        "lower, higher",
        [
            ("1.0.0", "1.0.1"),
            ("1.9.0", "1.10.0"),
            ("1.0.0-rc.1", "1.0.0"),
            ("1.0.0-alpha.1", "1.0.0-alpha.2"),
            ("1.0.0-alpha.1", "1.0.0-alpha.beta"),
            ("1.0.0-alpha", "1.0.0-alpha.1"),
        ],
    )
    def test_semver_precedence(self, filter_module, lower, higher):
        assert filter_module.semver_key(lower) < filter_module.semver_key(higher)

    def test_semver_ignores_build_metadata(self, filter_module):
        assert filter_module.semver_key("1.2.3+build.5") == filter_module.semver_key("1.2.3")

    def test_semver_rejects_non_semver(self, filter_module):
        assert filter_module.semver_key("not-a-version") is None
        assert filter_module.semver_key("1.2") is None


class TestEvaluate:
    def test_defers_when_the_fix_is_inside_the_window(self, filter_module, windows):
        published = NOW - timedelta(days=2)

        verdict = filter_module.evaluate(
            _finding(filter_module), windows, _times({"3.1.55": published}), NOW
        )

        assert isinstance(verdict, filter_module.Deferred)
        assert verdict.target == "3.1.55"
        assert verdict.unlocks_at == published + timedelta(days=3)

    def test_blocks_when_the_fix_is_older_than_the_window(self, filter_module, windows):
        verdict = filter_module.evaluate(
            _finding(filter_module), windows, _times({"3.1.55": NOW - timedelta(days=4)}), NOW
        )

        assert isinstance(verdict, filter_module.Actionable)
        assert verdict.target == "3.1.55"

    def test_blocks_the_moment_the_window_expires(self, filter_module, windows):
        exactly_three_days_old = NOW - timedelta(days=3)

        verdict = filter_module.evaluate(
            _finding(filter_module), windows, _times({"3.1.55": exactly_three_days_old}), NOW
        )

        assert isinstance(verdict, filter_module.Actionable)

    def test_defers_one_second_before_the_window_expires(self, filter_module, windows):
        verdict = filter_module.evaluate(
            _finding(filter_module),
            windows,
            _times({"3.1.55": NOW - timedelta(days=3) + timedelta(seconds=1)}),
            NOW,
        )

        assert isinstance(verdict, filter_module.Deferred)

    def test_targets_the_lowest_fix_above_the_installed_version_not_an_old_branch_backport(
        self, filter_module, windows
    ):
        finding = _finding(filter_module, package="django", installed="5.2.0", fixed_versions=("4.2.9", "5.2.1"))

        verdict = filter_module.evaluate(
            finding,
            windows,
            _times({"4.2.9": NOW - timedelta(days=400), "5.2.1": NOW - timedelta(days=1)}),
            NOW,
        )

        assert isinstance(verdict, filter_module.Deferred)
        assert verdict.target == "5.2.1"

    def test_picks_the_lowest_of_several_reachable_fixes(self, filter_module, windows):
        finding = _finding(filter_module, installed="1.0.0", fixed_versions=("3.0.0", "1.0.1", "2.0.0"))

        verdict = filter_module.evaluate(
            finding,
            windows,
            _times({v: NOW - timedelta(days=10) for v in ("1.0.1", "2.0.0", "3.0.0")}),
            NOW,
        )

        assert verdict.target == "1.0.1"

    def test_blocks_when_no_fix_is_newer_than_the_installed_version(self, filter_module, windows):
        finding = _finding(filter_module, installed="4.0.0", fixed_versions=("3.1.55",))

        verdict = filter_module.evaluate(finding, windows, _times({"3.1.55": NOW}), NOW)

        assert isinstance(verdict, filter_module.Actionable)
        assert verdict.target is None

    def test_blocks_when_the_registry_lookup_fails(self, filter_module, windows):
        verdict = filter_module.evaluate(
            _finding(filter_module), windows, lambda ecosystem, name: None, NOW
        )

        assert isinstance(verdict, filter_module.Actionable)
        assert "registry" in verdict.reason

    def test_blocks_when_the_registry_has_no_date_for_the_target(self, filter_module, windows):
        verdict = filter_module.evaluate(
            _finding(filter_module), windows, _times({"3.1.54": NOW - timedelta(days=9)}), NOW
        )

        assert isinstance(verdict, filter_module.Actionable)

    def test_blocks_for_an_ecosystem_with_no_configured_window(self, filter_module, windows):
        finding = _finding(filter_module, ecosystem="Go", package="golang.org/x/net", installed="0.1.0")

        verdict = filter_module.evaluate(finding, windows, _times({"0.2.0": NOW - timedelta(days=9)}), NOW)

        assert isinstance(verdict, filter_module.Actionable)

    def test_uses_the_npm_window_for_npm_findings(self, filter_module):
        finding = _finding(
            filter_module, ecosystem="npm", package="brace-expansion", installed="5.0.7", fixed_versions=("5.0.8",)
        )
        mixed = {filter_module.PYPI: timedelta(days=0), filter_module.NPM: timedelta(days=3)}

        verdict = filter_module.evaluate(finding, mixed, _times({"5.0.8": NOW - timedelta(days=2)}), NOW)

        assert isinstance(verdict, filter_module.Deferred)


class TestLockoutWindows:
    def test_reads_the_windows_this_repo_actually_enforces(self, filter_module):
        resolved = filter_module.lockout_windows(REPO_ROOT)

        assert not isinstance(resolved, filter_module.WindowError)
        assert resolved == {filter_module.PYPI: timedelta(days=3), filter_module.NPM: timedelta(days=3)}

    def test_reads_a_custom_day_count(self, filter_module, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[tool.uv]\nexclude-newer = "7 days"\n')
        (tmp_path / ".npmrc").write_text("ignore-scripts=true\nmin-release-age=5\n")

        assert filter_module.lockout_windows(tmp_path) == {
            filter_module.PYPI: timedelta(days=7),
            filter_module.NPM: timedelta(days=5),
        }

    def test_no_window_configured_means_no_deferral(self, filter_module, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.uv]\n")

        assert filter_module.lockout_windows(tmp_path) == {
            filter_module.PYPI: timedelta(0),
            filter_module.NPM: timedelta(0),
        }

    def test_rejects_an_exclude_newer_form_it_cannot_interpret(self, filter_module, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.uv]\nexclude-newer = 2026-01-01\n")

        assert isinstance(filter_module.lockout_windows(tmp_path), filter_module.WindowError)

    def test_errors_when_pyproject_is_missing(self, filter_module, tmp_path):
        assert isinstance(filter_module.lockout_windows(tmp_path), filter_module.WindowError)


class TestMain:
    def _write(self, tmp_path: Path, payload: dict) -> Path:
        results = tmp_path / "osv-results.json"
        results.write_text(json.dumps(payload), encoding="utf-8")
        return results

    def test_exits_zero_when_the_only_finding_is_inside_the_window(self, filter_module, tmp_path, capsys):
        results = self._write(
            tmp_path,
            _osv_payload(
                path="uv.lock",
                ecosystem="PyPI",
                name="gitpython",
                version="3.1.54",
                vuln_id="GHSA-94p4-4cq8-9g67",
                fixed=["3.1.55"],
            ),
        )
        recent = datetime.now(timezone.utc) - timedelta(hours=1)

        status = filter_module.main(
            ["--results", str(results), "--repo-root", str(REPO_ROOT)],
            publish_times=_times({"3.1.55": recent}),
        )

        assert status == 0
        assert "GHSA-94p4-4cq8-9g67" in capsys.readouterr().out

    def test_exits_one_when_the_fix_is_installable_today(self, filter_module, tmp_path):
        results = self._write(
            tmp_path,
            _osv_payload(
                path="uv.lock",
                ecosystem="PyPI",
                name="gitpython",
                version="3.1.54",
                vuln_id="GHSA-94p4-4cq8-9g67",
                fixed=["3.1.55"],
            ),
        )
        old = datetime.now(timezone.utc) - timedelta(days=30)

        status = filter_module.main(
            ["--results", str(results), "--repo-root", str(REPO_ROOT)],
            publish_times=_times({"3.1.55": old}),
        )

        assert status == 1

    def test_exits_zero_on_a_clean_scan(self, filter_module, tmp_path):
        results = self._write(tmp_path, {"results": []})

        assert filter_module.main(["--results", str(results), "--repo-root", str(REPO_ROOT)]) == 0

    def test_exits_one_when_the_results_file_is_missing(self, filter_module, tmp_path):
        assert filter_module.main(["--results", str(tmp_path / "nope.json"), "--repo-root", str(REPO_ROOT)]) == 1

    def test_writes_a_step_summary_when_github_provides_one(self, filter_module, tmp_path, monkeypatch):
        results = self._write(
            tmp_path,
            _osv_payload(
                path="uv.lock",
                ecosystem="PyPI",
                name="gitpython",
                version="3.1.54",
                vuln_id="GHSA-94p4-4cq8-9g67",
                fixed=["3.1.55"],
            ),
        )
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

        filter_module.main(
            ["--results", str(results), "--repo-root", str(REPO_ROOT)],
            publish_times=_times({"3.1.55": datetime.now(timezone.utc)}),
        )

        assert "GHSA-94p4-4cq8-9g67" in summary.read_text(encoding="utf-8")


class TestRender:
    def test_names_the_date_a_deferred_finding_starts_failing(self, filter_module, windows):
        deferred = filter_module.Deferred(
            finding=_finding(filter_module),
            target="3.1.55",
            published=datetime(2026, 7, 23, 2, 52, tzinfo=timezone.utc),
            unlocks_at=datetime(2026, 7, 26, 2, 52, tzinfo=timezone.utc),
        )

        report = filter_module.render(windows, (deferred,), ())

        assert "3.1.54 -> 3.1.55" in report
        assert "2026-07-26 02:52 UTC" in report
        assert "Blocking (0)" in report

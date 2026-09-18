import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile

import pytest

from tests.mcp_dependency_tests import check_environment, runner


def wheel(tmp_path: Path, name: str = "litellm") -> Path:
    path = tmp_path / "test.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "litellm-1.dist-info/METADATA",
            f"Name: {name}\nVersion: 1\nRequires-Python: >=3.10,<3.15\n"
            "Requires-Dist: pydantic>=2.10,<3\n"
            "Requires-Dist: mcp>=1.28.1,<2; extra == 'mcp'\n"
            "Provides-Extra: mcp\n",
        )
    return path


def test_project_derives_requirements_and_security_policy(tmp_path: Path) -> None:
    path = wheel(tmp_path)
    policy = tmp_path / "pyproject.toml"
    policy.write_text(
        '[tool.uv]\nconstraint-dependencies=["packaging>=24"]\noverride-dependencies=["cryptography>=50"]'
    )
    candidate = tomllib.loads(runner.project_text(path, "mcp", tmp_path))
    core = tomllib.loads(runner.project_text(path, "core", tmp_path))
    assert candidate["project"]["requires-python"] == ">=3.10,<3.15"
    assert "mcp>=1.28.1,<2; extra == 'mcp'" in candidate["project"]["dependencies"]
    assert "httpx2>=2.12.0" in candidate["project"]["dependencies"]
    assert candidate["tool"]["uv"]["override-dependencies"] == ["cryptography>=50", "mcp==2.2.0"]
    assert candidate["tool"]["uv"]["constraint-dependencies"] == ["packaging>=24"]
    assert core["tool"]["uv"]["override-dependencies"] == ["cryptography>=50"]
    assert "httpx2>=2.12.0" not in core["project"]["dependencies"]


def test_rejects_missing_extra(tmp_path: Path) -> None:
    path = wheel(tmp_path)
    with pytest.raises(ValueError, match="does not provide extra proxy"):
        runner.project_text(path, "proxy")


def test_rejects_other_distribution(tmp_path: Path) -> None:
    path = wheel(tmp_path, "unrelated")
    with pytest.raises(ValueError, match="expected a litellm wheel"):
        runner.wheel_project(path)


def test_rejects_ambiguous_metadata(tmp_path: Path) -> None:
    path = wheel(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("other.dist-info/METADATA", "Name: other")
    with pytest.raises(ValueError, match="exactly one wheel METADATA"):
        runner.wheel_project(path)


@pytest.mark.parametrize("change", ["requirements", "profile", "mode"])
def test_rejects_stale_snapshot(change: str) -> None:
    original = runner.fingerprint("requirements", "mcp", "locked")
    snapshot = f"# inputs-sha256: {original}\nmcp==2.2.0\n"
    with pytest.raises(ValueError, match="snapshot is stale"):
        runner.validate_snapshot(
            snapshot,
            "changed" if change == "requirements" else "requirements",
            "proxy" if change == "profile" else "mcp",
            "minimum" if change == "mode" else "locked",
        )


def test_accepts_current_snapshot() -> None:
    digest = runner.fingerprint("requirements", "mcp", "locked")
    runner.validate_snapshot(f"# inputs-sha256: {digest}\n", "requirements", "mcp", "locked")
    assert digest == runner.fingerprint("requirements", "mcp", "locked")


def test_inventory_honors_target_python_markers() -> None:
    snapshot = "foo==1 ; python_version < '3.13' \\\n    --hash=sha256:abc\nfoo==2 ; python_version >= '3.13' \\\n    --hash=sha256:def\n"
    report = {"environment": {"python_version": "3.13"}, "installed": {"litellm": "1", "foo": "2"}}
    runner.verify_inventory(snapshot, report, {"litellm": "1"})
    assert runner.locked_versions(snapshot, {"python_version": "3.12"}) == {"foo": "1"}


@pytest.mark.parametrize("installed", [{"foo": "2"}, {}, {"foo": "1", "unexpected": "1"}])
def test_inventory_rejects_drift(installed: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="do not match snapshot"):
        runner.verify_inventory("foo==1\n", {"environment": {}, "installed": installed}, {})


def test_inventory_rejects_invalid_report() -> None:
    with pytest.raises(ValueError, match="invalid environment inventory"):
        runner.verify_inventory("foo==1\n", {"environment": None, "installed": None}, {})


def test_existing_environment_is_never_modified(tmp_path: Path) -> None:
    path = wheel(tmp_path)
    profile = runner.project_text(path, "mcp")
    (tmp_path / "mcp-locked.txt").write_text(f"# inputs-sha256: {runner.fingerprint(profile, 'mcp', 'locked')}\n")
    sentinel = tmp_path / "existing"
    sentinel.mkdir()
    (sentinel / "owned").write_text("preserve")
    with pytest.raises(ValueError, match="existing environments are never modified"):
        runner.check(path, "mcp", "locked", tmp_path, "3.12", sentinel)
    assert (sentinel / "owned").read_text() == "preserve"


def test_subprocess_failure_is_not_a_pass(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError) as error:
        runner.run((sys.executable, "-c", "raise SystemExit(7)"), tmp_path)
    assert error.value.returncode == 7


def test_subprocess_uses_isolated_working_directory(tmp_path: Path) -> None:
    runner.run((sys.executable, "-c", "from pathlib import Path; Path('proof').write_text('isolated')"), tmp_path)
    assert (tmp_path / "proof").read_text() == "isolated"


def proxy_wheel(tmp_path: Path, companion_requirement: str) -> Path:
    path = wheel(tmp_path)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "litellm-1.dist-info/METADATA",
            "Name: litellm\nVersion: 1\nRequires-Python: >=3.10,<3.15\nProvides-Extra: proxy\n",
        )
    for name in runner.COMPANIONS:
        with zipfile.ZipFile(tmp_path / f"{name.replace('-', '_')}-1-py3-none-any.whl", "w") as archive:
            archive.writestr(
                f"{name}-1.dist-info/METADATA",
                f"Name: {name}\nVersion: 1\nRequires-Dist: {companion_requirement}\n",
            )
    return path


def test_same_filename_companion_dependency_change_invalidates_snapshot(tmp_path: Path) -> None:
    path = proxy_wheel(tmp_path, "packaging>=24")
    old_project = runner.project_text(path, "proxy")
    snapshot = f"# inputs-sha256: {runner.fingerprint(old_project, 'proxy', 'locked')}\n"
    proxy_wheel(tmp_path, "packaging>=26")
    with pytest.raises(ValueError, match="snapshot is stale"):
        runner.validate_snapshot(snapshot, runner.project_text(path, "proxy"), "proxy", "locked")


def test_changed_cutoff_invalidates_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = wheel(tmp_path)
    candidate = (runner.HERE / "candidate.toml").read_text()
    (tmp_path / "candidate.toml").write_text(candidate)
    monkeypatch.setattr(runner, "HERE", tmp_path)
    project = runner.project_text(path, "mcp")
    snapshot = f"# inputs-sha256: {runner.fingerprint(project, 'mcp', 'locked')}\n"
    (tmp_path / "candidate.toml").write_text(
        candidate.replace(tomllib.loads(candidate)["exclude-newer"], "2000-01-01T00:00:00Z")
    )
    with pytest.raises(ValueError, match="snapshot is stale"):
        runner.validate_snapshot(snapshot, runner.project_text(path, "mcp"), "mcp", "locked")


@pytest.mark.parametrize("profile,mode", [("core", "minimum"), ("mcp", "locked")])
def test_lock_cli_generates_hashed_replayable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str, mode: str
) -> None:
    path = wheel(tmp_path)
    snapshots = tmp_path / "snapshots"
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "lock", "--wheel", str(path), "--profile", profile, "--mode", mode, "--snapshots", str(snapshots)],
    )
    runner.main()
    snapshot = (snapshots / f"{profile}-{mode}.txt").read_text()
    runner.validate_snapshot(snapshot, runner.project_text(path, profile), profile, mode)
    versions = runner.locked_versions(snapshot, {"python_version": "3.12", "python_full_version": "3.12.12"})
    assert "--hash=sha256:" in snapshot
    if profile == "core":
        assert versions["pydantic"] == "2.10.0"
        assert "mcp" not in versions
    else:
        assert versions["mcp"] == "2.2.0"
        assert "httpx2" in versions


def test_check_cli_requires_explicit_new_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = wheel(tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "check", "--wheel", str(path), "--profile", "core", "--mode", "locked"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert tuple(tmp_path.iterdir()) == (path,)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_proxy_rejects_missing_or_ambiguous_companions(tmp_path: Path, ambiguous: bool) -> None:
    path = proxy_wheel(tmp_path, "packaging>=24")
    companion = next(tmp_path.glob("litellm_enterprise*.whl"))
    if ambiguous:
        (tmp_path / "litellm_enterprise-2-py3-none-any.whl").write_bytes(companion.read_bytes())
    else:
        companion.unlink()
    with pytest.raises(ValueError, match="exactly one enterprise"):
        runner.project_text(path, "proxy")


@pytest.mark.parametrize("name", ["Foo.Bar", "Foo__BAR", "foo--bar", "foo-bar"])
def test_inventory_accepts_equivalent_distribution_names(tmp_path: Path, name: str) -> None:
    metadata = tmp_path / "foo_bar-1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1\n")
    installed = check_environment.installed_versions(importlib.metadata.distributions(path=[str(tmp_path)]))
    runner.verify_inventory("foo-bar==1\n", {"environment": {}, "installed": installed}, {})
    assert installed == {"foo-bar": "1"}

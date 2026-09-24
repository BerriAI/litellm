import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final, cast

import pytest

HARNESS: Final = Path(__file__).parents[2] / ".github" / "scripts" / "run_merge_smoke.py"

CASE_IDS: Final = (
    "CHAT-JSON",
    "CHAT-TEXT-STREAM",
    "CHAT-TOOL-STREAM",
    "MODEL-ALLOW",
    "MODEL-DENY",
    "COST-EXPLICIT",
    "COST-ZERO",
    "LOG-CONTENT-ON",
    "LOG-CONTENT-OFF",
    "CALLBACK-SUCCESS",
    "CALLBACK-FAILURE",
)


def _write_fake_tests(root: Path, body: str) -> Path:
    package: Final = root / "fake_tests"
    package.mkdir()
    (package / "test_cases.py").write_text(body)
    return package


def _manifest(root: Path, **overrides: str) -> Path:
    cases: Final[dict[str, str]] = {
        case_id: f"fake_tests/test_cases.py::test_{case_id.lower().replace('-', '_')}" for case_id in CASE_IDS
    }
    cases.update(overrides)
    path: Final = root / "manifest.json"
    path.write_text(json.dumps({"cases": cases}))
    return path


def _run(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(HARNESS), *argv],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _passing_tests() -> str:
    return "\n".join(f"def test_{case_id.lower().replace('-', '_')}():\n    assert True" for case_id in CASE_IDS)


def test_all_eleven_cases_pass(tmp_path: Path) -> None:
    _write_fake_tests(tmp_path, _passing_tests())
    manifest: Final = _manifest(tmp_path)

    proc: Final = _run(tmp_path, "pytest", "--manifest", str(manifest), "--rootdir", str(tmp_path))

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("PASS") >= 11
    for case_id in CASE_IDS:
        assert f"{case_id}  PASS" in proc.stdout


def test_missing_test_node_id_fails(tmp_path: Path) -> None:
    _write_fake_tests(tmp_path, _passing_tests())
    manifest: Final = _manifest(tmp_path, **{"COST-ZERO": "fake_tests/test_cases.py::test_does_not_exist"})

    proc: Final = _run(tmp_path, "pytest", "--manifest", str(manifest), "--rootdir", str(tmp_path))

    assert proc.returncode != 0
    assert "COST-ZERO" in proc.stderr or "test_does_not_exist" in proc.stderr


def test_skipped_case_fails(tmp_path: Path) -> None:
    _write_fake_tests(
        tmp_path,
        _passing_tests().replace(
            "def test_cost_zero():\n    assert True",
            "def test_cost_zero():\n    import pytest\n    pytest.skip('nope')",
        ),
    )
    manifest: Final = _manifest(tmp_path)

    proc: Final = _run(tmp_path, "pytest", "--manifest", str(manifest), "--rootdir", str(tmp_path))

    assert proc.returncode != 0
    assert "COST-ZERO" in proc.stderr


def test_xfail_case_fails(tmp_path: Path) -> None:
    _write_fake_tests(
        tmp_path,
        "import pytest\n"
        + _passing_tests().replace(
            "def test_cost_zero():\n    assert True",
            "@pytest.mark.xfail\ndef test_cost_zero():\n    assert False",
        ),
    )
    manifest: Final = _manifest(tmp_path)

    proc: Final = _run(tmp_path, "pytest", "--manifest", str(manifest), "--rootdir", str(tmp_path))

    assert proc.returncode != 0
    assert "COST-ZERO" in proc.stderr


def test_xpass_case_fails(tmp_path: Path) -> None:
    _write_fake_tests(
        tmp_path,
        "import pytest\n"
        + _passing_tests().replace(
            "def test_cost_zero():\n    assert True",
            "@pytest.mark.xfail\ndef test_cost_zero():\n    assert True",
        ),
    )
    manifest: Final = _manifest(tmp_path)

    proc: Final = _run(tmp_path, "pytest", "--manifest", str(manifest), "--rootdir", str(tmp_path))

    assert proc.returncode != 0
    assert "COST-ZERO" in proc.stderr


def test_duplicate_manifest_key_fails(tmp_path: Path) -> None:
    manifest: Final = tmp_path / "manifest.json"
    manifest.write_text('{"cases": {"CHAT-JSON": "a::b", "CHAT-JSON": "a::c"}}')

    proc: Final = _run(tmp_path, "pytest", "--manifest", str(manifest))

    assert proc.returncode != 0
    assert "CHAT-JSON" in proc.stderr


def test_missing_case_id_fails(tmp_path: Path) -> None:
    manifest: Final = tmp_path / "manifest.json"
    cases: Final = {c: f"t::{c}" for c in CASE_IDS[:-1]}
    manifest.write_text(json.dumps({"cases": cases}))

    proc: Final = _run(tmp_path, "pytest", "--manifest", str(manifest))

    assert proc.returncode != 0
    assert "case ids" in proc.stderr


def test_extra_case_id_fails(tmp_path: Path) -> None:
    manifest: Final = tmp_path / "manifest.json"
    cases: Final = {c: f"t::{c}" for c in CASE_IDS}
    cases["EXTRA"] = "t::x"
    manifest.write_text(json.dumps({"cases": cases}))

    proc: Final = _run(tmp_path, "pytest", "--manifest", str(manifest))

    assert proc.returncode != 0
    assert "case ids" in proc.stderr


def test_teardown_error_fails(tmp_path: Path) -> None:
    body: Final = (
        "import pytest\n\n@pytest.fixture\ndef boom():\n    yield\n    raise RuntimeError('teardown-boom')\n\n"
        + _passing_tests().replace(
            "def test_cost_zero():\n    assert True",
            "def test_cost_zero(boom):\n    assert True",
        )
    )
    _write_fake_tests(tmp_path, body)
    manifest: Final = _manifest(tmp_path)

    proc: Final = _run(tmp_path, "pytest", "--manifest", str(manifest), "--rootdir", str(tmp_path))

    assert proc.returncode != 0
    assert "COST-ZERO" in proc.stderr


def _fake_litellm(tmp_path: Path, script: str) -> Path:
    path: Final = tmp_path / "fake-litellm"
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(script))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_proxy_startup_exits_early_fails(tmp_path: Path) -> None:
    fake: Final = _fake_litellm(tmp_path, "import sys\nsys.exit(1)\n")
    diagnostics: Final = tmp_path / "diag"

    proc: Final = _run(
        tmp_path,
        "proxy-startup",
        "--diagnostics-dir",
        str(diagnostics),
        "--litellm-bin",
        str(fake),
    )

    assert proc.returncode != 0
    assert "exited early" in proc.stderr
    assert (diagnostics / "proxy.log").exists()


def test_proxy_startup_readiness_timeout_fails(tmp_path: Path) -> None:
    fake: Final = _fake_litellm(
        tmp_path,
        "import os, pathlib, sys, time\npathlib.Path(sys.argv[0]).with_name('fake.pid').write_text(str(os.getpid()))\ntime.sleep(3600)\n",
    )
    diagnostics: Final = tmp_path / "diag"

    proc: Final = _run(
        tmp_path,
        "proxy-startup",
        "--diagnostics-dir",
        str(diagnostics),
        "--litellm-bin",
        str(fake),
        "--ready-deadline",
        "3",
        "--shutdown-deadline",
        "2",
    )

    assert proc.returncode != 0
    assert "readiness" in proc.stderr
    assert (diagnostics / "proxy.log").exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "fake.pid").read_text()), 0)


def test_proxy_startup_healthy_succeeds(tmp_path: Path) -> None:
    fake: Final = _fake_litellm(
        tmp_path,
        """
        import http.server, json, sys
        port = int(sys.argv[sys.argv.index("--port") + 1])
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"status": "healthy", "db": "Not connected"}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *a):
                pass
        http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
        """,
    )
    diagnostics: Final = tmp_path / "diag"

    proc: Final = _run(
        tmp_path,
        "proxy-startup",
        "--diagnostics-dir",
        str(diagnostics),
        "--litellm-bin",
        str(fake),
        "--ready-deadline",
        "15",
    )

    assert proc.returncode == 0, proc.stderr
    result: Final = cast(dict[str, object], json.loads((diagnostics / "result.json").read_text()))
    assert result["outcome"] == "ok"
    assert result["readiness"] == '{"status": "healthy", "db": "Not connected"}'


def test_proxy_startup_sigterm_ignored_forces_kill(tmp_path: Path) -> None:
    fake: Final = _fake_litellm(
        tmp_path,
        """
        import http.server, json, os, pathlib, signal, sys
        port = int(sys.argv[sys.argv.index("--port") + 1])
        pathlib.Path(sys.argv[0]).with_name("fake.pid").write_text(str(os.getpid()))
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"status": "healthy", "db": "Not connected"}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *a):
                pass
        http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
        """,
    )
    diagnostics: Final = tmp_path / "diag"

    proc: Final = _run(
        tmp_path,
        "proxy-startup",
        "--diagnostics-dir",
        str(diagnostics),
        "--litellm-bin",
        str(fake),
        "--ready-deadline",
        "15",
        "--shutdown-deadline",
        "2",
    )

    assert proc.returncode != 0
    assert "forced kill" in proc.stderr
    result: Final = cast(dict[str, object], json.loads((diagnostics / "result.json").read_text()))
    assert result["outcome"] == "failed"
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "fake.pid").read_text()), 0)


def test_proxy_startup_waits_through_not_ready_status(tmp_path: Path) -> None:
    fake: Final = _fake_litellm(
        tmp_path,
        """
        import http.server, json, sys
        port = int(sys.argv[sys.argv.index("--port") + 1])
        hits = [0]
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hits[0] += 1
                if hits[0] <= 2:
                    self.send_response(503)
                    self.end_headers()
                    return
                body = json.dumps({"status": "healthy", "db": "Not connected"}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *a):
                pass
        http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
        """,
    )
    diagnostics: Final = tmp_path / "diag"

    proc: Final = _run(
        tmp_path,
        "proxy-startup",
        "--diagnostics-dir",
        str(diagnostics),
        "--litellm-bin",
        str(fake),
        "--ready-deadline",
        "15",
    )

    assert proc.returncode == 0, proc.stderr


def test_proxy_startup_wrong_body_fails(tmp_path: Path) -> None:
    fake: Final = _fake_litellm(
        tmp_path,
        """
        import http.server, json, sys
        port = int(sys.argv[sys.argv.index("--port") + 1])
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"status": "healthy", "db": "connected"}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *a):
                pass
        http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
        """,
    )
    diagnostics: Final = tmp_path / "diag"

    proc: Final = _run(
        tmp_path,
        "proxy-startup",
        "--diagnostics-dir",
        str(diagnostics),
        "--litellm-bin",
        str(fake),
        "--ready-deadline",
        "15",
    )

    assert proc.returncode != 0
    assert "connected" in proc.stderr


def test_interpreter_expect_mismatch_fails() -> None:
    proc: Final = _run(Path.cwd(), "interpreter", "--expect", "9.99")

    assert proc.returncode != 0
    assert "9.99" in proc.stderr


def test_interpreter_expect_match_passes() -> None:
    expect: Final = f"{sys.version_info.major}.{sys.version_info.minor}"

    proc: Final = _run(Path.cwd(), "interpreter", "--expect", expect)

    assert proc.returncode == 0
    assert f"OK interpreter {expect}" in proc.stdout

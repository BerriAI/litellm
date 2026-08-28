from __future__ import annotations

import http.client
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

from bench_sdk import file_bytes, lock_text, snapshot, wheel_info
from bench_sdk_runtime import PROBE, probe, provider, runtime_environment, summary

FAKE_SDK: Final = """
import sys
import time
if sys.argv[1] != "diagnostic":
    assert "json" not in sys.modules
    assert "typing" not in sys.modules
    assert "http.client" not in sys.modules
    assert "psutil" not in sys.modules
    assert "pyperf" not in sys.modules
time.sleep(0.02)

def completion(**arguments):
    import json
    from types import SimpleNamespace
    from urllib.request import Request, urlopen
    request = Request(
        arguments["api_base"] + "/chat/completions",
        data=json.dumps({"model": "benchmark-model", "messages": arguments["messages"]}).encode(),
        headers={"Authorization": "Bearer " + arguments["api_key"], "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        payload = json.load(response)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=payload["choices"][0]["message"]["content"]
    ))])
"""


def make_wheel(directory: Path) -> Path:
    wheel: Final = directory / "litellm-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("litellm/__init__.py", FAKE_SDK)
        archive.writestr(
            "litellm-0.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: litellm\nVersion: 0.0.0\nProvides-Extra: proxy\n",
        )
        archive.writestr(
            "litellm-0.0.0.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        archive.writestr("litellm-0.0.0.dist-info/RECORD", "")
    return wheel


def make_target(directory: Path, code: str = FAKE_SDK) -> Path:
    target: Final = directory / "target"
    subprocess.run((sys.executable, "-I", "-m", "venv", "--without-pip", str(target)), check=True)
    site: Final = target / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (site / "litellm.py").write_text(code)
    (directory / "litellm.py").write_text('raise RuntimeError("Imported from current directory")')
    return target / "bin" / "python"


class BenchmarkTests(unittest.TestCase):
    def test_parallel_runs_use_distinct_environments_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            wheel: Final = make_wheel(root)

            def run(index: int) -> dict[str, object]:
                result: Final = subprocess.run(
                    (
                        sys.executable,
                        str(Path(__file__).with_name("bench_sdk.py")),
                        "--wheel",
                        str(wheel),
                        "--wheelhouse",
                        str(root),
                        "--output",
                        str(root / str(index)),
                        "--samples",
                        "1",
                        "--install-samples",
                        "1",
                    ),
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)

            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = tuple(pool.map(run, (0, 1)))
            self.assertNotEqual(
                first["environment"]["runtime_variables"]["HOME"],
                second["environment"]["runtime_variables"]["HOME"],
            )
            self.assertTrue((root / "0" / "result.json").exists())
            self.assertTrue((root / "1" / "result.json").exists())

    def test_local_snapshot_preserves_working_changes_without_build_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            source: Final = root / "checkout"
            source.mkdir()
            subprocess.run(("git", "init", "-q", str(source)), check=True)
            (source / ".gitignore").write_text("target/\n")
            (source / "tracked.py").write_text("original")
            (source / "deleted.py").write_text("deleted")
            subprocess.run(("git", "-C", str(source), "add", "."), check=True)
            (source / "tracked.py").write_text("working changes")
            (source / "deleted.py").unlink()
            (source / "untracked.py").write_text("new source")
            (source / "linked.py").symlink_to("tracked.py")
            (source / "target").mkdir()
            (source / "target" / "old.so").write_text("old build")
            copied: Final = root / "copied"
            snapshot(source, copied)
            self.assertEqual((copied / "tracked.py").read_text(), "working changes")
            self.assertEqual((copied / "untracked.py").read_text(), "new source")
            self.assertFalse((copied / "deleted.py").exists())
            self.assertFalse((copied / "target").exists())
            self.assertFalse((copied / ".git").exists())
            self.assertTrue((copied / "linked.py").is_symlink())
            self.assertEqual((copied / "linked.py").read_text(), "working changes")
            (copied / "tracked.py").write_text("build mutated its copy")
            self.assertEqual((copied / "linked.py").read_text(), "build mutated its copy")
            self.assertEqual((source / "tracked.py").read_text(), "working changes")

    def test_cli_requires_one_explicit_source(self) -> None:
        script: Final = str(Path(__file__).with_name("bench_sdk.py"))
        for arguments in (("--output", "/unused"), ("--local", ".", "--package", "0.0.0", "--output", "/unused")):
            result: Final = subprocess.run((sys.executable, script, *arguments), capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)

    def test_summary_retains_samples_and_does_not_invent_single_sample_variance(self) -> None:
        result: Final = summary((1.0, 2.0, 6.0), "seconds")
        self.assertEqual(result["samples"], (1.0, 2.0, 6.0))
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["mean"], 3.0)
        self.assertEqual(result["mad"], 1.0)
        self.assertIsNone(summary((2.0,), "seconds")["stdev"])
        with self.assertRaises(SystemExit):
            summary((), "seconds")

    def test_size_counts_bytecode_but_not_interpreter_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            (root / "code.py").write_bytes(b"123")
            (root / "code.pyc").write_bytes(b"12345")
            (root / "python").symlink_to(sys.executable)
            self.assertEqual(file_bytes(root), 8)

    def test_wheel_lock_preserves_extras_version_and_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            wheel: Final = wheel_info(make_wheel(root))
            self.assertEqual(wheel.tags, ("py3-none-any",))
            self.assertEqual(wheel.extras, ("proxy",))
            self.assertGreater(wheel.uncompressed_bytes, 0)
            self.assertEqual(lock_text((wheel,), "proxy"), f"litellm[proxy]==0.0.0 --hash=sha256:{wheel.sha256}\n")

    def test_runtime_does_not_inherit_credentials_or_import_overrides(self) -> None:
        environment: Final = runtime_environment(Path("/tmp/benchmark-home"))
        for name in ("OPENAI_API_KEY", "AWS_ACCESS_KEY_ID", "PYTHONPATH", "HTTP_PROXY", "HTTPS_PROXY"):
            self.assertNotIn(name, environment)
        self.assertEqual(environment["LITELLM_LOCAL_MODEL_COST_MAP"], "True")

    def test_independent_providers_reject_unexpected_requests(self) -> None:
        with provider() as (first, first_requests), provider() as (second, second_requests):
            self.assertNotEqual(first, second)
            connection: Final = http.client.HTTPConnection(first.removeprefix("http://").removesuffix("/v1"))
            try:
                connection.request("POST", "/wrong", body="{}", headers={"Content-Type": "application/json"})
                response: Final = connection.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
            finally:
                connection.close()
            self.assertFalse(first_requests.get_nowait())
            self.assertTrue(second_requests.empty())

    def test_probe_uses_fresh_isolated_import_and_real_http_and_separate_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            python: Final = make_target(root)
            with (root / "probe.log").open("w") as log, provider() as (url, requests):
                timing, _ = probe(python, url, root, runtime_environment(root), log, requests)
                self.assertGreater(timing["import_ns"], 15_000_000)
                self.assertGreater(timing["launch_to_import_ns"], timing["import_ns"])
                self.assertGreater(timing["launch_to_first_response_ns"], timing["import_to_first_response_ns"])
                self.assertGreater(timing["first_request_ns"], 0)
                self.assertGreater(timing["second_request_ns"], 0)
                _, diagnostics = probe(python, url, root, runtime_environment(root), log, requests, diagnostic=True)
                self.assertEqual(
                    tuple(stage["stage"] for stage in diagnostics),
                    (
                        "after_import",
                        "after_configuration",
                        "after_first_response",
                        "after_second_response",
                    ),
                )
                self.assertIn("litellm", diagnostics[0]["modules"])
                self.assertGreater(diagnostics[0]["rss_bytes"], 0)

    def test_probe_timeout_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            python: Final = make_target(root, "import time\ntime.sleep(30)\n")
            with (root / "probe.log").open("w") as log, provider() as (url, requests):
                with self.assertRaisesRegex(SystemExit, "timed out"):
                    probe(python, url, root, runtime_environment(root), log, requests, timeout=0.1)

    def test_egress_guard_rejects_external_dns_before_network_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            python: Final = make_target(root, 'import socket\nsocket.getaddrinfo("example.invalid", 443)\n')
            result: Final = subprocess.run(
                (str(python), "-I", "-B", "-c", PROBE, "import_exit"),
                cwd=root,
                env=runtime_environment(root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("BENCH_EGRESS_BLOCKED", result.stderr)

    def test_full_cli_offline_artifacts_and_refusal_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            wheel: Final = make_wheel(root)
            output: Final = root / "results"
            command: Final = (
                sys.executable,
                str(Path(__file__).with_name("bench_sdk.py")),
                "--wheel",
                str(wheel),
                "--wheelhouse",
                str(root),
                "--output",
                str(output),
                "--samples",
                "2",
                "--install-samples",
                "2",
                "--extras",
                "proxy",
            )
            result: Final = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr + (output / "run.log").read_text())
            payload: Final = json.loads(result.stdout)
            self.assertEqual(payload, json.loads((output / "result.json").read_text()))
            self.assertEqual(payload["timings"]["import"]["count"], 2)
            self.assertEqual(payload["source"]["kind"], "wheel")
            self.assertFalse(payload["source"]["built_from_source"])
            self.assertEqual(payload["timings"]["offline_install"]["count"], 2)
            self.assertEqual(payload["sizes"]["resolved_wheelhouse_bytes"], wheel.stat().st_size)
            self.assertGreater(
                payload["sizes"]["installed_delta_bytes"][0], payload["sizes"]["root_uncompressed_bytes"]
            )
            self.assertEqual(
                tuple(
                    item["metadata"]["name"]
                    for item in json.loads((output / "installed.json").read_text())["installed"]
                ),
                ("litellm",),
            )
            duplicate: Final = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("already exists", duplicate.stderr)
            published: Final = subprocess.run(
                (
                    sys.executable,
                    str(Path(__file__).with_name("bench_sdk.py")),
                    "--package",
                    "0.0.0",
                    "--wheelhouse",
                    str(root),
                    "--output",
                    str(root / "published"),
                    "--samples",
                    "1",
                    "--install-samples",
                    "1",
                ),
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(published.returncode, 0, published.stderr)
            self.assertEqual(
                json.loads(published.stdout)["source"],
                {
                    "kind": "package",
                    "requested": "0.0.0",
                    "built_from_source": False,
                },
            )
            self.assertNotIn("'wheel'", (root / "published" / "run.log").read_text())


if __name__ == "__main__":
    unittest.main()

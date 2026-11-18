import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / ".github/scripts/docker_pull_with_retries.sh"
DOCKER: Final = r'''#!/usr/bin/env bash
set -eu
count=$(cat "$STATE/count" 2>/dev/null || printf 0)
count=$((count + 1))
printf '%s' "$count" > "$STATE/count"
printf '%s\n' "$*" >> "$STATE/calls"
sed -n "${count}p" "$STATE/messages" >&2
status=$(sed -n "${count}p" "$STATE/statuses")
exit "${status:-99}"
'''
SLEEP: Final = '#!/usr/bin/env bash\nprintf "%s\\n" "$1" >> "$STATE/sleeps"\n'


class DockerPullRetryTests(unittest.TestCase):
    def run_pull(
        self, statuses: tuple[int, ...], messages: tuple[str, ...], args: tuple[str, ...] = ("alpine:3.20",)
    ) -> tuple[subprocess.CompletedProcess[str], tuple[str, ...], tuple[str, ...]]:
        with tempfile.TemporaryDirectory() as folder:
            state: Final = Path(folder)
            (state / "statuses").write_text("\n".join(map(str, statuses)))
            (state / "messages").write_text("\n".join(messages))
            for name, source in (("docker", DOCKER), ("sleep", SLEEP)):
                (state / name).write_text(source)
                (state / name).chmod(0o755)
            result: Final = subprocess.run(
                ("bash", str(SCRIPT), *args),
                env={**os.environ, "STATE": folder, "PATH": folder + os.pathsep + os.environ["PATH"]},
                capture_output=True,
                text=True,
                timeout=10,
            )
            calls: Final = tuple((state / "calls").read_text().splitlines()) if (state / "calls").exists() else ()
            sleeps: Final = tuple((state / "sleeps").read_text().splitlines()) if (state / "sleeps").exists() else ()
            return result, calls, sleeps

    def test_success_runs_once(self) -> None:
        result, calls, sleeps = self.run_pull((0,), ("downloaded",))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls, ("pull alpine:3.20",))
        self.assertEqual(sleeps, ())

    def test_transient_failures_recover_without_rerunning_tests(self) -> None:
        for message in (
            "docker: received unexpected HTTP status: 502 Bad Gateway",
            "HTTP status: 503 Service Unavailable",
            "HTTP status: 429 Too Many Requests",
            "toomanyrequests: rate limit exceeded",
            "net/http: TLS handshake timeout",
            "read: connection reset by peer",
            "dial tcp: i/o timeout",
        ):
            with self.subTest(message=message):
                result, calls, sleeps = self.run_pull((1, 0), (message, "downloaded"))
                self.assertEqual(result.returncode, 0)
                self.assertEqual(calls, ("pull alpine:3.20",) * 2)
                self.assertEqual(sleeps, ("2",))
                self.assertIn(message, result.stderr)

    def test_timeout_status_is_retried(self) -> None:
        result, calls, sleeps = self.run_pull((124, 0), ("", "downloaded"))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, ("2",))

    def test_exhausted_transient_failure_stays_failed(self) -> None:
        result, calls, sleeps = self.run_pull((1, 1, 1, 0), ("HTTP status: 502 Bad Gateway",) * 4)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, ("2", "4"))

    def test_exhausted_timeout_preserves_exit_status(self) -> None:
        result, calls, sleeps = self.run_pull((124, 124, 124), ("", "", ""))
        self.assertEqual(result.returncode, 124)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, ("2", "4"))

    def test_permanent_failures_do_not_retry(self) -> None:
        for status, message in (
            (1, "unauthorized: incorrect credentials"),
            (1, "manifest unknown"),
            (1, "x509: certificate signed by unknown authority"),
            (125, "invalid reference format"),
            (127, "docker: command not found"),
            (137, "Killed"),
        ):
            with self.subTest(message=message):
                result, calls, sleeps = self.run_pull((status, 0), (message, "downloaded"))
                self.assertEqual(result.returncode, status)
                self.assertEqual(len(calls), 1)
                self.assertEqual(sleeps, ())

    def test_invalid_arguments_do_not_run_docker(self) -> None:
        for args in ((), ("",), ("--help",), ("one", "two")):
            with self.subTest(args=args):
                result, calls, sleeps = self.run_pull((0,), ("",), args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, ())
                self.assertEqual(sleeps, ())


if __name__ == "__main__":
    unittest.main()

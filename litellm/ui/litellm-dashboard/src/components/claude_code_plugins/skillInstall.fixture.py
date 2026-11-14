import errno
import json
import os
import pty
import select
import shutil
import signal
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, TypeAlias, cast

Environment: TypeAlias = Mapping[str, str]


def external_command(tool: str) -> None:
    root: Final = Path(os.environ["SKILL_FIXTURE_ROOT"])
    args: Final = tuple(sys.argv[1:])
    header: Final = sys.stdin.read() if tool == "curl" and args[args.index("--header") + 1] == "@-" else ""
    (root / f"{tool}.json").write_text(json.dumps({"args": args, "env": dict(os.environ), "header": header}))
    if tool == "curl":
        output: Final = Path(args[args.index("--output") + 1])
        (output.parent / ".skill-install-fixture").write_text(str(root))
        if os.environ["SKILL_FIXTURE_FAIL"] == "1":
            sys.exit(22)
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("SKILL.md", "fixture skill")
        return
    source: Final = Path(args[args.index("add") + 1])
    if "--copy" in args:
        shutil.copytree(source, root / "installed")
    else:
        (root / "installed").symlink_to(source, target_is_directory=True)


def run(command: str, fail: bool) -> None:
    secret: Final = "private-fixture-key"
    with tempfile.TemporaryDirectory() as directory:
        root: Final = Path(directory)
        (root / "bin").mkdir()
        (root / "tmp").mkdir()
        for tool in ("curl", "npx"):
            executable: Path = root / "bin" / tool
            executable.write_text(f"#!{sys.executable}\n" + Path(__file__).read_text())
            executable.chmod(0o700)
        environment: Final[Environment] = {
            "PATH": f"{root / 'bin'}{os.pathsep}{os.defpath}",
            "TMPDIR": str(root / "tmp"),
            "SKILL_FIXTURE_ROOT": str(root),
            "SKILL_FIXTURE_FAIL": "1" if fail else "0",
            "skill_key": "previously-exported-value",
        }
        child, terminal = pty.fork()
        if child == 0:
            os.execve("/bin/bash", ["bash", "-c", command], environment)
        output: bytes = b""  # rebind-ok: accumulate partial PTY reads
        sent: bool = False  # rebind-ok: send the key once after its prompt arrives
        ended: bool = False  # rebind-ok: distinguish PTY EOF from timeout cleanup
        deadline: Final = time.monotonic() + 5
        try:
            while True:
                ready, _, _ = select.select([terminal], [], [], max(0, deadline - time.monotonic()))
                if not ready:
                    raise TimeoutError("Install command did not finish")
                try:
                    chunk: bytes = os.read(terminal, 65536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    chunk = b""
                if not chunk:
                    ended = True
                    break
                output += chunk
                if not sent and b"LiteLLM API key: " in output:
                    os.write(terminal, f"{secret}\n".encode())
                    sent = True
        finally:
            if not ended:
                os.killpg(child, signal.SIGKILL)
            os.close(terminal)
            _, status = os.waitpid(child, 0)
        curl: Final = cast(Mapping[str, object], json.loads((root / "curl.json").read_text()))  # cast-ok: own JSON
        record: Final = root / "npx.json"
        npx: Final[Mapping[str, object] | None] = json.loads(record.read_text()) if record.exists() else None
        curl_args: Final = cast(Sequence[str], curl["args"])  # cast-ok: fixture records argv
        installed: Final = root / "installed" / "SKILL.md"
        temporary: Final = Path(curl_args[curl_args.index("--output") + 1]).parent
        sys.stdout.write(json.dumps({
            "status": os.waitstatus_to_exitcode(status),
            "header": curl["header"],
            "installArgs": npx["args"] if npx else None,
            "telemetry": cast(Environment, npx["env"]).get("DISABLE_TELEMETRY") if npx else None,  # cast-ok: own env
            "secretLeaked": secret in json.dumps((output.decode(), curl_args, curl["env"], npx)),
            "retained": installed.read_text() if installed.exists() else None,
            "temporaryRemoved": not temporary.exists(),
        }))
        marker: Final = temporary / ".skill-install-fixture"
        if marker.exists() and marker.read_text() == str(root):
            shutil.rmtree(temporary)


if Path(sys.argv[0]).name in ("curl", "npx"):
    external_command(Path(sys.argv[0]).name)
else:
    run(sys.stdin.read(), sys.argv[1] == "fail")

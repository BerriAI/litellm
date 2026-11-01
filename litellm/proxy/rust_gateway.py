import asyncio
import os
import shutil
import signal
import socket
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter

GATEWAY_FD: Final = "_LITELLM_GATEWAY_FD"


def gateway_binary() -> str | None:
    name: Final = "litellm-gateway.exe" if os.name == "nt" else "litellm-gateway"
    return shutil.which(name, path=sysconfig.get_path("scripts")) or shutil.which(name)


def startup_error(binary: str | None, unsupported: Mapping[str, object]) -> str | None:
    flag: Final = next((name for name, value in unsupported.items() if value), None)
    if flag is not None:
        return f"--rust_gateway does not support {flag}"
    if os.name != "posix":
        return "--rust_gateway requires POSIX inherited sockets and process signals"
    if binary is None:
        return "--rust_gateway requires the litellm-gateway binary. Install litellm[proxy] in this environment"
    return None


def child_argv(argv: Sequence[str], port: int) -> tuple[str, ...]:
    return (*(arg for arg in argv if arg != "--rust_gateway"), "--host", "127.0.0.1", "--port", str(port))


def inherited_fd() -> int | None:
    value: Final = os.environ.get(GATEWAY_FD)
    return int(value) if value is not None else None


def metrics_host(host: str) -> str:
    return os.environ.get("_LITELLM_GATEWAY_METRICS_HOST", host) if inherited_fd() is not None else host


def terminate_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


async def _supervise(binary: str, command: Sequence[str], host: str, port: int) -> int:
    loop: Final = asyncio.get_running_loop()
    stopped: Final[asyncio.Future[signal.Signals]] = loop.create_future()

    def stop(signum: signal.Signals) -> None:
        if not stopped.done():
            stopped.set_result(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop, signum)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        _, upstream_port = TypeAdapter(tuple[str, int]).validate_python(listener.getsockname())
        python: Final = await asyncio.create_subprocess_exec(
            *child_argv(command, upstream_port),
            env={
                **os.environ,
                "LITELLM_RUST_GATEWAY": "false",
                GATEWAY_FD: str(listener.fileno()),
                "_LITELLM_GATEWAY_METRICS_HOST": host,
                "FORWARDED_ALLOW_IPS": "127.0.0.1",
            },
            pass_fds=(listener.fileno(),),
            start_new_session=True,
        )
        try:
            rust: Final = await asyncio.create_subprocess_exec(
                binary,
                env={
                    **os.environ,
                    "HOST": host,
                    "PORT": str(port),
                    "LITELLM_FORWARD_UPSTREAM": f"http://127.0.0.1:{upstream_port}",
                },
                start_new_session=True,
            )
        except OSError:
            terminate_group(python)
            await python.wait()
            raise
    python_done: Final = asyncio.create_task(python.wait())
    rust_done: Final = asyncio.create_task(rust.wait())
    try:
        done, _ = await asyncio.wait((stopped, python_done, rust_done), return_when=asyncio.FIRST_COMPLETED)
        if python_done in done or rust_done in done:
            return 1
        rust.send_signal(stopped.result())
        draining, _ = await asyncio.wait((python_done, rust_done), return_when=asyncio.FIRST_COMPLETED)
        if python_done in draining or rust_done.result() != 0:
            return 1
        python.send_signal(stopped.result())
        return 0 if await python_done == 0 else 1
    finally:
        terminate_group(rust)
        terminate_group(python)
        await asyncio.gather(python_done, rust_done)


def supervise(binary: str, command: Sequence[str], host: str, port: int) -> int:
    try:
        return asyncio.run(_supervise(binary, command, host, port))
    except OSError as error:
        sys.stderr.write(f"Rust gateway startup failed: {error}\n")
        return 1


def cli_command(argv: Sequence[str]) -> tuple[str, ...]:
    return (sys.executable, "-m", "litellm.proxy.proxy_cli", *argv)

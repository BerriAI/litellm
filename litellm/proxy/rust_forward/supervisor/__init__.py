import asyncio
import sys
import sysconfig
from collections.abc import Callable, Sequence
from typing import Final

from litellm.proxy.rust_forward.platform import Platform, current_platform
from litellm.proxy.rust_forward.server import PythonServer
from litellm.proxy.rust_forward.supervisor.lifecycle import FAILURE
from litellm.proxy.rust_forward.supervisor.preflight import StartupError, gateway_binary, startup_check
from litellm.proxy.rust_forward.supervisor.runtime import exit_like, run


def _nothing_to_prepare() -> None:
    pass


def supervise(
    server: PythonServer,
    command: Sequence[str],
    before_start: Callable[[], None] = _nothing_to_prepare,
    scripts_dir: str = sysconfig.get_path("scripts"),
    platform: Platform | None = None,
) -> int | StartupError:
    """Serve ``server``'s public port from Rust and run ``command`` behind it, or say why that cannot start.

    A refusal is returned for the caller to serve from Python alone, and ``before_start`` runs only once forwarding
    is known to be possible, so a refused start leaves nothing behind.
    """
    chosen: Final = current_platform() if platform is None else platform
    checked: Final = startup_check(chosen, server, gateway_binary(chosen, scripts_dir))
    if isinstance(checked, StartupError):
        sys.stderr.write(f"{checked.reason}, so Python serves the public port without Rust forwarding\n")
        return checked
    before_start()
    try:
        status: Final = asyncio.run(run(chosen, checked, command, server))
    except OSError as error:
        sys.stderr.write(f"Rust gateway startup failed: {error}\n")
        return FAILURE
    return exit_like(status)

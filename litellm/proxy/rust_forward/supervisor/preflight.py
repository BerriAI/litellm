import os
import shutil
from dataclasses import dataclass

from litellm.proxy.rust_forward.mode import SWITCH_ENV
from litellm.proxy.rust_forward.platform import Platform
from litellm.proxy.rust_forward.server import Listener, PythonServer, Server


@dataclass(frozen=True, slots=True)
class StartupError:
    reason: str


def gateway_binary(platform: Platform, scripts_dir: str) -> str | None:
    return shutil.which(os.path.join(scripts_dir, platform.gateway_name))


def startup_check(platform: Platform, server: PythonServer, binary: str | None) -> str | StartupError:
    if server.server in (Server.HYPERCORN, Server.GRANIAN):
        return StartupError(f"{SWITCH_ENV} does not support {server.server.value}")
    if server.server not in platform.servers:
        return StartupError(f"{SWITCH_ENV} does not support {server.server.value} on {platform.name}")
    if server.listener is not Listener.TCP:
        return StartupError(f"{SWITCH_ENV} does not support {server.listener.value} on the public port")
    if binary is None:
        return StartupError(f"{SWITCH_ENV} requires the litellm-gateway binary from the litellm-proxy package")
    return binary

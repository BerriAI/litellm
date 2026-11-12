from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from litellm.constants import GRACEFUL_SHUTDOWN_TEARDOWN_HEADROOM_SECONDS
from litellm.proxy.rust_forward.mode import RUST_HOP, Supervised, supervised_environment
from litellm.proxy.rust_forward.platform import Platform, PlatformListener, SpawnOptions
from litellm.proxy.rust_forward.server import PythonServer


@dataclass(frozen=True, slots=True)
class ChildSpec:
    argv: tuple[str, ...]
    env_overrides: Mapping[str, str]
    spawn: SpawnOptions


def python_spec(
    platform: Platform, command: Sequence[str], private: PlatformListener, server: PythonServer
) -> ChildSpec:
    return ChildSpec(
        argv=(*command, "--port", str(private.port)),
        env_overrides=supervised_environment(Supervised(private.listener, server.trusted_proxies)),
        spawn=platform.spawn_options(private.pass_fds),
    )


def rust_spec(platform: Platform, binary: str, upstream_port: int, server: PythonServer) -> ChildSpec:
    return ChildSpec(
        argv=(binary,),
        env_overrides=MappingProxyType(
            {
                "LITELLM_FORWARD_UPSTREAM": f"http://{RUST_HOP}:{upstream_port}",
                "LITELLM_FORWARD_HOST": server.host,
                "LITELLM_FORWARD_PORT": str(server.port),
                "LITELLM_FORWARD_UPSTREAM_KEEPALIVE": str(server.keepalive_seconds),
                "LITELLM_FORWARD_SHUTDOWN_DRAIN": str(
                    server.graceful_timeout_seconds + GRACEFUL_SHUTDOWN_TEARDOWN_HEADROOM_SECONDS
                ),
            }
        ),
        spawn=platform.spawn_options(()),
    )

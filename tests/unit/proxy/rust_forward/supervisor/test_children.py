import socket
from typing import Final

from litellm.constants import GRACEFUL_SHUTDOWN_TEARDOWN_HEADROOM_SECONDS
from litellm.proxy.proxy_cli import run_server
from litellm.proxy.rust_forward.mode import Supervised, rust_forward_mode
from litellm.proxy.rust_forward.platform import PlatformListener
from litellm.proxy.rust_forward.platform.posix import PosixPlatform
from litellm.proxy.rust_forward.platform.windows import CREATE_NEW_PROCESS_GROUP, WindowsPlatform
from litellm.proxy.rust_forward.supervisor.children import python_spec, rust_spec
from litellm.proxy.server_listener import AssignedPort, InheritedFd
from tests.unit.proxy.rust_forward.supervisor._support import python_server

POSIX: Final = PosixPlatform()
WINDOWS: Final = WindowsPlatform()


def test_python_child_serves_the_private_socket_with_the_operators_trust() -> None:
    private: Final = PlatformListener(InheritedFd(17), 45555, (17,))
    spec: Final = python_spec(POSIX, ("python",), private, python_server(trusted_proxies="10.0.0.5"))
    assert rust_forward_mode(spec.env_overrides) == Supervised(InheritedFd(17), "10.0.0.5")
    assert (spec.spawn.pass_fds, spec.spawn.start_new_session) == ((17,), True)


def test_windows_python_child_binds_the_assigned_port_in_its_own_process_group() -> None:
    with WINDOWS.private_listener() as private:
        pass
    spec: Final = python_spec(WINDOWS, ("python",), private, python_server())
    assert rust_forward_mode(spec.env_overrides) == Supervised(AssignedPort(private.port), "127.0.0.1")
    assert spec.argv[-2:] == ("--port", str(private.port))
    assert (spec.spawn.pass_fds, spec.spawn.creationflags) == ((), CREATE_NEW_PROCESS_GROUP)
    with socket.create_server(("127.0.0.1", private.port)):
        pass


def test_python_child_keeps_the_public_host_and_its_options() -> None:
    arguments: Final = ("--host", "0.0.0.0", "--port", "4000", "--num_workers", "2", "--reload")
    spec: Final = python_spec(POSIX, arguments, PlatformListener(InheritedFd(17), 45555, (17,)), python_server())
    params: Final = run_server.make_context("litellm", list(spec.argv)).params
    assert (params["host"], params["port"]) == ("0.0.0.0", 45555)
    assert (params["num_workers"], params["reload"]) == (2, True)


def test_rust_child_serves_public_port_and_forwards_to_private_listener() -> None:
    spec: Final = rust_spec(POSIX, "gateway", 45555, python_server())
    assert spec.argv == ("gateway",)
    assert spec.spawn.pass_fds == ()
    assert dict(spec.env_overrides) == {
        "LITELLM_FORWARD_UPSTREAM": "http://127.0.0.1:45555",
        "LITELLM_FORWARD_HOST": "0.0.0.0",
        "LITELLM_FORWARD_PORT": "4000",
        "LITELLM_FORWARD_UPSTREAM_KEEPALIVE": "2",
        "LITELLM_FORWARD_SHUTDOWN_DRAIN": str(2.5 + GRACEFUL_SHUTDOWN_TEARDOWN_HEADROOM_SECONDS),
    }

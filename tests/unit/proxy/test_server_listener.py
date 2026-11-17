import os
from typing import Final

import pytest

from litellm.proxy.server_listener import AssignedPort, InheritedFd, PrivateListener, gunicorn_overrides


def test_gunicorn_keeps_its_own_listener_and_trust_without_a_private_listener() -> None:
    assert dict(gunicorn_overrides(None)) == {}


@pytest.mark.skipif(os.name == "nt", reason="gunicorn does not run on Windows")
@pytest.mark.parametrize(
    ("listener", "address"),
    [
        (InheritedFd(17), 17),
        (AssignedPort(45556), ("127.0.0.1", 45556)),
    ],
)
def test_gunicorn_serves_a_private_listener_and_trusts_no_peer_to_name_the_client(
    listener: PrivateListener, address: int | tuple[str, int]
) -> None:
    gunicorn_config: Final = pytest.importorskip("gunicorn.config").Config()
    gunicorn_config.set("forwarded_allow_ips", "*")
    for name, value in gunicorn_overrides(listener).items():
        gunicorn_config.set(name, value)
    assert (gunicorn_config.address, gunicorn_config.forwarded_allow_ips) == ([address], [])

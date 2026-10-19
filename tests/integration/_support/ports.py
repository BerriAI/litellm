import fcntl
import os
import socket
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Final

FIRST_PORT: Final = 10_000
LAST_PORT: Final = 32_767
BLOCK_SIZE: Final = 200
LOCK_DIRECTORY: Final = Path(tempfile.gettempdir()) / "litellm-integration-ports"


def _blocks() -> Iterator[range]:
    return (range(start, start + BLOCK_SIZE) for start in range(FIRST_PORT, LAST_PORT + 2 - BLOCK_SIZE, BLOCK_SIZE))


def _claim_until_exit(block: range, directory: Path) -> bool:
    descriptor: Final = os.open(directory / f"{block.start}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return False
    return True


def _bindable(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _free_ports(directory: Path) -> Iterator[int]:
    directory.mkdir(parents=True, exist_ok=True)
    for block in _blocks():
        if _claim_until_exit(block, directory):
            yield from filter(_bindable, block)


class PortAllocator:
    def __init__(self, directory: Path) -> None:
        self._directory: Final = directory
        self._guard: Final = threading.Lock()
        self._ports: Final = _free_ports(directory)

    def allocate(self) -> int:
        with self._guard:
            port: Final = next(self._ports, None)
        assert port is not None, (
            f"Every integration port block in {FIRST_PORT}-{LAST_PORT} is claimed (locks in {self._directory})"
        )
        return port


allocate_port: Final = PortAllocator(LOCK_DIRECTORY).allocate

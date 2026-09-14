import os
import sys
from typing import Final

import psutil


def owned_processes(identity: str) -> tuple[psutil.Process, ...]:
    owned: Final[list[psutil.Process]] = []
    for process in psutil.process_iter(["uids"]):
        if process.info["uids"].real != os.getuid():
            continue
        try:
            if process.environ().get("INTEGRATION_RUN_ID") == identity:
                owned.append(process)
        except psutil.NoSuchProcess:
            continue
    return tuple(owned)


def main(identity: str) -> int:
    owned: Final = owned_processes(identity)
    for process in owned:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(owned, timeout=8)
    remaining: Final = owned_processes(identity)
    for process in remaining:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(remaining, timeout=2)
    survivors: Final = owned_processes(identity)
    print(f"Owned integration processes: {len(owned)}, forced: {len(remaining)}, remaining: {len(survivors)}")
    return 1 if remaining or survivors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

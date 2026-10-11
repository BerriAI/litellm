import sys
from typing import Final

import psutil


def is_owned(process: psutil.Process, identity: str, owner_uid: int) -> bool:
    try:
        return process.uids().real == owner_uid and process.environ().get("INTEGRATION_RUN_ID") == identity
    except psutil.NoSuchProcess:
        return False


def owned_processes(identity: str, owner_uid: int) -> tuple[psutil.Process, ...]:
    return tuple(process for process in psutil.process_iter() if is_owned(process, identity, owner_uid))


def main(identity: str, owner_uid: int, root_pids: tuple[int, ...]) -> int:
    assert owner_uid > 0, "The integration process owner must be a non-root UID"
    owned: Final = owned_processes(identity, owner_uid)
    roots: Final = tuple(process for process in owned if process.pid in root_pids)
    for process in roots:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(roots, timeout=30)
    residual: Final = owned_processes(identity, owner_uid)
    for process in residual:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(residual, timeout=10)
    remaining: Final = owned_processes(identity, owner_uid)
    for process in remaining:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(remaining, timeout=2)
    survivors: Final = owned_processes(identity, owner_uid)
    print(
        f"Owned integration processes: {len(owned)}, roots: {len(roots)}, "
        f"residual: {len(residual)}, forced: {len(remaining)}, remaining: {len(survivors)}"
    )
    for process in remaining:
        print(f"Forced cleanup was required for PID {process.pid}")
    return 1 if remaining or survivors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]), tuple(int(value) for value in sys.argv[3:] if value)))

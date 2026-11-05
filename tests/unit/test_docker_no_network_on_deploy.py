from pathlib import Path
from typing import Final

_NETWORK_FETCH_TOKENS: Final = ("curl", "wget", "fetch", "http://", "https://")


def test_container_build_no_network_fetch() -> None:
    dockerfile: Final = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text()
    startup_lines: Final = tuple(
        line.strip() for line in dockerfile.splitlines() if line.strip().upper().startswith(("CMD", "ENTRYPOINT"))
    )
    fetching_lines: Final = tuple(
        line for line in startup_lines if any(token in line.lower() for token in _NETWORK_FETCH_TOKENS)
    )

    assert startup_lines != ()
    assert fetching_lines == ()

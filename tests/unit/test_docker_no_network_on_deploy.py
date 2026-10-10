from pathlib import Path
from typing import Final


def test_container_build_no_network_fetch() -> None:
    dockerfile: Final = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text()
    problematic: Final = tuple(
        f"Line {index}: {line.strip()}"
        for index, line in enumerate(dockerfile.splitlines(), 1)
        if line.strip().upper().startswith(("CMD", "ENTRYPOINT"))
        and any(
            token in line.lower()
            for token in ("curl", "wget", "fetch", "http://", "https://")
        )
    )

    assert not problematic

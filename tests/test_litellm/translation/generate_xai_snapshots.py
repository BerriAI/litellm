"""Regenerate the characterization_xai snapshots from v1 IN-PROCESS at HEAD.

Run: LITELLM_LOCAL_MODEL_COST_MAP=True \
    python -m tests.test_litellm.translation.generate_xai_snapshots

Provenance: there are no recorded xai vendor fixtures anywhere (the
characterization branch has zero), so the snapshots pin v1-as-executed —
the primary differential reference. The drift gate in the xai differential
tests re-runs the same invokers and fails if v1 at HEAD ever stops matching
the committed snapshots; regenerating is a reviewed snapshot-diff, never a
silent step.

Ambient freeze mirrors tests' ``frozen_ambient``: stream chunks stamp
``created`` from ``time.time`` and the wrapper mints fastuuid ids.
"""

import itertools
import pathlib
import sys
import time
import uuid


def _freeze_ambient() -> None:
    import fastuuid

    import litellm._uuid

    counter = itertools.count(1)

    def fake_uuid4():
        return uuid.UUID(int=next(counter))

    uuid.uuid4 = fake_uuid4  # type: ignore[assignment]
    fastuuid.uuid4 = fake_uuid4
    litellm._uuid.uuid4 = fake_uuid4
    time.time = lambda: FROZEN_TIME  # type: ignore[assignment]


def _write(path: pathlib.Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    from tests.test_litellm.translation._xai_corpus import (
        FROZEN_TIME,
        SNAPSHOTS_DIR,
        canonical_json,
        corpus,
        run_v1_request_transform,
        run_v1_response_transform,
        replay_xai_sse_lines,
    )

    _freeze_ambient()
    for name, case in corpus("cases").items():
        _write(
            SNAPSHOTS_DIR / "requests" / f"{name}.json",
            canonical_json(run_v1_request_transform(case)),
        )
    for name, row in corpus("responses").items():
        _write(
            SNAPSHOTS_DIR / "responses" / f"{name}.json",
            canonical_json(run_v1_response_transform(row["body"], row["model"])),
        )
    for name, row in corpus("streams").items():
        _write(
            SNAPSHOTS_DIR / "streams" / f"{name}.json",
            canonical_json(replay_xai_sse_lines(row["events"], row["stream_options"])),
        )

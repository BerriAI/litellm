"""Pytest plumbing for the translation characterization corpus.

Determinism: every test runs with ``uuid.uuid4`` and ``time.time`` frozen
(see ``deterministic_ids``) because v1 injects generated ids
(``chatcmpl-<uuid>``, streaming response ids) and ``created`` timestamps
into ``ModelResponse``/``ModelResponseStream``.

Snapshot refresh: ``pytest tests/translation_characterization --snapshot-update``
or ``SNAPSHOT_UPDATE=1 pytest tests/translation_characterization``.
"""

import itertools
import os
import time
import uuid

import pytest

# Must be set before litellm is imported anywhere: keeps the model cost map
# local (no network) and gives the anthropic config a key for header building.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-char-test-key")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "AKIACHARTESTKEY00000")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "char-test-secret")
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")

FROZEN_TIME = 1718064000.0  # 2024-06-11T00:00:00Z


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Rewrite characterization snapshots (and generated corpus cases).",
    )


@pytest.fixture()
def snapshot_update(request: pytest.FixtureRequest) -> bool:
    try:
        flag = bool(request.config.getoption("--snapshot-update"))
    except ValueError:  # conftest not loaded as an initial conftest
        flag = False
    return flag or os.environ.get("SNAPSHOT_UPDATE", "") not in ("", "0", "false")


@pytest.fixture(autouse=True)
def deterministic_ids(monkeypatch: pytest.MonkeyPatch):
    """Freeze the two ambient generators v1 reaches for when minting ids.

    - ``uuid.uuid4`` -> a per-test deterministic sequence (UUID(int=1), ...)
    - ``time.time``  -> FROZEN_TIME (feeds ``ModelResponse.created``)
    """
    counter = itertools.count(1)

    def fake_uuid4():
        return uuid.UUID(int=next(counter))

    monkeypatch.setattr(uuid, "uuid4", fake_uuid4)
    import fastuuid  # litellm._uuid aliases this module as `uuid`

    import litellm._uuid

    monkeypatch.setattr(fastuuid, "uuid4", fake_uuid4)
    monkeypatch.setattr(litellm._uuid, "uuid4", fake_uuid4)
    monkeypatch.setattr(time, "time", lambda: FROZEN_TIME)
    yield

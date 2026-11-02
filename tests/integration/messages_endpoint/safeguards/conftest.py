from __future__ import annotations

from collections.abc import Iterator

import pytest
from integration._support.safeguards_rig import Rig, safeguards_rig


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with safeguards_rig(tmp_path_factory.mktemp("safeguards")) as value:
        yield value

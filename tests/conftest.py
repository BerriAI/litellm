from collections.abc import Iterator
from typing import Final

import pytest
import respx

from tests._support.respx_httpx2 import HTTPCore2AndLegacyMocker


@pytest.fixture
def respx_mock(request: pytest.FixtureRequest) -> Iterator[respx.MockRouter]:
    marker: Final = request.node.get_closest_marker("respx")
    options: Final = marker.kwargs if marker is not None else {}
    with respx.mock(using=HTTPCore2AndLegacyMocker.name, **options) as router:
        yield router

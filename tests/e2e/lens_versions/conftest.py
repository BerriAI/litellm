import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from e2e_http import unwrap
from models import UiLoginBody, UiLoginResponse

from .stack import UI_PASSWORD, UI_USERNAME, Stack, stack

OLDER_LENS: Final = "ghcr.io/berriai/lens:0.2.0"


def _output(release: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured: Final = os.environ.get("LENS_VERSION_TEST_OUTPUT")
    return Path(configured) / release if configured else tmp_path_factory.mktemp(release)


@pytest.fixture(scope="session")
def older(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    with stack(os.environ.get("LENS_VERSION_TEST_OLDER_IMAGE", OLDER_LENS), _output("older", tmp_path_factory)) as it:
        yield it


@pytest.fixture(scope="session")
def newest(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    image: Final = os.environ.get("LENS_VERSION_TEST_NEWEST_IMAGE")
    assert image, "LENS_VERSION_TEST_NEWEST_IMAGE must name the newest Lens image"
    with stack(image, _output("newest", tmp_path_factory)) as it:
        yield it


@pytest.fixture
def releases(older: Stack, newest: Stack) -> Mapping[str, Stack]:
    return MappingProxyType({"older": older, "newest": newest})


def signed_in(stack: Stack) -> str:
    return unwrap(
        stack.gateway.post(
            "/v2/login",
            headers=stack.gateway.master,
            json=UiLoginBody(username=UI_USERNAME, password=UI_PASSWORD),
            response_type=UiLoginResponse,
        )
    ).token

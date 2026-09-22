from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import httpx
import httpx2
import pytest

from tests._vcr_conftest_common import (
    detect_vcr_patch_leak,
    guard_vcr_patch_points,
    restore_vcr_patch_points,
    rewound_new_episodes_cassette,
)

_ORIGINAL_MOCK_HANDLE_ASYNC_REQUEST: Final = httpx.MockTransport.handle_async_request
_ORIGINAL_HTTPX2_MOCK_HANDLE_ASYNC_REQUEST: Final = httpx2.MockTransport.handle_async_request


@pytest.fixture
def leaked_cassette_dir(tmp_path: Path):
    context: Final = rewound_new_episodes_cassette(tmp_path)
    context.__enter__()
    yield tmp_path
    context.__exit__(None, None, None)


def test_no_leak_when_no_cassette_is_active():
    assert detect_vcr_patch_leak() is None


def test_leaked_cassette_is_detected_named_and_restorable(leaked_cassette_dir: Path):
    leak: Final = detect_vcr_patch_leak()

    assert leak is not None
    assert {"httpx.MockTransport.handle_async_request", "aiohttp.client.ClientSession._request"} <= set(
        leak.patch_points
    )
    assert leak.cassette_paths == (str(leaked_cassette_dir / "rewound_owner.yaml"),)

    restore_vcr_patch_points()

    assert detect_vcr_patch_leak() is None
    assert httpx.MockTransport.handle_async_request is _ORIGINAL_MOCK_HANDLE_ASYNC_REQUEST


def test_leak_is_detected_on_every_transport_family_vcrpy_patches(leaked_cassette_dir: Path):
    leak: Final = detect_vcr_patch_leak()

    assert leak is not None
    assert "httpx2.MockTransport.handle_async_request" in leak.patch_points
    assert httpx2.MockTransport.handle_async_request is not _ORIGINAL_HTTPX2_MOCK_HANDLE_ASYNC_REQUEST

    restore_vcr_patch_points()

    assert httpx2.MockTransport.handle_async_request is _ORIGINAL_HTTPX2_MOCK_HANDLE_ASYNC_REQUEST


def test_guard_fails_the_leaking_test_and_restores_the_originals(request, leaked_cassette_dir: Path):
    with pytest.raises(pytest.fail.Exception, match=re.escape(request.node.nodeid)) as failure:
        guard_vcr_patch_points(request.node, teardown_failed=False)

    assert str(leaked_cassette_dir / "rewound_owner.yaml") in str(failure.value)
    assert detect_vcr_patch_leak() is None


def test_guard_restores_silently_when_the_teardown_already_failed(request, leaked_cassette_dir: Path):
    guard_vcr_patch_points(request.node, teardown_failed=True)

    assert detect_vcr_patch_leak() is None

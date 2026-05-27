"""Regression tests for LIT-3386 / GH #28294 (Point 72).

The list_files endpoint runs proxy_logging_obj.post_call_success_hook after the
upstream provider call. UnifiedFileIdHook (managed files) returns an
openai.pagination.AsyncCursorPage for the list-files response shape (filtered
to the caller's owned files).

Previously the endpoint guarded reassignment with
isinstance(_response, OpenAIFileObject), which is always False for an
AsyncCursorPage, so the hook return was silently discarded. The in-place
mutation of response.data inside the hook masked the bug in production paths
but the type check itself was incorrect and would fail for any future hook
that returns a fresh AsyncCursorPage instance.

This regression test pins the broadened type check
(OpenAIFileObject, AsyncCursorPage).
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from openai.pagination import AsyncCursorPage
from openai.types.file_object import FileObject


def _file(file_id):
    return FileObject(
        id=file_id,
        object="file",
        bytes=10,
        created_at=0,
        filename="x.jsonl",
        purpose="batch",
        status="processed",
    )


@pytest.fixture
def client():
    # Import inside the fixture so we resolve the live proxy_server module after
    # tests/test_litellm/conftest.py::setup_and_teardown has reloaded it.
    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import (
        user_api_key_auth as user_api_key_auth_dep,
    )

    def _override_auth():
        return UserAPIKeyAuth(user_id="user-42", api_key="sk-test", token="sk-test")

    proxy_server.app.dependency_overrides[user_api_key_auth_dep] = _override_auth
    yield TestClient(proxy_server.app)
    proxy_server.app.dependency_overrides.pop(user_api_key_auth_dep, None)


@pytest.fixture
def unfiltered_page():
    return AsyncCursorPage(
        data=[
            _file("file-raw-input-aaa"),
            _file("file-raw-output-bbb"),
            _file("file-raw-other-ccc"),
        ]
    )


def _patch_provider(unfiltered_page):
    import litellm

    async def _fake(*args, **kwargs):
        return unfiltered_page

    return patch.object(litellm, "afile_list", side_effect=_fake)


def _patch_hook(side_effect):
    # Patch the live module attribute, not a stale imported reference -
    # conftest.py reloads litellm.proxy.proxy_server at module scope, which
    # replaces proxy_logging_obj.
    from litellm.proxy import proxy_server

    return patch.object(
        proxy_server.proxy_logging_obj,
        "post_call_success_hook",
        side_effect=side_effect,
    )


def test_list_files_honors_async_cursor_page_returned_by_hook(client, unfiltered_page):
    """LIT-3386: list_files must honor AsyncCursorPage returned by the post-call hook.

    The UnifiedFileIdHook (managed files) returns a filtered AsyncCursorPage for
    file-list responses. Previously the endpoint's isinstance check only allowed
    OpenAIFileObject, silently discarding the hook return. With Point 72's
    secondary bug fixed, the endpoint reassigns response to the hook return.
    """
    filtered_page = AsyncCursorPage(data=[_file("litellm_proxy:managed-aaa")])

    async def _hook(*, data, user_api_key_dict, response):
        return filtered_page

    with _patch_provider(unfiltered_page), _patch_hook(_hook):
        r = client.get("/v1/files?purpose=batch")

    assert r.status_code == 200, r.text
    ids = [f["id"] for f in r.json()["data"]]
    assert ids == ["litellm_proxy:managed-aaa"], (
        f"list_files dropped the hook return value; got {ids}. The endpoint "
        "type check must include AsyncCursorPage."
    )


def test_list_files_passes_through_unchanged_when_hook_returns_none(
    client, unfiltered_page
):
    """Hook returns None - endpoint must keep the upstream response unchanged."""

    async def _hook(*, data, user_api_key_dict, response):
        return None

    with _patch_provider(unfiltered_page), _patch_hook(_hook):
        r = client.get("/v1/files?purpose=batch")

    assert r.status_code == 200, r.text
    ids = [f["id"] for f in r.json()["data"]]
    assert ids == [
        "file-raw-input-aaa",
        "file-raw-output-bbb",
        "file-raw-other-ccc",
    ]

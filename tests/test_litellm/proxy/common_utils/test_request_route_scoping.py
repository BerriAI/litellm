import pytest

from litellm.proxy.common_utils.request_route_scoping import (
    is_file_upload_route,
    strip_root_path,
)


@pytest.mark.parametrize(
    "path, root_path, expected",
    [
        ("/v1/files", "", "/v1/files"),
        ("/v1/files", "/", "/v1/files"),
        ("/genai/v1/files", "/genai", "/v1/files"),
        ("/genai/v1/files", "/genai/", "/v1/files"),
        ("/api/genai/v1/files", "/api/genai", "/v1/files"),
        ("/api", "/api", "/"),
        # "/apifoo" is not a segment match for "/api", so it survives intact
        # rather than being truncated to "foo".
        ("/apifoo/v1/files", "/api", "/apifoo/v1/files"),
        ("/v1/files", "/v", "/v1/files"),
    ],
)
def test_strip_root_path(path, root_path, expected):
    assert strip_root_path(path=path, root_path=root_path) == expected


@pytest.mark.parametrize(
    "route",
    [
        "/files",
        "/v1/files",
        "/openai/v1/files",
        "/azure/v1/files",
    ],
)
def test_file_upload_routes_are_matched(route):
    assert is_file_upload_route(method="POST", route=route) is True


@pytest.mark.parametrize(
    "route",
    [
        "/chat/completions",
        "/v1/files/file-abc123",
        "/v1/files/file-abc123/content",
        "/v1/vector_stores/vs_1/files",
        "/genai/openai/v1/files",
        "/v1/audio/transcriptions",
        "/fine_tuning/jobs",
    ],
)
def test_non_upload_routes_are_not_matched(route):
    assert is_file_upload_route(method="POST", route=route) is False


@pytest.mark.parametrize("method", ["GET", "DELETE", "PUT", ""])
def test_only_post_is_an_upload(method):
    assert is_file_upload_route(method=method, route="/v1/files") is False


def test_method_casing_is_normalized():
    assert is_file_upload_route(method="post", route="/v1/files") is True

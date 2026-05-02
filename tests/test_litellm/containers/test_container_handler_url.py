import pytest

from litellm.llms.custom_httpx.container_handler import _build_url


def test_build_url_encodes_path_params_and_preserves_query():
    url = _build_url(
        api_base="https://example.com/v1/containers?api-version=v1",
        path_template="/containers/{container_id}/files/{file_id}/content",
        path_params={
            "container_id": "../../containers/other",
            "file_id": "file?download=1#frag",
        },
    )

    assert (
        url
        == "https://example.com/v1/containers/..%2F..%2Fcontainers%2Fother/files/file%3Fdownload%3D1%23frag/content?api-version=v1"
    )


def test_build_url_rejects_dot_segment_path_param():
    with pytest.raises(ValueError, match="container_id cannot be a dot path segment"):
        _build_url(
            api_base="https://example.com/v1/containers",
            path_template="/containers/{container_id}",
            path_params={"container_id": ".."},
        )

import socket

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy.vertex_ai_endpoints.langfuse_endpoints import (
    _build_langfuse_proxy_target,
    _get_langfuse_proxy_credentials,
)


def test_dynamic_langfuse_host_requires_dynamic_credentials(monkeypatch):
    monkeypatch.setattr(litellm, "user_url_validation", True, raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "global-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "global-secret")

    with pytest.raises(HTTPException) as exc:
        _get_langfuse_proxy_credentials(
            dynamic_host_supplied=True,
            dynamic_langfuse_public_key=None,
            dynamic_langfuse_secret_key=None,
        )

    assert exc.value.status_code == 400


def test_global_langfuse_host_can_use_env_credentials(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "global-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "global-secret")

    public_key, secret_key = _get_langfuse_proxy_credentials(
        dynamic_host_supplied=False,
        dynamic_langfuse_public_key=None,
        dynamic_langfuse_secret_key=None,
    )

    assert public_key == "global-public"
    assert secret_key == "global-secret"


@pytest.mark.parametrize(
    "endpoint",
    [
        "../api/public/projects",
        "%2e%2e/api/public/projects",
        "%252e%252e%252fapi/public/projects",
        "api\\public\\projects",
        "%2f%2fattacker.example/api",
    ],
)
def test_langfuse_proxy_target_rejects_traversal_paths(endpoint):
    with pytest.raises(HTTPException) as exc:
        _build_langfuse_proxy_target(
            endpoint=endpoint,
            base_target_url="https://cloud.langfuse.com",
            dynamic_host_supplied=False,
        )

    assert exc.value.status_code == 400


def test_dynamic_langfuse_proxy_target_rejects_internal_host(monkeypatch):
    monkeypatch.setattr(litellm, "user_url_validation", True, raising=False)

    with pytest.raises(HTTPException) as exc:
        _build_langfuse_proxy_target(
            endpoint="api/public/projects",
            base_target_url="http://127.0.0.1:3000",
            dynamic_host_supplied=True,
        )

    assert exc.value.status_code == 400


def test_dynamic_langfuse_proxy_target_preserves_host_header_for_http(monkeypatch):
    monkeypatch.setattr(litellm, "user_url_validation", True, raising=False)

    def fake_getaddrinfo(host, port, proto):
        assert host == "langfuse.example"
        assert port == 80
        assert proto == socket.IPPROTO_TCP
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("8.8.8.8", 80),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    target_url, headers = _build_langfuse_proxy_target(
        endpoint="api/public/projects",
        base_target_url="http://langfuse.example",
        dynamic_host_supplied=True,
    )

    assert target_url == "http://8.8.8.8/api/public/projects"
    assert headers["Host"] == "langfuse.example"

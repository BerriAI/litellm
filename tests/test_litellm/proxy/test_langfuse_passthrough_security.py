import socket
from base64 import b64encode

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy.vertex_ai_endpoints.langfuse_endpoints import (
    _build_langfuse_proxy_target,
    _extract_api_key_from_basic_auth,
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


@pytest.mark.parametrize(
    "authorization_header, expected",
    [
        # Every shape below except the two valid ones raised out of the route
        # before, turning an unauthenticated request into a 500 with a traceback.
        ("", ""),  # no Authorization header at all
        ("Basic ", ""),  # header present, credentials empty
        ("Basic " + b64encode(b"pk-lf-1:sk-lf-2").decode(), "sk-lf-2"),
        (b64encode(b"pk-lf-1:sk-lf-2").decode(), "sk-lf-2"),  # bare base64, no scheme
        ("Basic " + b64encode(b"sk-lf-2").decode(), ""),  # no ":" to split on
        ("Basic YWJj=", ""),  # not decodable base64
        ("Basic //4=", ""),  # decodes to bytes that are not utf-8
        ("Bearer sk-1234", ""),  # wrong scheme
    ],
)
def test_extract_api_key_from_basic_auth_never_raises(authorization_header, expected):
    assert _extract_api_key_from_basic_auth(authorization_header) == expected


def test_extract_api_key_keeps_a_secret_containing_a_colon():
    """RFC 7617 puts everything after the first ":" in the password, so a secret
    with a colon in it must survive whole. `split(":")[1]` truncated it to the
    first segment, which then failed authentication for a non-obvious reason."""
    header = "Basic " + b64encode(b"pk-lf-1:sk:with:colons").decode()

    assert _extract_api_key_from_basic_auth(header) == "sk:with:colons"


@pytest.mark.asyncio
async def test_langfuse_route_hands_missing_credentials_to_the_authenticator(monkeypatch):
    """An unauthenticated request must reach user_api_key_auth and be rejected
    there, rather than raising IndexError before authentication runs."""
    from fastapi import Request

    from litellm.proxy.vertex_ai_endpoints import langfuse_endpoints

    seen: dict = {}

    async def fake_user_api_key_auth(request, api_key):
        seen["api_key"] = api_key
        raise HTTPException(status_code=401, detail={"error": "Authentication Error"})

    monkeypatch.setattr(langfuse_endpoints, "user_api_key_auth", fake_user_api_key_auth)

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/langfuse/zzz",
            "headers": [],  # the reported case: no Authorization header
            "query_string": b"",
        }
    )

    with pytest.raises(HTTPException) as exc:
        await langfuse_endpoints.langfuse_proxy_route(endpoint="zzz", request=request, fastapi_response=None)

    assert exc.value.status_code == 401
    assert seen["api_key"] == "Bearer "

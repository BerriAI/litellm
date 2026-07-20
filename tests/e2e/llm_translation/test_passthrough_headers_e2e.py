"""Live e2e: custom pass-through endpoints inject configured headers and honor
x-pass-* client headers (prefix stripped) on the way to the upstream.

The upstream is a real public echo service (httpbin.org/anything). Creating the
route via POST /config/pass_through_endpoint, calling it with a virtual key, and
asserting the echo body is the product path operators use; a mock would not
prove the proxy actually rewrote the outbound request.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from e2e_config import unique_marker
from e2e_http import AuthHeaders, NoBody, StreamingResponse, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import KeyGenerateBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

ECHO_TARGET = "https://httpbin.org/anything"
STATIC_HEADER_NAME = "x-e2e-static-header"
PASS_HEADER_STEM = "e2e-client-marker"
PASS_HEADER_NAME = f"x-pass-{PASS_HEADER_STEM}"


class PassThroughCreateBody(BaseModel):
    path: str
    target: str
    headers: dict[str, str] = {}
    auth: bool = True
    include_subpath: bool = False


class PassThroughEndpoint(BaseModel):
    id: str | None = None
    path: str
    target: str


class PassThroughCreateResponse(BaseModel):
    endpoints: list[PassThroughEndpoint]


class PassThroughDeleteParams(BaseModel):
    endpoint_id: str


class EchoCallHeaders(AuthHeaders):
    content_type: str = Field(default="application/json", serialization_alias="Content-Type")
    x_pass_e2e_client_marker: str = Field(serialization_alias="x-pass-e2e-client-marker")


class EchoBody(BaseModel):
    ping: str


class EchoResponse(BaseModel):
    headers: dict[str, str]


def _create_passthrough(
    client: PassthroughClient, *, path: str, static_value: str
) -> PassThroughEndpoint:
    created = unwrap(
        client.proxy.transport.post(
            "/config/pass_through_endpoint",
            headers=client.proxy.transport.master,
            json=PassThroughCreateBody(
                path=path,
                target=ECHO_TARGET,
                headers={STATIC_HEADER_NAME: static_value},
            ),
            response_type=PassThroughCreateResponse,
        )
    )
    assert created.endpoints, "create returned no endpoints"
    endpoint = created.endpoints[0]
    assert endpoint.id, "created pass-through endpoint has no id"
    return endpoint


def _delete_passthrough(client: PassthroughClient, endpoint_id: str) -> None:
    _ = client.proxy.transport.delete(
        "/config/pass_through_endpoint",
        headers=client.proxy.transport.master,
        json=NoBody(),
        params=PassThroughDeleteParams(endpoint_id=endpoint_id),
        response_type=PassThroughCreateResponse,
    )


def _echo_headers(resp: StreamingResponse) -> dict[str, str]:
    try:
        echo = EchoResponse.model_validate_json(resp.body)
    except ValidationError as exc:
        pytest.fail(f"echo upstream did not return a headers map: {exc}; body={resp.body[:300]}")
    return {k.lower(): v for k, v in echo.headers.items()}


class TestPassthroughHeaders:
    @pytest.mark.covers(
        "other.config.passthrough.headers_forwarded",
        exercised_on=[],
    )
    def test_static_and_x_pass_headers_reach_upstream(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        path = f"/e2e-passthrough-headers-{marker}"
        static_value = f"static-{marker}"
        client_value = f"client-{marker}"

        endpoint = _create_passthrough(client, path=path, static_value=static_value)
        assert endpoint.id is not None
        resources.defer(lambda: _delete_passthrough(client, endpoint.id or ""))

        key = client.proxy.generate_key(
            KeyGenerateBody(
                models=[],
                allowed_passthrough_routes=[path],
                user_id=f"e2e-pass-headers-{marker}",
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        result = client.proxy.transport.send(
            path,
            headers=EchoCallHeaders(
                authorization=f"Bearer {key}",
                x_pass_e2e_client_marker=client_value,
            ),
            json=EchoBody(ping=marker),
        )
        require_successful_call(result)

        upstream = _echo_headers(result)
        assert upstream.get(STATIC_HEADER_NAME) == static_value, (
            f"configured pass-through header {STATIC_HEADER_NAME!r} not on upstream "
            f"request; got {upstream}"
        )
        assert upstream.get(PASS_HEADER_STEM) == client_value, (
            f"x-pass-* header should strip the prefix and forward as {PASS_HEADER_STEM!r}; "
            f"got {upstream}"
        )
        assert PASS_HEADER_NAME not in upstream, (
            "upstream must not see the x-pass- prefix; proxy should strip it"
        )

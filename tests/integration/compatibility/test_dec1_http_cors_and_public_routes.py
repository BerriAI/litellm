"""CORS, browser security headers, health probes and the public utility routes of the front door.

These are the answers that never reach an LLM handler: the CORS middleware's preflight reply, the three
anti-framing headers stamped on every response, the unauthenticated liveness and readiness probes that
Kubernetes and load balancers call, the Swagger page at `/`, `/openapi.json`, `/routes` and `/metrics`. They are
produced by Starlette middleware and tiny handlers today. A front door that replaces or sits before that stack
must return the same status codes, bodies and header values, and must keep the same routes public or protected.
The shared proxy runs with the default CORS settings: any origin, no credentials.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from integration.compatibility.dec1_http_support import CHAT, chat_body, front_door

ORIGIN: Final = "https://app.example"
SECURITY_HEADERS: Final = {
    "x-frame-options": "DENY",
    "content-security-policy": "frame-ancestors 'none'",
    "x-content-type-options": "nosniff",
}
PREFLIGHT: Final = {
    "Origin": ORIGIN,
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "authorization,content-type,x-stainless-os",
}
OPENAPI_SECONDS: Final = 180


def test_preflight_on_the_chat_path_is_answered_without_credentials(gateway: Gateway) -> None:
    """A browser sends the CORS preflight for a chat call: OPTIONS with `Origin`, `Access-Control-Request-Method:
    POST` and three requested headers, and no API key. The proxy must answer 200 with body `OK`,
    `access-control-allow-origin: *`, an `access-control-allow-methods` list that includes POST,
    `access-control-max-age: 600`, and `access-control-allow-headers` echoing the requested header list exactly."""
    response: Final = gateway.client.options(CHAT, headers=PREFLIGHT)
    assert response.status_code == 200, response.text
    assert response.text == "OK"
    assert response.headers["access-control-allow-origin"] == "*"
    assert "POST" in response.headers["access-control-allow-methods"].split(", ")
    assert response.headers["access-control-max-age"] == "600"
    assert response.headers["access-control-allow-headers"] == "authorization,content-type,x-stainless-os"


def test_preflight_does_not_grant_credentials_under_the_wildcard_origin(gateway: Gateway) -> None:
    """With no `LITELLM_CORS_ORIGINS` configured the proxy allows every origin, and for that case it deliberately
    turns credentials off (`_get_cors_config`, `proxy_server.py:2648`). A preflight must therefore carry no
    `access-control-allow-credentials` header, otherwise any site could make credentialed calls."""
    response: Final = gateway.client.options(CHAT, headers=PREFLIGHT)
    assert response.status_code == 200, response.text
    assert "access-control-allow-credentials" not in response.headers


def test_preflight_is_answered_for_a_path_no_route_serves(gateway: Gateway) -> None:
    """A browser preflights a path that does not exist. The CORS middleware answers before routing, so the reply
    must still be 200 `OK` with `access-control-allow-origin: *`. Only the real request that follows gets the 404."""
    response: Final = gateway.client.options(f"/dec1-no-such-route-{uuid.uuid4().hex}", headers=PREFLIGHT)
    assert response.status_code == 200, response.text
    assert response.headers["access-control-allow-origin"] == "*"


def test_a_wildcard_preflight_is_the_same_for_every_origin(gateway: Gateway) -> None:
    """Under the default wildcard CORS settings the preflight reply does not depend on the origin, which is why
    Starlette 1.3 (`CORSMiddleware`, the version uv.lock pins) only adds `Vary: Origin` when it echoes an explicit
    origin. Two preflights from different origins must both answer 200 with `access-control-allow-origin: *` and
    the same allow-methods and allow-headers, so a shared cache serving one origin's preflight to the other is
    harmless. Whether `vary` is present is not asserted, since newer Starlette releases add it here."""
    answers: Final = tuple(
        gateway.client.options(CHAT, headers={**PREFLIGHT, "Origin": origin})
        for origin in (ORIGIN, "https://other.example")
    )
    assert [response.status_code for response in answers] == [200, 200]
    seen: Final = {
        tuple(
            response.headers[name]
            for name in (
                "access-control-allow-origin",
                "access-control-allow-methods",
                "access-control-allow-headers",
            )
        )
        for response in answers
    }
    assert len(seen) == 1, seen
    assert next(iter(seen))[0] == "*"


def test_a_successful_chat_response_to_a_browser_origin_carries_the_cors_headers(gateway: Gateway) -> None:
    """A chat model is registered. The caller posts a chat request with an `Origin` header. The 200 response must
    carry `access-control-allow-origin: *` and an `access-control-expose-headers` list that includes
    `x-litellm-cache-key` and `x-litellm-applied-guardrails` (`LITELLM_UI_ALLOW_HEADERS`, `constants.py:308`),
    which is what lets browser code read those response headers."""
    with front_door(gateway) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()), headers={"Origin": ORIGIN})
        assert response.status_code == 200, response.text
        assert response.headers["access-control-allow-origin"] == "*"
        exposed: Final = [name.strip() for name in response.headers["access-control-expose-headers"].split(",")]
        assert {"x-litellm-cache-key", "x-litellm-applied-guardrails"} <= set(exposed)


def test_a_request_without_an_origin_gets_no_cors_headers(gateway: Gateway) -> None:
    """The caller calls the liveness probe with no `Origin` header, as every server-side client does. The
    response must carry no `access-control-allow-origin` header: the CORS middleware only decorates requests that
    declare an origin."""
    response: Final = gateway.client.get("/health/liveliness")
    assert response.status_code == 200, response.text
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(
    ("method", "path", "status"),
    (
        pytest.param("POST", CHAT, 401, id="unauthenticated-chat"),
        pytest.param("GET", "/dec1-no-such-route", 404, id="unknown-path"),
        pytest.param("GET", CHAT, 405, id="wrong-method"),
    ),
)
def test_error_responses_to_a_browser_origin_still_carry_the_cors_header(
    gateway: Gateway, method: str, path: str, status: int
) -> None:
    """The caller sends a request that fails (no key, unknown path, wrong method) with an `Origin` header. The
    error response must still carry `access-control-allow-origin: *`. Without it a browser hides the status and
    body from the page and reports an opaque network error instead of the real failure."""
    response: Final = gateway.client.request(method, path, headers={"Origin": ORIGIN})
    assert response.status_code == status, response.text
    assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.parametrize(
    ("method", "path", "status"),
    (
        pytest.param("GET", "/health/liveliness", 200, id="liveliness"),
        pytest.param("POST", CHAT, 401, id="unauthenticated-chat"),
        pytest.param("GET", "/dec1-no-such-route", 404, id="unknown-path"),
        pytest.param("GET", CHAT, 405, id="wrong-method"),
        pytest.param("GET", "/health/readiness", 200, id="readiness"),
    ),
)
def test_every_response_carries_the_three_anti_framing_headers(
    gateway: Gateway, method: str, path: str, status: int
) -> None:
    """Whatever the outcome, each response must carry `x-frame-options: DENY`, `content-security-policy:
    frame-ancestors 'none'` and `x-content-type-options: nosniff` (`security_headers_middleware.py:20`). These
    stop the admin UI being framed and are added by an outer middleware, so success and error paths agree."""
    response: Final = gateway.client.request(method, path)
    assert response.status_code == status, response.text
    assert {name: response.headers.get(name) for name in SECURITY_HEADERS} == SECURITY_HEADERS


def test_hsts_is_not_sent_unless_the_operator_opts_in(gateway: Gateway) -> None:
    """The shared proxy runs without `LITELLM_ENABLE_HSTS`. A response must not carry
    `strict-transport-security`, because the header would lock browsers out of plain-http deployments."""
    response: Final = gateway.client.get("/health/liveliness")
    assert "strict-transport-security" not in response.headers


def test_a_preflight_response_also_carries_the_anti_framing_headers(gateway: Gateway) -> None:
    """The security-headers middleware wraps the CORS middleware, so even the preflight reply that CORS produces
    on its own must carry the three anti-framing headers with their fixed values."""
    response: Final = gateway.client.options(CHAT, headers=PREFLIGHT)
    assert {name: response.headers.get(name) for name in SECURITY_HEADERS} == SECURITY_HEADERS


@pytest.mark.parametrize("path", ("/health/liveliness", "/health/liveness"))
def test_liveness_probe_answers_without_credentials(gateway: Gateway, path: str) -> None:
    """The caller GETs the liveness probe under both its historical name and the Kubernetes name, with no API key.
    Each must answer 200 with `content-type: application/json` and the JSON string `"I'm alive!"` as the whole
    body (`_health_endpoints.py:2451`). Probes compare this body in some deployments."""
    response: Final = gateway.client.get(path)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.content == b'"I\'m alive!"'


def test_readiness_probe_answers_without_credentials_and_reports_only_status_and_db(gateway: Gateway) -> None:
    """The caller GETs `/health/readiness` with no API key. With a healthy database the proxy must answer 200
    with exactly `{"status": "healthy", "db": "connected"}`. The public probe is deliberately low-detail: version,
    callbacks and cache type are only on the authenticated details route (`_health_endpoints.py:2326`)."""
    response: Final = gateway.client.get("/health/readiness")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "healthy", "db": "connected"}


def test_readiness_details_requires_a_key_and_then_reports_the_version(gateway: Gateway) -> None:
    """The caller GETs `/health/readiness/details` first without a key and then with the master key. The first
    must be 401 in the OpenAI error envelope with `type: auth_error`. The second must be 200 and include
    `status`, `db: connected` and a `litellm_version` field (`_health_endpoints.py:2160`), the detail the public
    probe withholds."""
    anonymous: Final = gateway.client.get("/health/readiness/details")
    assert anonymous.status_code == 401, anonymous.text
    assert object_value(JSON_OBJECT.validate_json(anonymous.content)["error"])["type"] == "auth_error"
    details: Final = gateway.get("/health/readiness/details")
    assert {"status", "db", "litellm_version"} <= set(details)
    assert details["db"] == "connected"


def test_the_full_health_check_route_requires_a_key(gateway: Gateway) -> None:
    """`/health` runs real model health checks, so unlike the probes it is protected. A GET without a key must
    answer 401 in the OpenAI error envelope with `code: "401"` and must not start any health check."""
    response: Final = gateway.client.get("/health")
    assert response.status_code == 401, response.text
    assert object_value(JSON_OBJECT.validate_json(response.content)["error"])["code"] == "401"


def test_the_drain_endpoint_is_a_404_unless_enabled(gateway: Gateway) -> None:
    """`/health/drain` permanently takes a worker out of rotation, so it is off unless
    `general_settings.enable_drain_endpoint` is set. On the shared proxy a GET must answer 404
    `{"detail": "Not Found"}` and the liveness probe must still answer 200 afterwards."""
    response: Final = gateway.client.get("/health/drain")
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Not Found"}
    assert gateway.client.get("/health/liveliness").status_code == 200


def test_the_root_path_serves_the_swagger_page_without_credentials(gateway: Gateway) -> None:
    """The caller GETs `/` with no API key. With the default `DOCS_URL` the Swagger UI is mounted at `/`, ahead of
    the authenticated `home` handler, so the proxy must answer 200 with an HTML page that references
    `openapi.json`. UNVERIFIED: route order read from `FastAPI(docs_url=get_docs_url())` at `proxy_server.py:2161`."""
    response: Final = gateway.client.get("/")
    assert response.status_code == 200, response.text[:300]
    assert response.headers["content-type"].startswith("text/html")
    assert "openapi.json" in response.text


def test_the_openapi_schema_is_public_and_lists_the_inference_routes(gateway: Gateway) -> None:
    """The caller GETs `/openapi.json` with no API key. The proxy must answer 200 with a JSON document that has an
    `openapi` version string, an `info.version`, and `paths` containing `/v1/chat/completions`, `/v1/embeddings`
    and `/v1/messages`. SDK generators and the Swagger page both read this document."""
    response: Final = gateway.client.get("/openapi.json", timeout=OPENAPI_SECONDS)
    assert response.status_code == 200, response.text[:300]
    schema: Final = JSON_OBJECT.validate_json(response.content)
    assert isinstance(schema["openapi"], str)
    assert isinstance(object_value(schema["info"])["version"], str)
    assert {"/v1/chat/completions", "/v1/embeddings", "/v1/messages"} <= set(object_value(schema["paths"]))


def test_the_openapi_schema_version_matches_the_version_response_header(gateway: Gateway) -> None:
    """A chat model is registered and called once. The `x-litellm-version` header on that response must equal
    `info.version` in `/openapi.json`: both are the installed `litellm` version (`proxy_server.py:1158`). A front
    door that stamps its own version header would break support workflows that read it."""
    with front_door(gateway) as door:
        response: Final = gateway.request("POST", CHAT, chat_body(door.chat_model()))
        assert response.status_code == 200, response.text
        schema: Final = JSON_OBJECT.validate_json(gateway.client.get("/openapi.json", timeout=OPENAPI_SECONDS).content)
        assert response.headers["x-litellm-version"] == object_value(schema["info"])["version"]


def test_the_routes_listing_is_public_and_lists_the_chat_route(gateway: Gateway) -> None:
    """The caller GETs `/routes` without a key. `/routes` is in `LiteLLMRoutes.public_routes`
    (`litellm/proxy/_types.py`), so the answer must be 200 with a `routes` list in which some entry has
    `path: "/v1/chat/completions"` and `POST` among its methods (`GetRoutes.get_app_routes`). A front door that
    starts demanding a key here breaks clients that discover routes before they have one."""
    response: Final = gateway.client.get("/routes")
    assert response.status_code == 200, response.text
    routes: Final = tuple(object_value(entry) for entry in list_value(object_value(response.json())["routes"]))
    chat: Final = tuple(entry for entry in routes if entry["path"] == CHAT)
    assert chat, "The chat route is missing from /routes"
    assert "POST" in list_value(chat[0]["methods"])


def test_metrics_without_a_key_is_refused_with_a_json_string(gateway: Gateway) -> None:
    """The caller GETs `/metrics` with no API key. Metrics require auth by default
    (`litellm.require_auth_for_metrics_endpoint`), so the middleware must answer 401 with
    `content-type: application/json` and a body that is one JSON string starting with
    "Unauthorized access to metrics endpoint". This body is a bare string, not the OpenAI error envelope."""
    response: Final = gateway.client.get("/metrics")
    assert response.status_code == 401, response.text
    assert response.headers["content-type"] == "application/json"
    body: Final = response.json()
    assert isinstance(body, str)
    assert body.startswith("Unauthorized access to metrics endpoint")


def test_a_json_chat_response_is_not_compressed_even_when_the_client_accepts_gzip(gateway: Gateway) -> None:
    """A chat model is registered. The caller posts a chat request with `Accept-Encoding: gzip`. The proxy only
    compresses a fixed list of dashboard GET routes (`gzip_middleware.py:12`), so this response must carry no
    `content-encoding` header and a `content-length` equal to the plain JSON body's size."""
    with front_door(gateway) as door:
        response: Final = gateway.request(
            "POST", CHAT, chat_body(door.chat_model()), headers={"Accept-Encoding": "gzip"}
        )
        assert response.status_code == 200, response.text
        assert "content-encoding" not in response.headers
        assert int(response.headers["content-length"]) == len(response.content)


def test_the_openapi_schema_is_not_compressed_even_when_the_client_accepts_gzip(gateway: Gateway) -> None:
    """The caller GETs `/openapi.json`, a body of several megabytes, with `Accept-Encoding: gzip`. It is not on
    the dashboard gzip list, so the response must carry no `content-encoding`. A front door that adds its own
    compression layer would change what proxies and CDNs in front of it see."""
    response: Final = gateway.client.get("/openapi.json", headers={"Accept-Encoding": "gzip"}, timeout=OPENAPI_SECONDS)
    assert response.status_code == 200, response.text[:300]
    assert "content-encoding" not in response.headers


def test_a_dashboard_aggregate_route_answers_the_same_json_with_and_without_gzip(gateway: Gateway) -> None:
    """The caller GETs `/user/daily/activity/aggregated` for one day twice, once with `Accept-Encoding: identity`
    and once with `Accept-Encoding: gzip`. Both must be 200 with equal JSON once decoded, the identity response
    must carry no `content-encoding`, and the gzip response may only be `gzip` or uncompressed. The route is on
    the gzip list, which compresses bodies of 500 bytes or more; the body size here depends on stored data."""
    params: Final = {"start_date": "2020-01-01", "end_date": "2020-01-01"}
    path: Final = "/user/daily/activity/aggregated"
    plain: Final = gateway.request("GET", path, params=params, headers={"Accept-Encoding": "identity"})
    packed: Final = gateway.request("GET", path, params=params, headers={"Accept-Encoding": "gzip"})
    assert (plain.status_code, packed.status_code) == (200, 200), (plain.text, packed.text)
    assert "content-encoding" not in plain.headers
    assert packed.headers.get("content-encoding") in (None, "gzip")
    assert packed.json() == plain.json()

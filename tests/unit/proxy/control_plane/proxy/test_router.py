from typing import Final

import httpx

from litellm.proxy.control_plane.auth.rbac import EDITOR_ROLE, USAGE_VIEWER_ROLE, VIEWER_ROLE
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient, RetryPolicy
from litellm.proxy.control_plane.dataplanes.dependencies import get_dataplane_client
from tests.unit.proxy.control_plane.support import (
    response,
    rig,
)

EU = "https://eu.test"


def test_get_is_forwarded_as_the_caller_with_query_and_the_dataplane_status_comes_back() -> None:
    client, transport, minter = rig({f"{EU}/key/list": [response(200, {"keys": ["k1"], "total_count": 1})]})
    reply: Final = client.get("/api/v1/dataplanes/eu/proxy/key/list?page=2&size=5", headers=minter.headers("eu"))
    assert (reply.status_code, reply.json()) == (200, {"keys": ["k1"], "total_count": 1})
    assert reply.headers["X-LiteLLM-Dataplane"] == "eu"
    url, params, headers = transport.calls[0]
    assert (url, params) == (f"{EU}/key/list", (("page", "2"), ("size", "5")))
    assert headers is not None and headers["X-LiteLLM-Impersonate-User"] == "alice"


def test_get_with_a_trailing_slash_reaches_the_dataplane_without_it() -> None:
    client, transport, minter = rig({f"{EU}/key/list": [response(200, {"keys": ["k1"], "total_count": 1})]})

    reply: Final = client.get("/api/v1/dataplanes/eu/proxy/key/list/", headers=minter.headers("eu"))

    assert (reply.status_code, reply.json()) == (200, {"keys": ["k1"], "total_count": 1})
    assert transport.calls[0][0] == f"{EU}/key/list"


def test_oversized_dataplane_response_is_a_control_plane_bad_gateway() -> None:
    client, transport, minter = rig({f"{EU}/key/list": [response(200, {"oversized": "body"})]})
    client.app.dependency_overrides[get_dataplane_client] = lambda: DataplaneClient(
        transport, timeout_seconds=1, max_response_bytes=4, retry=RetryPolicy(attempts=1)
    )

    reply: Final = client.get("/api/v1/dataplanes/eu/proxy/key/list", headers=minter.headers("eu"))

    assert reply.status_code == 502
    assert reply.json()["type"].endswith("dataplane-response-too-large")


def test_scope_denied_passthrough_reads_and_writes_are_audited() -> None:
    client, transport, minter = rig({})
    headers: Final = minter.headers("eu", role=USAGE_VIEWER_ROLE)

    read: Final = client.get("/api/v1/dataplanes/eu/proxy/key/sk-live-secret/info", headers=headers)
    write: Final = client.post("/api/v1/dataplanes/eu/proxy/key/sk-live-secret/regenerate", headers=headers)

    assert (read.status_code, write.status_code) == (403, 403)
    assert transport.calls == []
    trail: Final = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert {
        (event["actor"], event["dataplane_id"], event["method"], event["path"], event["decision"]) for event in trail
    } == {
        ("alice", "eu", "GET", "/key/sk-[redacted]", "denied"),
        ("alice", "eu", "POST", "/key/sk-[redacted]/regenerate", "denied"),
    }
    assert all("Missing scope" in event["detail"] for event in trail)


def test_get_preserves_repeated_query_parameters_in_order() -> None:
    client, transport, minter = rig({f"{EU}/key/list": [response(200, {})]})
    reply: Final = client.get(
        "/api/v1/dataplanes/eu/proxy/key/list?team_id=first&team_id=second",
        headers=minter.headers("eu"),
    )
    assert reply.status_code == 200
    assert transport.calls[0][1] == (("team_id", "first"), ("team_id", "second"))


def test_writes_need_proxy_write_go_out_once_and_are_audited() -> None:
    client, transport, minter = rig(
        {
            f"{EU}/key/generate": [response(200, {"key": "sk-new", "token_id": "t1"})],
            f"{EU}/team/update": [httpx.ReadTimeout("slow")],
            f"{EU}/key/delete": [response(400, {"error": {"message": "no such key"}})],
        }
    )
    viewer: Final = minter.headers("eu", role=VIEWER_ROLE)
    editor: Final = minter.headers("eu", role=EDITOR_ROLE)
    denied: Final = client.post("/api/v1/dataplanes/eu/proxy/key/generate", json={"key_alias": "x"}, headers=viewer)
    assert denied.status_code == 403 and "proxy:write" in denied.json()["detail"]
    created: Final = client.post("/api/v1/dataplanes/eu/proxy/key/generate", json={"key_alias": "x"}, headers=editor)
    assert (created.status_code, created.json()["token_id"]) == (200, "t1")
    assert transport.writes == [("POST", f"{EU}/key/generate", {"key_alias": "x"})]
    timed_out: Final = client.patch("/api/v1/dataplanes/eu/proxy/team/update", json={"team_id": "t"}, headers=editor)
    assert timed_out.status_code == 502 and "1 attempt" in timed_out.json()["detail"]
    failed: Final = client.delete("/api/v1/dataplanes/eu/proxy/key/delete", headers=editor)
    assert (failed.status_code, failed.json()) == (400, {"error": {"message": "no such key"}})
    trail: Final = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert [(e["method"], e["path"], e["decision"], e["status"]) for e in trail] == [
        ("DELETE", "/key/delete", "allowed", 400),
        ("PATCH", "/team/update", "unreachable", None),
        ("POST", "/key/generate", "allowed", 200),
        ("POST", "/key/generate", "denied", 403),
    ]
    assert trail[2]["actor"] == "alice" and trail[2]["credential"] == "service_token"


def test_passthrough_write_accepts_a_200_kilobyte_body() -> None:
    client, transport, minter = rig({f"{EU}/key/generate": [response(200, {"accepted": True})]})
    payload: Final = {"key_alias": "x" * (200 * 1024)}

    reply: Final = client.post(
        "/api/v1/dataplanes/eu/proxy/key/generate",
        json=payload,
        headers=minter.headers("eu", role=EDITOR_ROLE),
    )

    assert (reply.status_code, reply.json()) == (200, {"accepted": True})
    assert transport.writes == [("POST", f"{EU}/key/generate", payload)]


def test_denied_paths_are_403_audited_and_never_reach_the_dataplane() -> None:
    client, transport, minter = rig({})
    blocked: Final = client.post(
        "/api/v1/dataplanes/eu/proxy/v1/chat/completions", json={"model": "x"}, headers=minter.headers(role=EDITOR_ROLE)
    )
    assert blocked.status_code == 403 and "inference" in blocked.json()["detail"]
    assert transport.calls == []
    trail: Final = client.get("/api/v1/audit?dataplane=eu", headers=minter.headers(role="admin")).json()["items"]
    assert [(e["path"], e["decision"]) for e in trail] == [("/v1/chat/completions", "denied")]


def test_a_non_object_body_is_400_and_unbound_dataplane_is_403() -> None:
    client, _, minter = rig({})
    bad: Final = client.post(
        "/api/v1/dataplanes/eu/proxy/key/generate",
        content=b"[1,2]",
        headers={**minter.headers(role=EDITOR_ROLE), "Content-Type": "application/json"},
    )
    assert bad.status_code == 400
    unbound: Final = client.get("/api/v1/dataplanes/us/proxy/key/list", headers=minter.headers("eu"))
    assert unbound.status_code == 403
    trail: Final = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert [(event["method"], event["path"], event["decision"], event["status"]) for event in trail] == [
        ("GET", "/key/list", "denied", 403),
        ("POST", "/key/generate", "denied", 400),
    ]


def test_audit_read_needs_its_scope_and_filters_by_actor() -> None:
    client, _, minter = rig({f"{EU}/key/list": [response(200, {})]})
    client.get("/api/v1/dataplanes/eu/proxy/key/list", headers=minter.headers("eu"))
    assert client.get("/api/v1/audit", headers=minter.headers(role=EDITOR_ROLE)).status_code == 403
    assert client.get("/api/v1/audit?actor=bob", headers=minter.headers(role="admin")).json()["items"] == []
    assert len(client.get("/api/v1/audit?actor=alice", headers=minter.headers(role="admin")).json()["items"]) == 1


def test_audit_reads_are_clamped_to_the_callers_dataplane_binding() -> None:
    client, _, minter = rig({f"{EU}/key/list": [response(200, {})], "https://us.test/key/list": [response(200, {})]})
    client.get("/api/v1/dataplanes/eu/proxy/key/list", headers=minter.headers("eu"))
    client.get("/api/v1/dataplanes/us/proxy/key/list", headers=minter.headers("us"))
    eu_auditor: Final = minter.headers("eu", role="admin")
    unfiltered: Final = client.get("/api/v1/audit", headers=eu_auditor).json()["items"]
    assert [e["dataplane_id"] for e in unfiltered] == ["eu"]
    assert client.get("/api/v1/audit?dataplane=us", headers=eu_auditor).status_code == 403
    assert len(client.get("/api/v1/audit?dataplane=eu", headers=eu_auditor).json()["items"]) == 1
    fleet: Final = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert sorted(e["dataplane_id"] for e in fleet) == ["eu", "us"]


def test_a_redirecting_dataplane_is_a_502_and_the_redirect_is_never_followed() -> None:
    client, transport, minter = rig(
        {f"{EU}/v1/mcp/server/tools/call": [httpx.Response(302, headers={"Location": "http://127.0.0.1:9/"})]}
    )
    reply: Final = client.get(
        "/api/v1/dataplanes/eu/proxy/v1/mcp/server/tools/call", headers=minter.headers("eu")
    )
    assert reply.status_code == 502 and "redirect" in reply.json()["detail"]
    assert [url for url, _, _ in transport.calls] == [f"{EU}/v1/mcp/server/tools/call"]
    trail: Final = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert (trail[0]["decision"], trail[0]["status"]) == ("denied", 302)


def test_key_bearing_paths_are_redacted_in_the_audit_trail() -> None:
    client, _, minter = rig({f"{EU}/key/sk-live-123/reset_spend": [response(200, {})]})
    assert (
        client.post(
            "/api/v1/dataplanes/eu/proxy/key/sk-live-123/reset_spend",
            json={},
            headers=minter.headers("eu", role=EDITOR_ROLE),
        ).status_code
        == 200
    )
    trail: Final = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert trail[0]["path"] == "/key/sk-[redacted]/reset_spend"


def test_a_denied_key_bearing_path_is_redacted_in_the_403_too() -> None:
    client, _, minter = rig({})
    denied: Final = client.get("/api/v1/dataplanes/eu/proxy/key/sk-live-123/x%25y", headers=minter.headers("eu"))
    assert denied.status_code == 403 and "sk-live-123" not in denied.text and "sk-[redacted]" in denied.text


def test_non_json_management_replies_preserve_the_body_and_headers() -> None:
    content: Final = b"user_id,email\nu1,user@example.com\n"
    reply: Final = httpx.Response(
        200,
        content=content,
        headers={
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Disposition": 'attachment; filename="users.csv"',
            "Cache-Control": "private, max-age=0",
        },
        request=httpx.Request("GET", f"{EU}/user/export"),
    )
    client, _, minter = rig({f"{EU}/user/export": [reply]})

    response: Final = client.get("/api/v1/dataplanes/eu/proxy/user/export", headers=minter.headers("eu"))

    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == 'attachment; filename="users.csv"'
    assert response.headers["cache-control"] == "private, max-age=0"
    assert response.headers["X-LiteLLM-Dataplane"] == "eu"


def test_empty_management_replies_without_content_type_preserve_status_and_headers() -> None:
    reply: Final = httpx.Response(
        202,
        content=b"",
        headers={"Cache-Control": "private, max-age=0"},
        request=httpx.Request("GET", f"{EU}/user/export"),
    )
    client, _, minter = rig({f"{EU}/user/export": [reply]})

    response: Final = client.get("/api/v1/dataplanes/eu/proxy/user/export", headers=minter.headers("eu"))

    assert (response.status_code, response.content) == (202, b"")
    assert response.headers["cache-control"] == "private, max-age=0"
    assert response.headers["X-LiteLLM-Dataplane"] == "eu"


def test_empty_json_management_reply_preserves_empty_body_and_content_type() -> None:
    reply: Final = httpx.Response(
        200,
        content=b"",
        headers={"Content-Type": "application/json"},
        request=httpx.Request("GET", f"{EU}/user/export"),
    )
    client, _, minter = rig({f"{EU}/user/export": [reply]})

    response: Final = client.get("/api/v1/dataplanes/eu/proxy/user/export", headers=minter.headers("eu"))

    assert (response.status_code, response.content, response.headers["content-type"]) == (
        200,
        b"",
        "application/json",
    )


def test_invalid_json_management_reply_preserves_exact_body_and_content_type() -> None:
    content: Final = b'{"invalid":'
    reply: Final = httpx.Response(
        200,
        content=content,
        headers={"Content-Type": "application/json"},
        request=httpx.Request("GET", f"{EU}/user/export"),
    )
    client, _, minter = rig({f"{EU}/user/export": [reply]})

    response: Final = client.get("/api/v1/dataplanes/eu/proxy/user/export", headers=minter.headers("eu"))

    assert (response.status_code, response.content, response.headers["content-type"]) == (
        200,
        content,
        "application/json",
    )


def test_json_management_replies_forward_cache_control() -> None:
    reply: Final = httpx.Response(
        200,
        json={"users": []},
        headers={"Cache-Control": "private, max-age=0"},
        request=httpx.Request("GET", f"{EU}/user/list"),
    )
    client, _, minter = rig({f"{EU}/user/list": [reply]})

    response: Final = client.get("/api/v1/dataplanes/eu/proxy/user/list", headers=minter.headers("eu"))

    assert response.json() == {"users": []}
    assert response.headers["cache-control"] == "private, max-age=0"


def test_json_management_replies_forward_content_disposition() -> None:
    reply: Final = httpx.Response(
        200,
        json={"users": []},
        headers={"Content-Disposition": 'attachment; filename="users.json"'},
        request=httpx.Request("GET", f"{EU}/user/export"),
    )
    client, _, minter = rig({f"{EU}/user/export": [reply]})

    response: Final = client.get("/api/v1/dataplanes/eu/proxy/user/export", headers=minter.headers("eu"))

    assert response.json() == {"users": []}
    assert response.headers["content-disposition"] == 'attachment; filename="users.json"'


def test_head_is_forwarded_as_a_read_and_returns_no_body() -> None:
    reply: Final = httpx.Response(200, request=httpx.Request("HEAD", f"{EU}/key/list"))
    client, transport, minter = rig({f"{EU}/key/list": [reply]})

    response: Final = client.head("/api/v1/dataplanes/eu/proxy/key/list", headers=minter.headers("eu"))

    assert (response.status_code, response.content) == (200, b"")
    assert transport.writes == []
    trail: Final = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert (trail[0]["method"], trail[0]["decision"]) == ("HEAD", "allowed")


def test_guardrail_apply_is_denied_even_under_a_management_prefix() -> None:
    client, transport, minter = rig({})

    reply: Final = client.get("/api/v1/dataplanes/eu/proxy/guardrails/apply_guardrail", headers=minter.headers("eu"))

    assert reply.status_code == 403 and "guardrail apply" in reply.json()["detail"]
    assert transport.calls == []

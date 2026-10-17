import httpx

from litellm.proxy.control_plane.auth.rbac import EDITOR_ROLE, VIEWER_ROLE
from tests.unit.proxy.control_plane.support import response, rig

EU = "https://eu.test"


def test_get_is_forwarded_as_the_caller_with_query_and_the_dataplane_status_comes_back() -> None:
    client, transport, minter = rig({f"{EU}/key/list": [response(200, {"keys": ["k1"], "total_count": 1})]})
    reply = client.get("/api/v1/dataplanes/eu/proxy/key/list?page=2&size=5", headers=minter.headers("eu"))
    assert (reply.status_code, reply.json()) == (200, {"keys": ["k1"], "total_count": 1})
    assert reply.headers["X-LiteLLM-Dataplane"] == "eu"
    url, params, headers = transport.calls[0]
    assert (url, params) == (f"{EU}/key/list", {"page": "2", "size": "5"})
    assert headers is not None and headers["X-LiteLLM-Impersonate-User"] == "alice"


def test_writes_need_proxy_write_go_out_once_and_are_audited() -> None:
    client, transport, minter = rig(
        {
            f"{EU}/key/generate": [response(200, {"key": "sk-new", "token_id": "t1"})],
            f"{EU}/team/update": [httpx.ReadTimeout("slow")],
            f"{EU}/key/delete": [response(400, {"error": {"message": "no such key"}})],
        }
    )
    viewer = minter.headers("eu", role=VIEWER_ROLE)
    editor = minter.headers("eu", role=EDITOR_ROLE)
    denied = client.post("/api/v1/dataplanes/eu/proxy/key/generate", json={"key_alias": "x"}, headers=viewer)
    assert denied.status_code == 403 and "proxy:write" in denied.json()["detail"]
    created = client.post("/api/v1/dataplanes/eu/proxy/key/generate", json={"key_alias": "x"}, headers=editor)
    assert (created.status_code, created.json()["token_id"]) == (200, "t1")
    assert transport.writes == [("POST", f"{EU}/key/generate", {"key_alias": "x"})]
    timed_out = client.patch("/api/v1/dataplanes/eu/proxy/team/update", json={"team_id": "t"}, headers=editor)
    assert timed_out.status_code == 502 and "1 attempt" in timed_out.json()["detail"]
    failed = client.delete("/api/v1/dataplanes/eu/proxy/key/delete", headers=editor)
    assert (failed.status_code, failed.json()) == (400, {"error": {"message": "no such key"}})
    trail = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert [(e["method"], e["path"], e["decision"], e["status"]) for e in trail] == [
        ("DELETE", "/key/delete", "allowed", 400),
        ("PATCH", "/team/update", "unreachable", None),
        ("POST", "/key/generate", "allowed", 200),
    ]
    assert trail[2]["actor"] == "alice" and trail[2]["credential"] == "service_token"


def test_denied_paths_are_403_audited_and_never_reach_the_dataplane() -> None:
    client, transport, minter = rig({})
    blocked = client.post(
        "/api/v1/dataplanes/eu/proxy/v1/chat/completions", json={"model": "x"}, headers=minter.headers(role=EDITOR_ROLE)
    )
    assert blocked.status_code == 403 and "inference" in blocked.json()["detail"]
    assert transport.calls == []
    trail = client.get("/api/v1/audit?dataplane=eu", headers=minter.headers(role="admin")).json()["items"]
    assert [(e["path"], e["decision"]) for e in trail] == [("/v1/chat/completions", "denied")]


def test_a_non_object_body_is_400_and_unbound_dataplane_is_403() -> None:
    client, _, minter = rig({})
    bad = client.post(
        "/api/v1/dataplanes/eu/proxy/key/generate",
        content=b"[1,2]",
        headers={**minter.headers(role=EDITOR_ROLE), "Content-Type": "application/json"},
    )
    assert bad.status_code == 400
    assert client.get("/api/v1/dataplanes/us/proxy/key/list", headers=minter.headers("eu")).status_code == 403


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
    eu_auditor = minter.headers("eu", role="admin")
    unfiltered = client.get("/api/v1/audit", headers=eu_auditor).json()["items"]
    assert [e["dataplane_id"] for e in unfiltered] == ["eu"]
    assert client.get("/api/v1/audit?dataplane=us", headers=eu_auditor).status_code == 403
    assert len(client.get("/api/v1/audit?dataplane=eu", headers=eu_auditor).json()["items"]) == 1
    fleet = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert sorted(e["dataplane_id"] for e in fleet) == ["eu", "us"]


def test_a_redirecting_dataplane_is_a_502_and_the_redirect_is_never_followed() -> None:
    client, transport, minter = rig(
        {f"{EU}/v1/mcp/server/oauth/x/authorize": [httpx.Response(302, headers={"Location": "http://127.0.0.1:9/"})]}
    )
    reply = client.get("/api/v1/dataplanes/eu/proxy/v1/mcp/server/oauth/x/authorize", headers=minter.headers("eu"))
    assert reply.status_code == 502 and "redirect" in reply.json()["detail"]
    assert [url for url, _, _ in transport.calls] == [f"{EU}/v1/mcp/server/oauth/x/authorize"]
    trail = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
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
    trail = client.get("/api/v1/audit", headers=minter.headers(role="admin")).json()["items"]
    assert trail[0]["path"] == "/key/sk-[redacted]/reset_spend"


def test_a_denied_key_bearing_path_is_redacted_in_the_403_too() -> None:
    client, _, minter = rig({})
    denied = client.get("/api/v1/dataplanes/eu/proxy/key/sk-live-123/x%25y", headers=minter.headers("eu"))
    assert denied.status_code == 403 and "sk-live-123" not in denied.text and "sk-[redacted]" in denied.text

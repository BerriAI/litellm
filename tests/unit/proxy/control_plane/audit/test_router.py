from litellm.proxy.control_plane.auth.rbac import EDITOR_ROLE
from tests.unit.proxy.control_plane.support import response, rig

EU = "https://eu.test"


def test_ocsf_listing_returns_chained_records_newest_first_with_a_valid_chain() -> None:
    client, _, minter = rig({f"{EU}/key/list": [response(200, {})], f"{EU}/key/generate": [response(200, {})]})
    client.get("/api/v1/dataplanes/eu/proxy/key/list", headers=minter.headers("eu"))
    client.post("/api/v1/dataplanes/eu/proxy/key/generate", json={}, headers=minter.headers("eu", role=EDITOR_ROLE))
    page = client.get("/api/v1/audit/ocsf?dataplane=eu", headers=minter.headers(role="admin")).json()
    assert [(i["class_uid"], i["activity_name"], i["http_request"]["url"]["path"]) for i in page["items"]] == [
        (6003, "Create", "/key/generate"),
        (6003, "Read", "/key/list"),
    ]
    assert (page["chain_valid"], page["sequence_gaps"]) == (True, 0)
    assert page["items"][0]["unmapped"]["litellm"]["prev_hash"] == page["items"][1]["unmapped"]["litellm"]["hash"]
    newest = client.get("/api/v1/audit/ocsf?limit=1", headers=minter.headers(role="admin")).json()
    assert (newest["chain_valid"], newest["sequence_gaps"], len(newest["items"])) == (True, 0, 1)
    reads = client.get("/api/v1/audit/ocsf?actor=alice&dataplane=eu", headers=minter.headers(role="admin")).json()
    assert reads["chain_valid"] is True, "a filtered page is intact even though it skips records"


def test_logins_denied_scopes_and_rejected_tokens_are_audited_against_the_control_plane() -> None:
    client, _, minter = rig({}, UI_USERNAME="admin", UI_PASSWORD="pw")
    assert client.post("/api/v1/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"username": "admin", "password": "pw"}).status_code == 200
    assert client.get("/api/v1/audit", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert client.get("/api/v1/audit", headers=minter.headers(role=EDITOR_ROLE)).status_code == 403
    trail = client.get("/api/v1/audit?dataplane=control-plane", headers=minter.headers(role="admin")).json()["items"]
    assert [(e["actor"], e["path"], e["decision"], e["status"], e["detail"]) for e in trail] == [
        ("alice", "/api/v1/audit", "denied", 403, "missing scope: audit:read"),
        ("anonymous", "/api/v1/audit", "denied", 401, trail[1]["detail"]),
        ("admin", "/auth/login", "allowed", None, None),
        ("admin", "/auth/login", "denied", None, "invalid credentials"),
    ]
    ocsf = client.get(
        "/api/v1/audit/ocsf?dataplane=control-plane&limit=1&actor=admin", headers=minter.headers(role="admin")
    )
    assert (ocsf.json()["items"][0]["class_uid"], ocsf.json()["chain_valid"]) == (3002, True)
    bound = client.get("/api/v1/audit?dataplane=control-plane", headers=minter.headers("control-plane", role="admin"))
    assert bound.status_code == 403, "control plane events are fleet-wide only"
    implicit = client.get("/api/v1/audit", headers=minter.headers("control-plane", "eu", role="admin")).json()
    assert implicit["items"] == [], "a binding that names the control plane does not grant its events either"


def test_strict_mode_refuses_writes_but_not_reads_when_no_sink_can_take_the_record() -> None:
    client, transport, minter = rig(
        {f"{EU}/key/list": [response(200, {}), response(200, {})]}, audit_strict=True, audit_queue_size=1
    )
    client.get("/api/v1/dataplanes/eu/proxy/key/list", headers=minter.headers("eu"))
    refused = client.post(
        "/api/v1/dataplanes/eu/proxy/key/generate", json={}, headers=minter.headers("eu", role=EDITOR_ROLE)
    )
    assert (refused.status_code, refused.json()["type"]) == (503, "urn:litellm:error:audit-unavailable")
    assert transport.writes == [], "a refused write never reaches the dataplane"
    assert client.get("/api/v1/dataplanes/eu/proxy/key/list", headers=minter.headers("eu")).status_code != 503

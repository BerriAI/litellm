"""Unit tests for tools/import_mcp_servers.py (marketplace MCP importer).

Covers the pure transformation layer only — fetching is exercised via
fixtures, never live HTTP.
"""

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import import_mcp_servers as imp  # noqa: E402

# ---------------------------------------------------------------------------
# normalize_server_name
# ---------------------------------------------------------------------------


def test_normalize_server_name_replaces_hyphens_and_spaces():
    # '-' is the MCP tool prefix separator; server names must avoid it.
    assert imp.normalize_server_name("yahoo-finance") == "yahoo_finance"
    assert imp.normalize_server_name("GitHub MCP") == "github_mcp"


def test_normalize_server_name_strips_registry_namespace_chars():
    assert imp.normalize_server_name("io.github.wong2/fetch") == "io_github_wong2_fetch"


# ---------------------------------------------------------------------------
# parse_remote_listing — mcpservers.org /remote-mcp-servers/{slug} pages
# ---------------------------------------------------------------------------

NOTION_HTML = (
    '<script>self.__next_f.push({name:"Notion",description:"Connect to '
    'Notion workspaces",logo:"https://x/logo.png",endpoints:[{type:"http",'
    'url:"https://mcp.notion.com/mcp"}],authType:"oauth",externalLink:'
    '"https://developers.notion.com",featured:!0})</script>'
)

SQUARE_HTML = (
    '<script>{"name":"Square","description":"Payments API",'
    '"endpoints":[{"type":"sse","url":"https://mcp.squareup.com/sse"}],'
    '"authType":"oauth"}</script>'
)

PARALLEL_HTML = (
    "<script>{name:'Parallel',description:'Web search',"
    'endpoints:[{type:"http",url:"https://search.parallel.ai/mcp"}],'
    'authType:"open"}</script>'
)


def test_parse_remote_listing_unquoted_keys():
    rec = imp.parse_remote_listing(NOTION_HTML, slug="notion")
    assert rec["name"] == "Notion"
    assert rec["transport"] == "http"
    assert rec["url"] == "https://mcp.notion.com/mcp"
    assert rec["auth"] == "oauth"
    assert rec["description"] == "Connect to Notion workspaces"


def test_parse_remote_listing_quoted_keys_sse():
    rec = imp.parse_remote_listing(SQUARE_HTML, slug="square")
    assert rec["transport"] == "sse"
    assert rec["url"] == "https://mcp.squareup.com/sse"
    assert rec["auth"] == "oauth"


def test_parse_remote_listing_open_auth():
    rec = imp.parse_remote_listing(PARALLEL_HTML, slug="parallel")
    assert rec["auth"] == "open"


# Real production format (TanStack serializer): `$R[n]=` reference
# assignments appear between `:` and the value being serialized.
NOTION_TSR_HTML = (
    '{name:"Notion",description:"Pages, databases, comments, workspace '
    'search",logo:"https://svgl.app/library/notion.svg",'
    'endpoints:$R[20]=[$R[21]={type:"http",url:"https://mcp.notion.com/mcp"}],'
    'authType:"oauth",externalLink:"https://developers.notion.com/guides",'
    "featured:!0}"
)


def test_parse_remote_listing_tanstack_reference_assignments():
    rec = imp.parse_remote_listing(NOTION_TSR_HTML, slug="notion")
    assert rec["name"] == "Notion"
    assert rec["transport"] == "http"
    assert rec["url"] == "https://mcp.notion.com/mcp"
    assert rec["auth"] == "oauth"
    assert rec["description"] == "Pages, databases, comments, workspace search"


def test_parse_remote_listing_ignores_decoy_records_before_the_real_one():
    # Pages carry earlier serialized records (router state, nav items) that
    # also have name/description fields; the parser must bind to the record
    # the endpoints belong to, not the first match in the page.
    html = (
        '{name:"home",description:"nav item"}...'
        '{name:"Asana",description:"Work management",'
        'endpoints:$R[9]=[$R[10]={type:"http",url:"https://mcp.asana.com/v2/mcp"}],'
        'authType:"oauth"}'
    )
    rec = imp.parse_remote_listing(html, slug="asana")
    assert rec["name"] == "Asana"
    assert rec["description"] == "Work management"


def test_parse_remote_listing_no_endpoint_raises():
    with pytest.raises(ValueError):
        imp.parse_remote_listing("<html>nothing here</html>", slug="empty")


def test_parse_remote_listing_falls_back_to_slug_for_name():
    html = 'endpoints:[{type:"http",url:"https://a.example/mcp"}],authType:"open"'
    rec = imp.parse_remote_listing(html, slug="acme-tool")
    assert rec["name"] == "acme-tool"


# ---------------------------------------------------------------------------
# registry_entry_to_records — official registry server.json
# ---------------------------------------------------------------------------


def _registry_entry(**overrides):
    entry = {
        "name": "io.github.acme/widgets",
        "description": "Acme widgets MCP",
        "status": "active",
        "version": "1.2.0",
        "remotes": [],
        "packages": [],
    }
    entry.update(overrides)
    return entry


def test_registry_remote_streamable_http_maps_to_http():
    entry = _registry_entry(
        remotes=[{"type": "streamable-http", "url": "https://api.acme.dev/mcp"}]
    )
    records = imp.registry_entry_to_records(entry)
    assert len(records) == 1
    assert records[0]["transport"] == "http"
    assert records[0]["url"] == "https://api.acme.dev/mcp"
    assert records[0]["auth"] == "oauth"  # remote default: assume auth required


def test_registry_remote_sse_maps_to_sse():
    entry = _registry_entry(remotes=[{"type": "sse", "url": "https://a.dev/sse"}])
    records = imp.registry_entry_to_records(entry)
    assert records[0]["transport"] == "sse"


def test_registry_npm_package_maps_to_npx_stdio():
    entry = _registry_entry(
        packages=[
            {
                "registryType": "npm",
                "identifier": "@acme/widgets-mcp",
                "version": "1.2.0",
                "environmentVariables": [{"name": "ACME_API_KEY"}],
            }
        ]
    )
    records = imp.registry_entry_to_records(entry)
    assert len(records) == 1
    rec = records[0]
    assert rec["transport"] == "stdio"
    assert rec["command"] == "npx"
    assert rec["args"] == ["-y", "@acme/widgets-mcp@1.2.0"]
    assert rec["env"] == {"ACME_API_KEY": ""}


def test_registry_pypi_package_maps_to_uvx_stdio():
    entry = _registry_entry(
        packages=[{"registry_type": "pypi", "identifier": "acme-mcp", "version": "2.0"}]
    )
    records = imp.registry_entry_to_records(entry)
    assert records[0]["command"] == "uvx"
    assert records[0]["args"] == ["acme-mcp==2.0"]


def test_registry_oci_package_maps_to_docker_stdio():
    entry = _registry_entry(
        packages=[{"registryType": "oci", "identifier": "acme/mcp:1.0"}]
    )
    records = imp.registry_entry_to_records(entry)
    assert records[0]["command"] == "docker"
    assert records[0]["args"] == ["run", "-i", "--rm", "acme/mcp:1.0"]


def test_registry_deprecated_entry_yields_nothing():
    entry = _registry_entry(
        status="deprecated",
        remotes=[{"type": "streamable-http", "url": "https://a.dev/mcp"}],
    )
    assert imp.registry_entry_to_records(entry) == []


def test_registry_unknown_package_registry_skipped():
    entry = _registry_entry(
        packages=[{"registryType": "nuget", "identifier": "Acme.Mcp"}]
    )
    assert imp.registry_entry_to_records(entry) == []


# ---------------------------------------------------------------------------
# select_servers — tiered selection, dedupe, limit
# ---------------------------------------------------------------------------


def _remote_rec(name, url, **overrides):
    rec = {
        "name": name,
        "description": "",
        "transport": "http",
        "url": url,
        "auth": "oauth",
        "origin": "https://example.com/" + name,
    }
    rec.update(overrides)
    return rec


def test_select_servers_dedupes_by_url_across_tiers():
    tier1 = [_remote_rec("notion", "https://mcp.notion.com/mcp")]
    tier2 = [_remote_rec("notion_registry", "https://mcp.notion.com/mcp/")]
    selected = imp.select_servers([tier1, tier2], limit=10)
    assert len(selected) == 1
    assert selected[0]["name"] == "notion"  # tier 1 wins


def test_select_servers_resolves_name_collisions():
    tier1 = [_remote_rec("acme", "https://a.example/mcp")]
    tier2 = [_remote_rec("acme", "https://b.example/mcp")]
    selected = imp.select_servers([tier1, tier2], limit=10)
    names = [imp.normalize_server_name(r["name"]) for r in selected]
    assert len(names) == len(set(names)) == 2


def test_select_servers_respects_limit():
    tier = [_remote_rec(f"s{i}", f"https://s{i}.example/mcp") for i in range(20)]
    assert len(imp.select_servers([tier], limit=5)) == 5


def test_select_servers_caps_records_per_host():
    # Aggregator hosts (e.g. Smithery) re-host hundreds of servers; without
    # a per-host cap they crowd out everything else in the registry tier.
    tier = [
        _remote_rec(f"agg{i}", f"https://server.aggregator.example/{i}/mcp")
        for i in range(10)
    ] + [_remote_rec("vendor", "https://mcp.vendor.example/mcp")]
    selected = imp.select_servers([tier], limit=50, max_per_host=3)
    hosts = [r["url"].split("/")[2] for r in selected]
    assert hosts.count("server.aggregator.example") == 3
    assert "mcp.vendor.example" in hosts


# ---------------------------------------------------------------------------
# prioritize_registry_remotes — quality heuristic for the registry tier
# (the registry returns entries alphabetically, not by popularity)
# ---------------------------------------------------------------------------


def test_prioritize_registry_remotes_vendor_domains_first():
    github_ns = _remote_rec("io.github.someone/tool", "https://a.example/mcp")
    vendor_ns = _remote_rec("com.acme/widgets", "https://mcp.acme.example/mcp")
    aggregator = _remote_rec("com.zeta/hosted", "https://server.smithery.ai/zeta/mcp")
    ordered = imp.prioritize_registry_remotes([aggregator, github_ns, vendor_ns])
    assert ordered[0] is vendor_ns  # DNS-verified vendor namespace wins
    assert ordered[-1] is aggregator  # re-hosted aggregator endpoints last


# ---------------------------------------------------------------------------
# build_payload — final POST /v1/mcp/server body
# ---------------------------------------------------------------------------


def test_build_payload_remote_oauth():
    rec = _remote_rec("Yahoo-Finance", "https://gw.example/yahoo/mcp")
    payload = imp.build_payload(rec)
    assert payload["server_name"] == "yahoo_finance"
    assert payload["transport"] == "http"
    assert payload["url"] == "https://gw.example/yahoo/mcp"
    assert payload["auth_type"] == "oauth2"
    assert payload["allow_all_keys"] is True
    assert payload["source_url"] == rec["origin"]


def test_build_payload_remote_open_auth_maps_to_none():
    rec = _remote_rec("parallel", "https://search.parallel.ai/mcp", auth="open")
    payload = imp.build_payload(rec)
    assert payload["auth_type"] == "none"


def test_build_payload_stdio_is_gated_not_public():
    rec = {
        "name": "acme_widgets",
        "description": "Acme",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@acme/widgets-mcp@1.2.0"],
        "env": {"ACME_API_KEY": ""},
        "auth": "none",
        "origin": "https://registry.modelcontextprotocol.io",
    }
    payload = imp.build_payload(rec)
    assert payload["transport"] == "stdio"
    assert payload["command"] == "npx"
    # stdio servers are imported gated: not exposed to all keys, parked in a
    # review access group until an admin activates them.
    assert payload["allow_all_keys"] is False
    assert imp.STDIO_REVIEW_ACCESS_GROUP in payload["mcp_access_groups"]

#!/usr/bin/env python3
"""Bulk-import MCP servers from public marketplaces into a LiteLLM proxy.

Sources (tiered, earlier tiers win on dedupe):
  1. mcpservers.org /remote-mcp-servers — ~195 curated, hosted http/sse
     endpoints (vendor-official: GitHub, Notion, Stripe, ...). Parsed from
     the structured record embedded in each listing page.
  2. registry.modelcontextprotocol.io /v0/servers — the official MCP
     registry. `remotes[]` entries map to http/sse; `packages[]` (npm /
     pypi / oci) map to stdio via npx / uvx / docker.

Remote (http/sse) servers are imported live (`allow_all_keys: true`).
Stdio servers execute third-party code on the proxy host, so they are
imported *gated*: `allow_all_keys: false` and parked in the
`marketplace_stdio_pending_review` access group until an admin activates
them. Use --skip-stdio to not import them at all.

Usage:
    # Review what would be imported (writes a YAML manifest, no POSTs):
    python tools/import_mcp_servers.py --dry-run --limit 500 \\
        --manifest mcp_import_manifest.yaml

    # Import (idempotent — existing server names are skipped):
    LITELLM_ADMIN_KEY=sk-... \\
    python tools/import_mcp_servers.py --limit 500 \\
        --base-url https://tokenhub.xcity.one

    # Import a previously reviewed manifest verbatim:
    LITELLM_ADMIN_KEY=sk-... \\
    python tools/import_mcp_servers.py --from-manifest mcp_import_manifest.yaml
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import yaml

MCPSERVERS_ORG_SITEMAP = "https://mcpservers.org/sitemaps/remote-mcp-servers.xml"
OFFICIAL_REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"
USER_AGENT = "xcity-litellm-mcp-importer/1.0"

# Stdio servers run third-party code on the proxy host; they are imported
# gated behind this access group instead of being exposed to all keys.
STDIO_REVIEW_ACCESS_GROUP = "marketplace_stdio_pending_review"

# marketplace auth label -> LiteLLM MCPAuth value
AUTH_TYPE_MAP = {
    "oauth": "oauth2",
    "oauth2": "oauth2",
    "open": "none",
    "none": "none",
    "api_key": "api_key",
}

REMOTE_TRANSPORT_MAP = {
    "http": "http",
    "streamable-http": "http",
    "streamable_http": "http",
    "sse": "sse",
}


def normalize_server_name(name: str) -> str:
    """Lowercase and collapse to [a-z0-9_].

    '-' is the MCP tool prefix separator on this proxy, so hyphens (and any
    other punctuation) become underscores.
    """
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return normalized.strip("_")


# ---------------------------------------------------------------------------
# Tier 1: mcpservers.org remote listings
# ---------------------------------------------------------------------------

# Listing pages embed a serialized record whose keys may be quoted or bare,
# whose strings may use single or double quotes, and whose values may carry
# TanStack-serializer reference assignments (`$R[n]=`), e.g.
#   {name:"Notion",endpoints:$R[20]=[$R[21]={type:"http",url:"https://..."}],
#    authType:"oauth"}
_TSR_REF = r"(?:\$R\[\d+\]=)*"


def _extract_field(
    field: str, text: str, before: Optional[int] = None
) -> Optional[str]:
    """Value of `field` in the serialized record.

    With `before`, returns the LAST occurrence that starts before that
    offset — listing pages carry earlier decoy records (router state, nav
    items) that also have name/description fields, so the caller anchors
    on the endpoints match and binds fields belonging to the same record.
    """
    pattern = (
        r"[{,]\s*[\"']?"
        + re.escape(field)
        + r"[\"']?\s*:\s*"
        + _TSR_REF
        + r"([\"'])((?:(?!\1).)*)\1"
    )
    match = None
    for candidate in re.finditer(pattern, text):
        if before is not None and candidate.start() >= before:
            break
        match = candidate
        if before is None:
            break
    if not match:
        return None
    raw = match.group(2)
    return raw.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")


def parse_remote_listing(html: str, slug: str) -> Dict[str, Any]:
    """Extract one remote-server record from an mcpservers.org listing page.

    Raises ValueError when the page carries no endpoint (the parser is
    defensive: the embedded format is undocumented and may change).
    """
    endpoint = re.search(
        r"[\"']?endpoints[\"']?\s*:\s*"
        + _TSR_REF
        + r"\[\s*"
        + _TSR_REF
        + r"\{\s*[\"']?type[\"']?\s*:\s*[\"'](http|sse)[\"']\s*,\s*"
        + r"[\"']?url[\"']?\s*:\s*"
        + _TSR_REF
        + r"[\"']([^\"']+)[\"']",
        html,
    )
    if not endpoint:
        raise ValueError(f"no MCP endpoint found in listing page for {slug!r}")

    auth_match = re.search(r"[\"']?authType[\"']?\s*:\s*[\"'](\w+)[\"']", html)
    return {
        "name": _extract_field("name", html, before=endpoint.start()) or slug,
        "description": _extract_field("description", html, before=endpoint.start())
        or "",
        "transport": endpoint.group(1),
        "url": endpoint.group(2),
        "auth": auth_match.group(1) if auth_match else "oauth",
        "origin": f"https://mcpservers.org/remote-mcp-servers/{slug}",
    }


# ---------------------------------------------------------------------------
# Tier 2/3: official MCP registry (registry.modelcontextprotocol.io)
# ---------------------------------------------------------------------------


def _package_to_stdio(package: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    registry_type = (
        package.get("registryType") or package.get("registry_type") or ""
    ).lower()
    identifier = package.get("identifier") or package.get("name")
    if not identifier:
        return None
    version = package.get("version")

    if registry_type == "npm":
        command = "npx"
        args = ["-y", f"{identifier}@{version}" if version else identifier]
    elif registry_type == "pypi":
        command = "uvx"
        args = [f"{identifier}=={version}" if version else identifier]
    elif registry_type == "oci":
        command = "docker"
        args = ["run", "-i", "--rm", identifier]
    else:  # nuget / mcpb / unknown — no allowlisted runtime on the proxy
        return None

    env_vars = (
        package.get("environmentVariables")
        or package.get("environment_variables")
        or []
    )
    env = {
        var["name"]: "" for var in env_vars if isinstance(var, dict) and var.get("name")
    }
    return {"command": command, "args": args, "env": env}


def registry_entry_to_records(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map one official-registry server.json to importable records.

    A single entry may yield a remote record (preferred) and/or a stdio
    record. Non-active (deprecated/deleted) entries yield nothing.
    """
    if entry.get("status", "active") != "active":
        return []

    name = entry.get("name") or ""
    description = entry.get("description") or ""
    origin = f"https://registry.modelcontextprotocol.io/?search={name}"
    records: List[Dict[str, Any]] = []

    for remote in entry.get("remotes") or []:
        transport = REMOTE_TRANSPORT_MAP.get(
            remote.get("type") or remote.get("transport_type") or ""
        )
        url = remote.get("url")
        if not transport or not url:
            continue
        records.append(
            {
                "name": name,
                "description": description,
                "transport": transport,
                "url": url,
                # The registry does not expose auth metadata; assume the
                # hosted endpoint requires auth and let OAuth discovery /
                # the admin sort out the rest.
                "auth": "oauth",
                "origin": origin,
            }
        )

    for package in entry.get("packages") or []:
        stdio = _package_to_stdio(package)
        if stdio is None:
            continue
        records.append(
            {
                "name": name,
                "description": description,
                "transport": "stdio",
                "auth": "none",
                "origin": origin,
                **stdio,
            }
        )

    return records


# ---------------------------------------------------------------------------
# Selection + payload building
# ---------------------------------------------------------------------------


def _dedupe_key(record: Dict[str, Any]) -> Any:
    if record["transport"] == "stdio":
        return ("stdio", record["command"], tuple(record["args"]))
    return ("remote", record["url"].rstrip("/"))


def _record_host(record: Dict[str, Any]) -> str:
    url = record.get("url") or ""
    return urllib.parse.urlsplit(url).netloc.lower()


# Hosts that re-host many third-party servers under one domain; ranked
# below DNS-verified vendor endpoints and bounded by the per-host cap.
AGGREGATOR_HOSTS = {"server.smithery.ai"}


def prioritize_registry_remotes(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Quality-order registry remotes (the registry lists alphabetically).

    DNS-verified vendor namespaces (com.acme/...) outrank io.github.*
    entries, and aggregator-hosted endpoints sink to the bottom.
    """

    def sort_key(pair):
        index, record = pair
        name = (record.get("name") or "").lower()
        return (
            _record_host(record) in AGGREGATOR_HOSTS,
            name.startswith("io.github."),
            index,
        )

    return [r for _, r in sorted(enumerate(records), key=sort_key)]


def select_servers(
    tiers: List[List[Dict[str, Any]]],
    limit: int,
    max_per_host: int = 25,
) -> List[Dict[str, Any]]:
    """Merge tiers in priority order, dedupe, uniquify names, cap at limit.

    `max_per_host` bounds how many servers a single hostname may contribute
    so aggregator hosts can't crowd out the selection.
    """
    selected: List[Dict[str, Any]] = []
    seen_keys = set()
    used_names = set()
    host_counts: Dict[str, int] = {}

    for tier in tiers:
        for record in tier:
            if len(selected) >= limit:
                return selected
            key = _dedupe_key(record)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            host = _record_host(record)
            if host and host_counts.get(host, 0) >= max_per_host:
                continue
            if host:
                host_counts[host] = host_counts.get(host, 0) + 1

            base_name = normalize_server_name(record["name"]) or "imported_mcp"
            name = base_name
            suffix = 2
            while name in used_names:
                name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(name)
            selected.append({**record, "name": name})

    return selected


def build_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map an importable record onto a POST /v1/mcp/server body."""
    payload: Dict[str, Any] = {
        "server_name": normalize_server_name(record["name"]),
        "description": record.get("description") or None,
        "transport": record["transport"],
        "source_url": record.get("origin"),
    }

    if record["transport"] == "stdio":
        payload.update(
            {
                "command": record["command"],
                "args": record["args"],
                "env": record.get("env") or {},
                "auth_type": "none",
                "allow_all_keys": False,
                "mcp_access_groups": [STDIO_REVIEW_ACCESS_GROUP],
            }
        )
        return payload

    auth_type = AUTH_TYPE_MAP.get(record.get("auth") or "oauth", "oauth2")
    payload.update(
        {
            "url": record["url"],
            "auth_type": auth_type,
            "allow_all_keys": True,
            "mcp_access_groups": [],
        }
    )
    if auth_type == "oauth2":
        # Interactive upstream OAuth (user signs in to the vendor) — the
        # standard flow for hosted marketplace servers.
        payload["delegate_auth_to_upstream"] = True
    return payload


# ---------------------------------------------------------------------------
# Fetching (live HTTP — not exercised by unit tests)
# ---------------------------------------------------------------------------


def _fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def fetch_mcpservers_remote_tier(verbose: bool = True) -> List[Dict[str, Any]]:
    records = []
    try:
        sitemap = _fetch(MCPSERVERS_ORG_SITEMAP)
    except (urllib.error.URLError, OSError) as e:
        print(f"  ! tier-1 sitemap fetch failed: {e}", file=sys.stderr)
        return records

    urls = re.findall(r"<loc>([^<]+)</loc>", sitemap)
    page_urls = [u for u in urls if "/remote-mcp-servers/" in u]
    print(f"  tier-1: {len(page_urls)} remote listings on mcpservers.org")
    failures = 0
    for page_url in page_urls:
        slug = page_url.rstrip("/").rsplit("/", 1)[-1]
        try:
            records.append(parse_remote_listing(_fetch(page_url), slug))
        except (ValueError, urllib.error.URLError, OSError) as e:
            failures += 1
            if verbose:
                print(f"  ! tier-1 skip {slug}: {e}", file=sys.stderr)
    if page_urls and failures / len(page_urls) > 0.05:
        print(
            f"  ! tier-1 parse failure rate {failures}/{len(page_urls)} — "
            "the embedded page format may have changed",
            file=sys.stderr,
        )
    return records


def fetch_registry_tiers(
    max_pages: int = 40,
) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Returns (remote_records, stdio_records) from the official registry."""
    remote_records: List[Dict[str, Any]] = []
    stdio_records: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    for _ in range(max_pages):
        url = f"{OFFICIAL_REGISTRY_URL}?limit=100&version=latest"
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor)}"
        try:
            body = json.loads(_fetch(url))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            print(f"  ! registry fetch failed: {e}", file=sys.stderr)
            break

        for item in body.get("servers") or []:
            entry = item.get("server", item)
            for record in registry_entry_to_records(entry):
                if record["transport"] == "stdio":
                    stdio_records.append(record)
                else:
                    remote_records.append(record)

        metadata = body.get("metadata") or {}
        cursor = metadata.get("next_cursor") or metadata.get("nextCursor")
        if not cursor:
            break

    print(
        f"  tier-2: {len(remote_records)} remote / tier-3: {len(stdio_records)} "
        "stdio records from the official registry"
    )
    return remote_records, stdio_records


# ---------------------------------------------------------------------------
# Import (POST) + main
# ---------------------------------------------------------------------------


def post_server(base_url: str, key: str, payload: Dict[str, Any], timeout: int = 30):
    url = f"{base_url.rstrip('/')}/v1/mcp/server"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return 0, str(e)


def import_payloads(base_url: str, key: str, payloads: List[Dict[str, Any]]) -> int:
    created = skipped = failed = 0
    for payload in payloads:
        name = payload.get("server_name", "?")
        status, text = post_server(base_url, key, payload)
        if status in (200, 201):
            created += 1
            print(f"  ✓ {name}")
        elif status == 409 or "exist" in text.lower() or "duplicate" in text.lower():
            skipped += 1
            print(f"  = {name} (already exists)")
        else:
            failed += 1
            print(f"  ✗ {name} -> {status} {text[:200]}")
    print(
        f"\nDone. created={created} skipped={skipped} failed={failed} "
        f"total={len(payloads)}"
    )
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument(
        "--base-url",
        default=os.environ.get("LITELLM_BASEURL", "https://tokenhub.xcity.one"),
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--manifest",
        default="mcp_import_manifest.yaml",
        help="where --dry-run writes the reviewable payload list",
    )
    ap.add_argument(
        "--from-manifest",
        help="import a previously generated manifest instead of re-fetching",
    )
    ap.add_argument(
        "--skip-stdio",
        action="store_true",
        help="do not import stdio (package-based) servers at all",
    )
    ap.add_argument("--max-registry-pages", type=int, default=40)
    args = ap.parse_args()

    key = (
        os.environ.get("LITELLM_ADMIN_KEY")
        or os.environ.get("LITELLM_MASTER_KEY")
        or os.environ.get("PROXY_MASTER_KEY")
        or os.environ.get("LITELLM_API_KEY")
    )
    if not key and not args.dry_run:
        sys.exit(
            "Set LITELLM_ADMIN_KEY / LITELLM_MASTER_KEY (proxy admin key) "
            "in the environment."
        )

    if args.from_manifest:
        with open(args.from_manifest, "r", encoding="utf-8") as f:
            payloads = yaml.safe_load(f) or []
        print(f"Loaded {len(payloads)} payloads from {args.from_manifest}")
    else:
        print("Fetching sources ...")
        tier1 = fetch_mcpservers_remote_tier()
        tier2, tier3 = fetch_registry_tiers(max_pages=args.max_registry_pages)
        tier2 = prioritize_registry_remotes(tier2)
        tiers = [tier1, tier2] if args.skip_stdio else [tier1, tier2, tier3]
        records = select_servers(tiers, limit=args.limit)
        payloads = [build_payload(r) for r in records]
        remote_count = sum(1 for p in payloads if p["transport"] != "stdio")
        print(
            f"Selected {len(payloads)} servers "
            f"({remote_count} remote, {len(payloads) - remote_count} stdio-gated)"
        )

    if args.dry_run:
        with open(args.manifest, "w", encoding="utf-8") as f:
            yaml.safe_dump(payloads, f, sort_keys=False, allow_unicode=True)
        print(f"[dry-run] wrote {len(payloads)} payloads to {args.manifest}")
        return

    print(f"Importing into {args.base_url} ...")
    if import_payloads(args.base_url, key, payloads):
        sys.exit(1)


if __name__ == "__main__":
    main()

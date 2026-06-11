#!/usr/bin/env python3
"""Bulk-register the agents defined in a LiteLLM `agent_list` config into a
running LiteLLM proxy via the Agent Hub API (POST /v1/agents).

This is how you populate the prod proxy (e.g. tokenhub.xcity.one) so the agents
show up in xct-chat's Agent Marketplace. The agent cards are display-ready;
actual A2A invocation additionally requires the agent gateway referenced by each
card's `url` to be reachable from the proxy.

Usage:
    LITELLM_ADMIN_KEY=sk-... \\
    python tools/register_agents.py \\
        --config dev_config_with_agents.yaml \\
        --base-url https://tokenhub.xcity.one \\
        [--public] [--dry-run]

Idempotent: an agent that already exists (409 / duplicate) is reported as
"skipped" rather than failing the run.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import yaml


def load_agents(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    agents = cfg.get("agent_list") or []
    if not isinstance(agents, list):
        sys.exit(f"'agent_list' in {config_path} is not a list")
    return agents


def post_agent(base_url, key, agent, timeout=20):
    url = f"{base_url.rstrip('/')}/v1/agents"
    body = json.dumps(agent).encode("utf-8")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dev_config_with_agents.yaml")
    ap.add_argument(
        "--base-url",
        default=os.environ.get("LITELLM_BASEURL", "https://tokenhub.xcity.one"),
    )
    ap.add_argument(
        "--public", action="store_true", help="force make_public: true on every agent"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = (
        os.environ.get("LITELLM_ADMIN_KEY")
        or os.environ.get("LITELLM_MASTER_KEY")
        or os.environ.get("PROXY_MASTER_KEY")
        or os.environ.get("LITELLM_API_KEY")
    )
    if not key and not args.dry_run:
        sys.exit(
            "Set LITELLM_ADMIN_KEY / LITELLM_MASTER_KEY (proxy admin key) in the environment."
        )

    agents = load_agents(args.config)
    print(f"Loaded {len(agents)} agents from {args.config}; target {args.base_url}")

    created = skipped = failed = 0
    for i, agent in enumerate(agents, 1):
        name = agent.get("agent_name") or agent.get("agent_card_params", {}).get(
            "name", f"#{i}"
        )
        if args.public:
            agent.setdefault("litellm_params", {})["make_public"] = True
        if args.dry_run:
            print(f"  [dry-run] would POST {name}")
            continue
        status, text = post_agent(args.base_url, key, agent)
        if status in (200, 201):
            created += 1
            print(f"  ✓ {name}")
        elif status in (409,) or "exist" in text.lower() or "duplicate" in text.lower():
            skipped += 1
            print(f"  = {name} (already exists)")
        else:
            failed += 1
            print(f"  ✗ {name} -> {status} {text[:200]}")

    print(
        f"\nDone. created={created} skipped={skipped} failed={failed} total={len(agents)}"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

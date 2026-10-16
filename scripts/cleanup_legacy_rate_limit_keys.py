#!/usr/bin/env python3
"""Find and optionally remove legacy Redis rate-limit keys.

The rate-limit key format changed from a local-clock ``HH-MM`` suffix/prefix to
versioned UTC epoch-minute keys. This maintenance command finds only the old
formats. It is a dry-run by default; deleting keys requires both ``--apply``
and the explicit confirmation token.

Usage:
    python scripts/cleanup_legacy_rate_limit_keys.py [options]

The Redis connection is read from LiteLLM's normal REDIS_* configuration. A
namespace can be supplied to narrow the SCAN. Apply mode requires a namespace
and removes only keys with a permanent TTL (``TTL=-1``), so a dry-run is the
only unscoped mode and active finite-TTL counters are retained. redis-py
RedisCluster's ``scan_iter`` is used as-is, so cluster scans remain node-aware.
Deletions are sent one key at a time to avoid a cross-slot multi-key command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

CONFIRMATION_TOKEN = "DELETE_LEGACY_RATE_LIMIT_KEYS"
_COUNTER_KINDS = frozenset({"tpm", "rpm", "itpm", "otpm"})
_WINDOW_RE = re.compile(r"^(?:[01][0-9]|2[0-3])-[0-5][0-9]$")


@dataclass
class CleanupReport:
    """Counters returned by a legacy-key scan."""

    scanned: int = 0
    candidates: int = 0
    permanent: int = 0
    finite: int = 0
    unknown_ttl: int = 0
    ttl_errors: int = 0
    deleted: int = 0
    failed: int = 0
    by_kind: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "candidates": self.candidates,
            "permanent": self.permanent,
            "finite": self.finite,
            "unknown_ttl": self.unknown_ttl,
            "ttl_errors": self.ttl_errors,
            "deleted": self.deleted,
            "failed": self.failed,
            "by_kind": dict(sorted(self.by_kind.items())),
        }


def _normalize_namespace(namespace: str | None) -> str | None:
    if namespace is None:
        return None
    if not namespace or namespace != namespace.strip() or any(char.isspace() for char in namespace):
        raise ValueError("namespace must be non-empty and must not contain whitespace")
    normalized = namespace.rstrip(":")
    if not normalized:
        raise ValueError("namespace must contain at least one non-colon character")
    return normalized


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _is_legacy_window(value: str) -> bool:
    return _WINDOW_RE.fullmatch(value) is not None


def legacy_key_kind(key: str | bytes, namespace: str | None = None) -> str | None:
    """Return the legacy key kind, or ``None`` for a v2/unrelated key.

    Counter keys are recognized by their final ``:<kind>:HH-MM`` segments,
    which covers the historical global-router and model-group variants. The
    dynamic limiter used ``HH-MM:<model>`` and is recognized separately.
    """

    normalized_namespace = _normalize_namespace(namespace)
    key_text = _as_text(key)
    if normalized_namespace is not None:
        namespace_prefix = f"{normalized_namespace}:"
        if not key_text.startswith(namespace_prefix):
            return None
        key_text = key_text[len(namespace_prefix) :]

    parts = key_text.split(":")
    if len(parts) >= 3 and parts[-2] in _COUNTER_KINDS and _is_legacy_window(parts[-1]):
        return parts[-2]
    if len(parts) >= 2 and _is_legacy_window(parts[0]) and ":".join(parts[1:]):
        return "dynamic"
    return None


def _delete_one(client: object, key: str) -> None:
    """Delete one key so a Cluster client never receives a cross-slot batch."""

    unlink = getattr(client, "unlink", None)
    if callable(unlink):
        unlink(key)
        return
    delete = getattr(client, "delete", None)
    if not callable(delete):
        raise AttributeError("Redis client has neither unlink nor delete")
    delete(key)


def cleanup_legacy_keys(
    client: object,
    *,
    namespace: str | None = None,
    count: int = 1000,
    apply: bool = False,
) -> CleanupReport:
    """Scan a Redis client and optionally remove unique legacy rate-limit keys."""

    normalized_namespace = _normalize_namespace(namespace)
    if count <= 0:
        raise ValueError("count must be positive")
    if apply and normalized_namespace is None:
        raise ValueError("namespace is required when apply=True")

    scan_iter = getattr(client, "scan_iter", None)
    ttl = getattr(client, "ttl", None)
    if not callable(scan_iter) or not callable(ttl):
        raise AttributeError("Redis client must provide scan_iter and ttl")

    match = f"{normalized_namespace}:*" if normalized_namespace is not None else "*"
    report = CleanupReport()
    seen_candidates: set[str] = set()
    for raw_key in scan_iter(match=match, count=count):
        report.scanned += 1
        key = _as_text(raw_key)
        kind = legacy_key_kind(key, normalized_namespace)
        if kind is None or key in seen_candidates:
            continue

        seen_candidates.add(key)
        report.candidates += 1
        report.by_kind[kind] += 1
        try:
            ttl_value = ttl(key)
        except Exception:
            report.ttl_errors += 1
            continue

        is_permanent = False
        if ttl_value == -1:
            report.permanent += 1
            is_permanent = True
        elif isinstance(ttl_value, int) and ttl_value >= 0:
            report.finite += 1
        else:
            report.unknown_ttl += 1

        if apply and is_permanent:
            try:
                _delete_one(client, key)
            except Exception:
                report.failed += 1
            else:
                report.deleted += 1

    return report


def _connection_kwargs(args: argparse.Namespace) -> dict[str, object]:
    if args.host is None and (args.port is not None or args.db is not None):
        raise ValueError("--host is required when using --port or --db")
    if args.host is not None and args.port is None:
        raise ValueError("--port is required when using --host")

    values = {
        "host": args.host,
        "port": args.port,
        "db": args.db,
    }
    return {name: value for name, value in values.items() if value is not None}


def _connect(args: argparse.Namespace) -> object:
    # Importing at runtime keeps the unit-testable key classifier independent of
    # LiteLLM's optional provider imports, while reusing its masked REDIS_* auth
    # and Cluster/Sentinel connection handling for the real command.
    from litellm._redis import get_redis_client

    connection_kwargs = _connection_kwargs(args)
    if connection_kwargs and (os.getenv("REDIS_CLUSTER_NODES") or os.getenv("REDIS_SENTINEL_NODES")):
        raise ValueError("explicit host overrides cannot be combined with Redis Cluster or Sentinel configuration")
    return get_redis_client(**connection_kwargs)


def _write_report(report: CleanupReport, *, apply: bool, json_output: bool) -> None:
    if json_output:
        payload = report.as_dict()
        payload["mode"] = "apply" if apply else "dry-run"
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return

    mode = "apply" if apply else "dry-run"
    summary = report.as_dict()
    sys.stdout.write(
        f"mode={mode} scanned={summary['scanned']} candidates={summary['candidates']} "
        f"permanent={summary['permanent']} finite={summary['finite']} "
        f"unknown_ttl={summary['unknown_ttl']} ttl_errors={summary['ttl_errors']} "
        f"deleted={summary['deleted']} failed={summary['failed']}\n"
    )
    kinds = ", ".join(f"{kind}={amount}" for kind, amount in sorted(report.by_kind.items()))
    if kinds:
        sys.stdout.write(f"candidate_kinds={kinds}\n")
    if not apply:
        sys.stdout.write("dry-run: no keys changed\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", help="only inspect keys beginning with NAMESPACE:")
    parser.add_argument("--count", type=int, default=1000, help="Redis SCAN count hint (default: 1000)")
    parser.add_argument("--host", help="Redis host override; use with --port")
    parser.add_argument("--port", type=int, help="Redis port override; use with --host")
    parser.add_argument("--db", type=int, help="Redis database override; use with --host")
    parser.add_argument("--json", action="store_true", dest="json_output", help="emit machine-readable JSON")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete matching keys; requires --confirm " + CONFIRMATION_TOKEN,
    )
    parser.add_argument("--confirm", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.apply and args.confirm != CONFIRMATION_TOKEN:
        parser.error("--apply requires --confirm " + CONFIRMATION_TOKEN)

    try:
        report = cleanup_legacy_keys(
            _connect(args),
            namespace=args.namespace,
            count=args.count,
            apply=args.apply,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        # Do not echo connection details: a configured URL may contain credentials.
        sys.stderr.write(f"ERROR: Redis legacy-key cleanup failed ({type(exc).__name__})\n")
        return 1

    _write_report(report, apply=args.apply, json_output=args.json_output)
    return 1 if report.failed or report.ttl_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

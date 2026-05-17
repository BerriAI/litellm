#!/usr/bin/env python3
"""Flush the bloated image-edit VCR cassettes from the cassette Redis.

Run this **once** after merging the multipart-boundary stabilization
PR. The pre-fix cassettes for the async image-edit tests have
accumulated >50 episodes (random multipart boundary on every run +
``record_mode="new_episodes"`` = monotonic growth), so the persister
refuses to save updates -- meaning every CI run after the fix would
still try to re-record against the stale 51-entry cassette, hit
``MAX_EPISODES_PER_CASSETTE`` again, get refused, and re-bill the live
provider.

Deleting these keys forces the next CI run to record a clean cassette
under the new fixed-boundary + raw-bytes fixtures (1 episode per
unique call), after which the 24-hour TTL replay loop kicks in
normally.

Scope is intentionally narrow:
  * Only ``tests/image_gen_tests/test_image_edits/*`` cassette keys
    are touched. Image-*generation* cassettes (TestOpenAIGPTImage1
    etc.) are unaffected -- they were already in the VCR HIT state.
  * Lists every match in dry-run mode before deleting anything so the
    operator can confirm the impact.

Usage:
    CASSETTE_REDIS_URL=redis://... \
        uv run python scripts/flush_image_edit_vcr_cassettes.py --dry-run

    CASSETTE_REDIS_URL=redis://... \
        uv run python scripts/flush_image_edit_vcr_cassettes.py --yes

``CASSETTE_REDIS_URL`` is the same env var the persister reads at CI
start (see ``tests/_vcr_redis_persister.py``).
"""

from __future__ import annotations

import argparse
import os
import sys

import redis


CASSETTE_REDIS_URL_ENV = "CASSETTE_REDIS_URL"
REDIS_KEY_PREFIX = "litellm:vcr:cassette:"
TARGET_KEY_PATTERN = f"{REDIS_KEY_PREFIX}tests/image_gen_tests/test_image_edits/*"


def _build_client(url: str) -> redis.Redis:
    return redis.Redis.from_url(
        url,
        socket_timeout=10,
        socket_connect_timeout=10,
        decode_responses=False,
    )


def _scan_matching_keys(client: redis.Redis, pattern: str) -> list[bytes]:
    return sorted(client.scan_iter(match=pattern, count=500))


def _delete_keys(client: redis.Redis, keys: list[bytes]) -> int:
    if not keys:
        return 0
    # Batch into chunks so a single DEL call does not exceed the
    # server's argument-count or buffer limits on large key sets.
    deleted = 0
    chunk_size = 200
    for start in range(0, len(keys), chunk_size):
        batch = keys[start : start + chunk_size]
        deleted += int(client.delete(*batch))
    return deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete the matched keys. Without this flag the script "
        "runs in dry-run mode and only lists what would be deleted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matched keys without deleting (default behaviour when "
        "--yes is omitted; kept as an explicit flag for clarity).",
    )
    parser.add_argument(
        "--pattern",
        default=TARGET_KEY_PATTERN,
        help=f"Override the SCAN match pattern. Default: {TARGET_KEY_PATTERN}",
    )
    args = parser.parse_args(argv)

    url = os.environ.get(CASSETTE_REDIS_URL_ENV)
    if not url:
        print(
            f"error: {CASSETTE_REDIS_URL_ENV} is not set. Set it to the "
            "cassette Redis URL (same URL the persister reads in CI).",
            file=sys.stderr,
        )
        return 2

    client = _build_client(url)
    try:
        client.ping()
    except redis.RedisError as exc:
        print(f"error: cannot reach cassette Redis: {exc}", file=sys.stderr)
        return 2

    matches = _scan_matching_keys(client, args.pattern)
    print(f"matched {len(matches)} key(s) under pattern: {args.pattern}")
    for key in matches:
        print(f"  {key.decode('utf-8', errors='replace')}")

    if not matches:
        return 0

    if not args.yes:
        print("\ndry run -- pass --yes to actually delete these keys.")
        return 0

    deleted = _delete_keys(client, matches)
    print(f"\ndeleted {deleted} key(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

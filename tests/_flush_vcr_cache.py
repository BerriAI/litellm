from __future__ import annotations

import os
import sys

import redis

from tests._vcr_redis_persister import CASSETTE_REDIS_URL_ENV, _redis_url_from_env

PREFIX = "litellm:vcr:cassette:"
SCAN_BATCH = 500


def _client() -> redis.Redis:
    url = _redis_url_from_env()
    if not url:
        sys.exit(f"Set {CASSETTE_REDIS_URL_ENV} to flush the VCR cache")
    return redis.Redis.from_url(
        url,
        socket_timeout=5,
        socket_connect_timeout=5,
        decode_responses=False,
    )


def main() -> None:
    client = _client()
    deleted = 0
    pipeline = client.pipeline(transaction=False)
    pending = 0
    for key in client.scan_iter(match=f"{PREFIX}*", count=SCAN_BATCH):
        pipeline.delete(key)
        pending += 1
        if pending >= SCAN_BATCH:
            deleted += sum(pipeline.execute())
            pipeline = client.pipeline(transaction=False)
            pending = 0
    if pending:
        deleted += sum(pipeline.execute())
    print(f"Deleted {deleted} VCR cassette key(s) under {PREFIX!r}")


if __name__ == "__main__":
    main()

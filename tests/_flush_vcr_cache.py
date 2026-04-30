from __future__ import annotations

import os
import sys

import redis

PREFIX = "litellm:vcr:cassette:"
SCAN_BATCH = 500


def _client() -> redis.Redis:
    host = os.environ.get("REDIS_HOST")
    if not host:
        sys.exit("REDIS_HOST is not set; cannot flush VCR cache")
    return redis.Redis(
        host=host,
        port=int(os.environ.get("REDIS_PORT", 6379)),
        password=os.environ.get("REDIS_PASSWORD") or None,
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

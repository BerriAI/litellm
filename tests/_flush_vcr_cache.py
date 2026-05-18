from __future__ import annotations

import os
import shutil
import sys

from tests._vcr_redis_persister import (
    CASSETTE_LOCAL_CACHE_DIR_ENV,
    CASSETTE_REDIS_URL_ENV,
    CASSETTE_S3_BUCKET_ENV,
    CASSETTE_S3_ENDPOINT_ENV,
    CASSETTE_S3_REGION_ENV,
    REDIS_KEY_PREFIX,
    _local_cache_dir_from_env,
    _maybe_build_redis_client,
    _redis_url_from_env,
    _s3_bucket_from_env,
)

PREFIX = REDIS_KEY_PREFIX
SCAN_BATCH = 500


def _flush_redis() -> int:
    client = _maybe_build_redis_client()
    if client is None:
        return 0
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
    print(f"Deleted {deleted} VCR cassette key(s) from Redis under {PREFIX!r}")
    return deleted


def _flush_s3() -> int:
    bucket = _s3_bucket_from_env()
    if not bucket:
        return 0
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        sys.exit(f"boto3 is required to flush the S3 cassette cache: {exc}")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get(CASSETTE_S3_ENDPOINT_ENV) or None,
        region_name=os.environ.get(CASSETTE_S3_REGION_ENV) or "auto",
        config=Config(s3={"addressing_style": "path"}),
    )
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=PREFIX):
        contents = page.get("Contents") or []
        if not contents:
            continue
        objects = [{"Key": obj["Key"]} for obj in contents]
        # delete_objects accepts up to 1000 keys per request.
        for i in range(0, len(objects), 1000):
            chunk = objects[i : i + 1000]
            resp = client.delete_objects(Bucket=bucket, Delete={"Objects": chunk})
            deleted += len(resp.get("Deleted", []))
    print(f"Deleted {deleted} VCR cassette object(s) from S3 bucket {bucket!r}")
    return deleted


def _flush_local() -> int:
    cache_dir = _local_cache_dir_from_env()
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0
    count = 0
    for root, _dirs, files in os.walk(cache_dir):
        count += len(files)
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"Deleted {count} VCR cassette file(s) from local cache {cache_dir!r}")
    return count


def main() -> None:
    if not (
        _redis_url_from_env() or _s3_bucket_from_env() or _local_cache_dir_from_env()
    ):
        sys.exit(
            f"Set {CASSETTE_REDIS_URL_ENV}, {CASSETTE_S3_BUCKET_ENV}, or "
            f"{CASSETTE_LOCAL_CACHE_DIR_ENV} to flush a VCR cache."
        )
    _flush_redis()
    _flush_s3()
    _flush_local()


if __name__ == "__main__":
    main()

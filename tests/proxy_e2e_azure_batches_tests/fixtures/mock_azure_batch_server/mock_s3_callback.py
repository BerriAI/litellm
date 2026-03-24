"""
Mock S3 callback receiver for testing LiteLLM S3 callbacks.

This module provides S3-compatible endpoints that capture callback data
sent by LiteLLM's s3_v2 callback handler after batch completion.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3CallbackRecord(BaseModel):
    key: str
    bucket: str
    content: Dict[str, Any]
    timestamp: int
    content_type: Optional[str] = None


callback_storage: List[S3CallbackRecord] = []


def setup_s3_callback_routes(app: FastAPI):
    @app.put("/{bucket}/{key:path}")
    async def s3_put_object(bucket: str, key: str, request: Request):
        content_type = request.headers.get("content-type", "application/json")
        body = await request.body()

        try:
            content = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            content = {"raw": body.decode("utf-8", errors="replace")}

        record = S3CallbackRecord(
            key=key,
            bucket=bucket,
            content=content,
            timestamp=int(time.time()),
            content_type=content_type,
        )
        callback_storage.append(record)

        logger.info(f"S3 callback received: bucket={bucket}, key={key}")
        logger.debug(f"Callback content: {json.dumps(content, indent=2)[:500]}")

        return {
            "ETag": f'"{hash(body)}"',
            "VersionId": None,
        }

    @app.get("/mock-s3/callbacks")
    async def list_callbacks(
        bucket: Optional[str] = None,
        key_prefix: Optional[str] = None,
        limit: int = 100,
    ):
        results = callback_storage

        if bucket:
            results = [r for r in results if r.bucket == bucket]

        if key_prefix:
            results = [r for r in results if r.key.startswith(key_prefix)]

        return {
            "count": len(results),
            "callbacks": [r.model_dump() for r in results[-limit:]],
        }

    @app.get("/mock-s3/callbacks/count")
    async def count_callbacks(bucket: Optional[str] = None):
        if bucket:
            count = sum(1 for r in callback_storage if r.bucket == bucket)
        else:
            count = len(callback_storage)

        return {"count": count}

    @app.get("/mock-s3/callbacks/latest")
    async def get_latest_callback():
        if not callback_storage:
            return {"callback": None}
        return {"callback": callback_storage[-1].model_dump()}

    @app.delete("/mock-s3/callbacks")
    async def clear_callbacks():
        count = len(callback_storage)
        callback_storage.clear()
        logger.info(f"Cleared {count} S3 callbacks")
        return {"cleared": count}

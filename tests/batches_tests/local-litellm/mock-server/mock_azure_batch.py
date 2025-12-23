import asyncio
import io
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure-like credential validation
# Set MOCK_REQUIRE_CREDENTIALS=true to enforce credential checks (like real Azure)
REQUIRE_CREDENTIALS = os.environ.get("MOCK_REQUIRE_CREDENTIALS", "true").lower() == "true"
VALID_API_KEYS = {"fake-key", "sk-1234", "test-key"}  # Accept these API keys

api_key_header = APIKeyHeader(name="api-key", auto_error=False)
auth_header = APIKeyHeader(name="Authorization", auto_error=False)


def validate_credentials(
    api_key: Optional[str] = Depends(api_key_header),
    authorization: Optional[str] = Depends(auth_header),
):
    """
    Validate Azure-style credentials.
    Azure accepts either:
    - api-key header
    - Authorization: Bearer <token> header
    """
    if not REQUIRE_CREDENTIALS:
        return True
    
    # Check api-key header
    if api_key:
        if api_key in VALID_API_KEYS:
            return True
        logger.warning(f"Invalid api-key provided: {api_key[:8]}...")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "401",
                    "message": "Access denied due to invalid subscription key or wrong API endpoint. "
                              "Make sure to provide a valid key for an active subscription and use a "
                              "correct regional API endpoint for your resource."
                }
            }
        )
    
    # Check Authorization header (Bearer token)
    if authorization:
        if authorization.startswith("Bearer "):
            # Accept any bearer token for mock purposes
            return True
        logger.warning(f"Invalid Authorization header format")
    
    # No credentials provided
    logger.warning("No credentials provided in request")
    raise HTTPException(
        status_code=401,
        detail={
            "error": {
                "code": "401",
                "message": "Missing credentials. Please pass one of `api_key`, `azure_ad_token`, "
                          "`azure_ad_token_provider`, or the `AZURE_OPENAI_API_KEY` or "
                          "`AZURE_OPENAI_AD_TOKEN` environment variables."
            }
        }
    )


class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str = "processed"
    status_details: Optional[str] = None
    expires_at: Optional[int] = None


class BatchObject(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    errors: Optional[Dict] = None
    input_file_id: str
    completion_window: str
    status: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: Optional[Dict[str, int]] = None
    metadata: Optional[Dict] = None


class BatchListResponse(BaseModel):
    object: str = "list"
    data: List[Dict]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


file_storage: Dict[str, Dict] = {}
batch_storage: Dict[str, BatchObject] = {}
batch_results: Dict[str, List[Dict]] = {}

PROCESSING_DELAY_SECONDS = float(1)
VALIDATING_DELAY_SECONDS = float(3)


async def process_batch(batch_id: str):
    logger.info(f"Starting batch processing for {batch_id}")
    try:
        batch = batch_storage[batch_id]

        await asyncio.sleep(VALIDATING_DELAY_SECONDS)
        batch.status = "in_progress"
        batch.in_progress_at = int(time.time())
        logger.info(f"Batch {batch_id} status: in_progress")

        await process_batch_requests(batch_id)
        await asyncio.sleep(PROCESSING_DELAY_SECONDS)

        batch.status = "finalizing"
        batch.finalizing_at = int(time.time())
        logger.info(f"Batch {batch_id} status: finalizing")
        await asyncio.sleep(PROCESSING_DELAY_SECONDS)

        await create_output_file(batch_id)

        batch.status = "completed"
        batch.completed_at = int(time.time())
        logger.info(f"Batch {batch_id} status: completed")

    except Exception as e:
        logger.error(f"Batch {batch_id} failed: {e}")
        batch = batch_storage[batch_id]
        batch.status = "failed"
        batch.failed_at = int(time.time())
        batch.errors = {
            "object": "list",
            "data": [{"code": "processing_error", "message": str(e)}],
        }


async def process_batch_requests(batch_id: str):
    batch = batch_storage[batch_id]
    input_file = file_storage[batch.input_file_id]

    requests = []
    for line in input_file["content"].split("\n"):
        if line.strip():
            try:
                requests.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON line in batch {batch_id}: {e}")

    logger.info(f"Batch {batch_id} has {len(requests)} requests")

    results = []
    failed_count = 0
    for req in requests:
        result = await process_single_request(req)
        if result.get("error"):
            failed_count += 1
        results.append(result)

    batch_results[batch_id] = results
    batch.request_counts = {
        "total": len(requests),
        "completed": len(results) - failed_count,
        "failed": failed_count,
    }


async def process_single_request(request_data: Dict) -> Dict:
    custom_id = request_data.get("custom_id")
    url = request_data.get("url", "/v1/chat/completions")
    body = request_data.get("body", {})

    if "/chat/completions" in url:
        response_body = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "gpt-4o"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Mock batch response."},
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        status_code = 200
    else:
        response_body = {"error": {"message": f"Unsupported endpoint: {url}"}}
        status_code = 400

    return {
        "id": f"batch_req_{uuid.uuid4().hex[:12]}",
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "request_id": f"req_{uuid.uuid4().hex[:12]}",
            "body": response_body,
        },
        "error": None,
    }


async def create_output_file(batch_id: str):
    results = batch_results.get(batch_id, [])
    output_lines = [json.dumps(result) for result in results]
    output_content = "\n".join(output_lines)

    output_file_id = f"file-batch-output-{uuid.uuid4().hex[:12]}"
    file_storage[output_file_id] = {
        "content": output_content,
        "filename": f"batch_output_{batch_id}.jsonl",
        "purpose": "batch_output",
        "bytes": len(output_content.encode()),
        "created_at": int(time.time()),
    }

    batch = batch_storage[batch_id]
    batch.output_file_id = output_file_id
    logger.info(f"Created output file {output_file_id} for batch {batch_id}")


def validate_batch_input(content: str) -> tuple[bool, str, List[Dict]]:
    requests = []
    custom_ids = set()

    lines = content.strip().split("\n")
    if not lines or all(not line.strip() for line in lines):
        return False, "empty_batch", []

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            return False, "invalid_json_line", []

        for field in ["custom_id", "method", "url", "body"]:
            if field not in req:
                return False, "invalid_request", []

        if req["custom_id"] in custom_ids:
            return False, "duplicate_custom_id", []
        custom_ids.add(req["custom_id"])

        requests.append(req)

    if len(requests) > 100000:
        return False, "too_many_tasks", []

    return True, "", requests


def setup_batch_routes(app: FastAPI):
    # Files endpoints (OpenAI and Azure paths)
    # All endpoints require credentials (like real Azure)
    @app.post("/openai/v1/files")
    @app.post("/openai/files")
    @app.post("/v1/files")
    @app.post("/files")
    async def create_file(request: Request, _=Depends(validate_credentials)):
        form = await request.form()
        logger.info(f"File upload form fields: {list(form.keys())}")

        file: UploadFile = form.get("file")
        purpose: str = form.get("purpose", "batch")

        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        logger.info(f"Uploading file: {file.filename}, purpose: {purpose}")

        content = await file.read()
        content_str = content.decode("utf-8")

        file_id = f"file-{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        expires_at = None
        expires_after_seconds = form.get("expires_after[seconds]")
        if expires_after_seconds:
            try:
                seconds = int(expires_after_seconds)
                logger.info(f"expires_after[seconds] = {seconds}")
                if seconds < 259200 or seconds > 2592000:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "code": "invalidPayload",
                                "message": "Value for Seconds must be between 259200 and 2592000.",
                            },
                        },
                    )
                expires_at = created_at + seconds
                logger.info(f"Calculated expires_at: {expires_at}")
            except ValueError as e:
                logger.warning(f"Failed to parse expires_after[seconds]: {e}")

        file_storage[file_id] = {
            "content": content_str,
            "filename": file.filename or "batch_input.jsonl",
            "purpose": purpose,
            "bytes": len(content),
            "created_at": created_at,
            "expires_at": expires_at,
        }

        logger.info(f"Created file {file_id}, expires_at={expires_at}")
        return FileObject(
            id=file_id,
            bytes=len(content),
            created_at=created_at,
            filename=file.filename or "batch_input.jsonl",
            purpose=purpose,
            expires_at=expires_at,
        ).model_dump()

    @app.get("/openai/v1/files/{file_id}")
    @app.get("/openai/files/{file_id}")
    @app.get("/v1/files/{file_id}")
    @app.get("/files/{file_id}")
    async def get_file(file_id: str, _=Depends(validate_credentials)):
        logger.info(f"Getting file: {file_id}")
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")

        file_data = file_storage[file_id]
        return FileObject(
            id=file_id,
            bytes=file_data["bytes"],
            created_at=file_data["created_at"],
            filename=file_data["filename"],
            purpose=file_data["purpose"],
            expires_at=file_data.get("expires_at"),
        ).model_dump()

    @app.get("/openai/v1/files/{file_id}/content")
    @app.get("/openai/files/{file_id}/content")
    @app.get("/v1/files/{file_id}/content")
    @app.get("/files/{file_id}/content")
    async def get_file_content(file_id: str, _=Depends(validate_credentials)):
        logger.info(f"Getting file content: {file_id}")
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")

        file_data = file_storage[file_id]
        content = file_data["content"]

        return StreamingResponse(
            io.StringIO(content),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={file_data['filename']}",
            },
        )

    @app.delete("/openai/v1/files/{file_id}")
    @app.delete("/openai/files/{file_id}")
    @app.delete("/v1/files/{file_id}")
    @app.delete("/files/{file_id}")
    async def delete_file(file_id: str, _=Depends(validate_credentials)):
        logger.info(f"Deleting file: {file_id}")
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")

        del file_storage[file_id]
        return {"id": file_id, "object": "file", "deleted": True}

    @app.get("/openai/v1/files")
    @app.get("/openai/files")
    @app.get("/v1/files")
    @app.get("/files")
    async def list_files(
        purpose: Optional[str] = None,
        limit: int = Query(10000, le=10000),
        _=Depends(validate_credentials),
    ):
        logger.info(f"Listing files, purpose: {purpose}, limit: {limit}")
        files = []
        for file_id, file_data in file_storage.items():
            if purpose is None or file_data.get("purpose") == purpose:
                files.append(
                    FileObject(
                        id=file_id,
                        bytes=file_data["bytes"],
                        created_at=file_data["created_at"],
                        filename=file_data["filename"],
                        purpose=file_data["purpose"],
                        expires_at=file_data.get("expires_at"),
                    ).model_dump(),
                )
        return {"object": "list", "data": files[:limit]}

    # Batches endpoints (OpenAI and Azure paths)
    @app.post("/openai/v1/batches")
    @app.post("/openai/batches")
    @app.post("/v1/batches")
    @app.post("/batches")
    async def create_batch(request_data: dict, _=Depends(validate_credentials)):
        input_file_id = request_data.get("input_file_id")
        endpoint = request_data.get("endpoint", "/v1/chat/completions")
        completion_window = request_data.get("completion_window", "24h")
        metadata = request_data.get("metadata", {})
        output_expires_after = request_data.get("output_expires_after")

        logger.info(
            f"Creating batch with input_file: {input_file_id}, endpoint: {endpoint}, output_expires_after: {output_expires_after}",
        )

        if not input_file_id or input_file_id not in file_storage:
            raise HTTPException(status_code=400, detail="Input file not found")

        input_file = file_storage[input_file_id]
        is_valid, error_code, _ = validate_batch_input(input_file["content"])
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": error_code,
                        "message": f"Validation failed: {error_code}",
                    },
                },
            )

        batch_id = f"batch_{uuid.uuid4()}"
        created_at = int(time.time())

        if output_expires_after:
            seconds = (
                output_expires_after.get("seconds", 0)
                if isinstance(output_expires_after, dict)
                else 0
            )
            expires_at = created_at + seconds
            logger.info(
                f"Using output_expires_after: {seconds}s, expires_at: {expires_at}",
            )
        elif completion_window == "24h":
            expires_at = created_at + (24 * 60 * 60)
        else:
            expires_at = created_at + (24 * 60 * 60)

        batch = BatchObject(
            id=batch_id,
            endpoint=endpoint,
            input_file_id=input_file_id,
            completion_window=completion_window,
            status="validating",
            created_at=created_at,
            expires_at=expires_at,
            request_counts={"total": 0, "completed": 0, "failed": 0},
            metadata=metadata,
        )

        batch_storage[batch_id] = batch
        logger.info(f"Created batch {batch_id}")

        asyncio.create_task(process_batch(batch_id))

        return batch.model_dump()

    @app.get("/openai/v1/batches/{batch_id}")
    @app.get("/openai/batches/{batch_id}")
    @app.get("/v1/batches/{batch_id}")
    @app.get("/batches/{batch_id}")
    async def get_batch(batch_id: str, _=Depends(validate_credentials)):
        logger.info(f"Getting batch: {batch_id}")
        if batch_id not in batch_storage:
            raise HTTPException(status_code=404, detail="Batch not found")

        return batch_storage[batch_id].model_dump()

    @app.get("/openai/v1/batches")
    @app.get("/openai/batches")
    @app.get("/v1/batches")
    @app.get("/batches")
    async def list_batches(
        after: Optional[str] = Query(None),
        limit: int = Query(20, le=100),
        _=Depends(validate_credentials),
    ):
        logger.info(f"Listing batches, after: {after}, limit: {limit}")
        batches = list(batch_storage.values())
        batches.sort(key=lambda x: x.created_at, reverse=True)

        if after:
            after_index = next((i for i, b in enumerate(batches) if b.id == after), -1)
            if after_index >= 0:
                batches = batches[after_index + 1 :]

        batches = batches[:limit]

        return BatchListResponse(
            data=[batch.model_dump() for batch in batches],
            first_id=batches[0].id if batches else None,
            last_id=batches[-1].id if batches else None,
            has_more=len(batches) == limit,
        ).model_dump()

    @app.post("/openai/v1/batches/{batch_id}/cancel")
    @app.post("/openai/batches/{batch_id}/cancel")
    @app.post("/v1/batches/{batch_id}/cancel")
    @app.post("/batches/{batch_id}/cancel")
    async def cancel_batch(batch_id: str, _=Depends(validate_credentials)):
        logger.info(f"Cancelling batch: {batch_id}")
        if batch_id not in batch_storage:
            raise HTTPException(status_code=404, detail="Batch not found")

        batch = batch_storage[batch_id]
        if batch.status in ["completed", "failed", "cancelled", "expired"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel batch in {batch.status} status",
            )

        batch.status = "cancelled"
        batch.cancelled_at = int(time.time())
        logger.info(f"Batch {batch_id} cancelled")

        return batch.model_dump()

    # Debug endpoints
    @app.get("/debug/batches")
    async def debug_list_batches():
        return {
            "batches": {
                batch_id: batch.model_dump()
                for batch_id, batch in batch_storage.items()
            },
            "files": {
                file_id: {k: v for k, v in data.items() if k != "content"}
                for file_id, data in file_storage.items()
            },
        }

    @app.post("/reset")
    @app.post("/debug/clear")
    async def reset_all():
        file_storage.clear()
        batch_storage.clear()
        batch_results.clear()
        logger.info("All data cleared")
        return {"message": "All data cleared"}

    @app.get("/debug/status")
    async def debug_status():
        return {
            "files_count": len(file_storage),
            "batches_count": len(batch_storage),
            "batch_statuses": {bid: b.status for bid, b in batch_storage.items()},
        }

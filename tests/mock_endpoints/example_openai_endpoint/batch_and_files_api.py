from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Dict, Literal, Optional
import uuid
import hashlib
from datetime import datetime

router = APIRouter()


# ============= Models =============

class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str


class BatchRequestCounts(BaseModel):
    total: int
    completed: int
    failed: int


class BatchObject(BaseModel):
    id: str
    object: Literal["batch"] = "batch"
    endpoint: str
    errors: Optional[Dict] = None
    input_file_id: str
    completion_window: str
    status: Literal["validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"]
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
    request_counts: Optional[BatchRequestCounts] = None
    metadata: Optional[Dict[str, str]] = None


class CreateBatchRequest(BaseModel):
    input_file_id: str
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"]
    completion_window: Literal["24h"]
    metadata: Optional[Dict[str, str]] = None
    status: str


# ============= Files Endpoints =============

@router.post("/files", response_model=FileObject)
async def create_file(
    file: UploadFile = File(...),
    purpose: str = Form(...)
):
    """
    Upload a file that can be used for batch processing.
    
    Compatible with: https://platform.openai.com/docs/api-reference/files/create
    """
    content = await file.read()
    
    # Generate consistent file ID based on filename and content
    content_hash = hashlib.md5(f"{file.filename}{len(content)}".encode()).hexdigest()[:8]
    file_id = f"file-{content_hash}"
    
    return FileObject(
        id=file_id,
        bytes=len(content),
        created_at=int(datetime.now().timestamp()),
        filename=file.filename or "uploaded_file",
        purpose=purpose,
        status="completed"
    )


@router.get("/files/{file_id}", response_model=FileObject)
async def retrieve_file(file_id: str):
    """
    Returns information about a specific file.
    
    Compatible with: https://platform.openai.com/docs/api-reference/files/retrieve
    """
    # Return stubbed file information
    return FileObject(
        id=file_id,
        bytes=1024,  # Stubbed file size
        created_at=1698768000,  # Stubbed timestamp
        filename="example_file.jsonl",
        purpose="batch",
        status="completed"
    )


@router.get("/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    """
    Returns the contents of the specified file.
    
    Compatible with: https://platform.openai.com/docs/api-reference/files/retrieve-contents
    """
    # Return stubbed file content
    stubbed_content = '{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello world"}]}}\n{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "How are you?"}]}}'
    return stubbed_content


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete a file.
    
    Compatible with: https://platform.openai.com/docs/api-reference/files/delete
    """
    # Return stubbed deletion response
    return {"id": file_id, "object": "file", "deleted": True}


# ============= Batches Endpoints =============

@router.post("/batches", response_model=BatchObject)
async def create_batch(request: CreateBatchRequest):
    """
    Creates and executes a batch from an uploaded file of requests.
    
    Compatible with: https://platform.openai.com/docs/api-reference/batch/create
    """
    # Generate consistent batch ID based on input file ID
    batch_hash = hashlib.md5(request.input_file_id.encode()).hexdigest()[:8]
    batch_id = f"batch_{batch_hash}"
    created_at = int(datetime.now().timestamp())
    
    # Return stubbed batch object
    return BatchObject(
        id=batch_id,
        object="batch",
        endpoint=request.endpoint,
        errors=None,
        input_file_id=request.input_file_id,
        completion_window=request.completion_window,
        status="completed",
        output_file_id=f"file-output-{batch_hash}",
        error_file_id=None,
        created_at=created_at,
        in_progress_at=created_at + 10,
        expires_at=created_at + 86400,  # 24 hours
        finalizing_at=created_at + 300,
        completed_at=created_at + 600,
        failed_at=None,
        expired_at=None,
        cancelling_at=None,
        cancelled_at=None,
        request_counts=BatchRequestCounts(total=2, completed=2, failed=0),
        metadata=request.metadata or {}
    )


@router.get("/batches/{batch_id}", response_model=BatchObject)
async def retrieve_batch(batch_id: str):
    """
    Retrieves a batch.
    
    Compatible with: https://platform.openai.com/docs/api-reference/batch/retrieve
    """
    # Extract hash from batch_id for consistent output file ID
    batch_hash = batch_id.split("_")[-1] if "_" in batch_id else "stubbed"
    created_at = 1698768000  # Stubbed timestamp
    
    # Return stubbed batch object
    return BatchObject(
        id=batch_id,
        object="batch",
        endpoint="/v1/chat/completions",
        errors=None,
        input_file_id=f"file-{batch_hash}",
        completion_window="24h",
        status="completed",
        output_file_id=f"file-output-{batch_hash}",
        error_file_id=None,
        created_at=created_at,
        in_progress_at=created_at + 10,
        expires_at=created_at + 86400,
        finalizing_at=created_at + 300,
        completed_at=created_at + 600,
        failed_at=None,
        expired_at=None,
        cancelling_at=None,
        cancelled_at=None,
        request_counts=BatchRequestCounts(total=2, completed=2, failed=0),
        metadata={}
    )


@router.post("/batches/{batch_id}/cancel", response_model=BatchObject)
async def cancel_batch(batch_id: str):
    """
    Cancels an in-progress batch.
    
    Compatible with: https://platform.openai.com/docs/api-reference/batch/cancel
    """
    # Extract hash from batch_id for consistent output file ID
    batch_hash = batch_id.split("_")[-1] if "_" in batch_id else "stubbed"
    created_at = 1698768000  # Stubbed timestamp
    cancelled_at = int(datetime.now().timestamp())
    
    # Return stubbed cancelled batch object
    return BatchObject(
        id=batch_id,
        object="batch",
        endpoint="/v1/chat/completions",
        errors=None,
        input_file_id=f"file-{batch_hash}",
        completion_window="24h",
        status="cancelled",
        output_file_id=None,
        error_file_id=None,
        created_at=created_at,
        in_progress_at=created_at + 10,
        expires_at=created_at + 86400,
        finalizing_at=None,
        completed_at=None,
        failed_at=None,
        expired_at=None,
        cancelling_at=cancelled_at - 5,
        cancelled_at=cancelled_at,
        request_counts=BatchRequestCounts(total=2, completed=0, failed=0),
        metadata={}
    )


@router.get("/batches")
async def list_batches(limit: int = 20, after: Optional[str] = None):
    """
    List your organization's batches.
    
    Compatible with: https://platform.openai.com/docs/api-reference/batch/list
    """
    # Return stubbed list of batches
    stubbed_batches = [
        BatchObject(
            id="batch_example1",
            object="batch",
            endpoint="/v1/chat/completions",
            errors=None,
            input_file_id="file-example1",
            completion_window="24h",
            status="completed",
            output_file_id="file-output-example1",
            error_file_id=None,
            created_at=1698768000,
            in_progress_at=1698768010,
            expires_at=1698854400,
            finalizing_at=1698768300,
            completed_at=1698768600,
            failed_at=None,
            expired_at=None,
            cancelling_at=None,
            cancelled_at=None,
            request_counts=BatchRequestCounts(total=5, completed=5, failed=0),
            metadata={}
        ),
        BatchObject(
            id="batch_example2",
            object="batch",
            endpoint="/v1/embeddings",
            errors=None,
            input_file_id="file-example2",
            completion_window="24h",
            status="in_progress",
            output_file_id=None,
            error_file_id=None,
            created_at=1698767000,
            in_progress_at=1698767010,
            expires_at=1698853400,
            finalizing_at=None,
            completed_at=None,
            failed_at=None,
            expired_at=None,
            cancelling_at=None,
            cancelled_at=None,
            request_counts=BatchRequestCounts(total=3, completed=1, failed=0),
            metadata={"project": "test"}
        )
    ]
    
    # Apply pagination logic to stubbed data
    if after:
        try:
            start_idx = next(i for i, b in enumerate(stubbed_batches) if b.id == after) + 1
            stubbed_batches = stubbed_batches[start_idx:]
        except StopIteration:
            pass
    
    stubbed_batches = stubbed_batches[:limit]
    
    return {
        "object": "list",
        "data": stubbed_batches,
        "has_more": False  # Stubbed - no more data
    }

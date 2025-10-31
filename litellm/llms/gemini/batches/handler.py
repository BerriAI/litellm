"""
Gemini Batch API Handler

Implementation for Google AI Studio Batch API endpoints
API Ref: https://ai.google.dev/gemini-api/docs/batch-mode
"""
import json
import os
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import LiteLLMBatch


class GeminiBatchesAPI:
    """
    Gemini Batch API methods:
    - retrieve_batch() - Retrieve batch job status and results
    """

    def __init__(self) -> None:
        pass

    def retrieve_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: Optional[str],
        api_key: str,
        timeout: Union[float, httpx.Timeout],
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        """
        Retrieve a Gemini batch job via Google AI Studio API
        
        Args:
            _is_async: Whether to use async call
            batch_id: Batch ID (e.g., "batches/abc123")
            api_base: API base URL
            api_key: Gemini API key
            timeout: Request timeout
            
        Returns:
            LiteLLMBatch object
        """
        # Construct the URL
        # GET https://generativelanguage.googleapis.com/v1beta/batches/{batch_id}?key={api_key}
        batch_url = f"{api_base}/v1beta/{batch_id}"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {
            "key": api_key,
        }
        
        if _is_async:
            return self._async_retrieve_batch(
                batch_url=batch_url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
        
        # Sync call
        sync_handler = _get_httpx_client()
        
        response = sync_handler.get(
            url=batch_url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        
        if response.status_code != 200:
            raise Exception(
                f"Error retrieving Gemini batch: {response.status_code} {response.text}"
            )
        
        _json_response = response.json()
        
        # Transform Gemini batch response to LiteLLMBatch format
        return self._transform_gemini_batch_response(_json_response)

    async def _async_retrieve_batch(
        self,
        batch_url: str,
        headers: Dict[str, str],
        params: Dict[str, str],
        timeout: Union[float, httpx.Timeout],
    ) -> LiteLLMBatch:
        """Async version of Gemini batch retrieval"""
        async_client = get_async_httpx_client()
        
        response = await async_client.get(
            url=batch_url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        
        if response.status_code != 200:
            raise Exception(
                f"Error retrieving Gemini batch: {response.status_code} {response.text}"
            )
        
        _json_response = response.json()
        
        # Transform Gemini batch response to LiteLLMBatch format
        return self._transform_gemini_batch_response(_json_response)

    def _transform_gemini_batch_response(self, gemini_response: dict) -> LiteLLMBatch:
        """
        Transform Gemini batch API response to LiteLLMBatch format
        
        Args:
            gemini_response: Raw Gemini batch API response
            
        Returns:
            LiteLLMBatch object
        """
        # Map Gemini states to OpenAI batch states
        state_mapping = {
            "JOB_STATE_PENDING": "validating",
            "JOB_STATE_RUNNING": "in_progress",
            "JOB_STATE_SUCCEEDED": "completed",
            "JOB_STATE_FAILED": "failed",
            "JOB_STATE_CANCELLED": "cancelled",
        }
        
        batch_state = gemini_response.get("state", {}).get("name", "JOB_STATE_PENDING")
        openai_status = state_mapping.get(batch_state, "validating")
        
        # Extract timestamps
        create_time = gemini_response.get("createTime", "")
        # Convert ISO timestamp to unix timestamp if present
        created_at = 0
        if create_time:
            from datetime import datetime as dt

            try:
                created_at = int(
                    dt.fromisoformat(create_time.replace("Z", "+00:00")).timestamp()
                )
            except:
                pass
        
        # Check if this is a file-based or inline batch
        dest = gemini_response.get("dest", {})
        output_file_id = None
        
        if "fileName" in dest:
            # File-based batch
            output_file_id = dest.get("fileName")
        
        # Create LiteLLMBatch object
        batch = LiteLLMBatch(
            id=gemini_response.get("name", ""),
            object="batch",
            endpoint="/v1/chat/completions",  # Gemini batches are chat completions
            errors=None,
            input_file_id=gemini_response.get("src", {}).get("fileName", ""),
            completion_window="24h",
            status=openai_status,
            output_file_id=output_file_id,
            error_file_id=None,
            created_at=created_at,
            in_progress_at=None,
            expires_at=None,
            finalizing_at=None,
            completed_at=None,
            failed_at=None,
            expired_at=None,
            cancelling_at=None,
            cancelled_at=None,
            request_counts=None,
            metadata=gemini_response.get("metadata", {}),
        )
        
        return batch


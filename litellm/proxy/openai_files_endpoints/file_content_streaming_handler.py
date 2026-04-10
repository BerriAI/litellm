from typing import Any, AsyncIterator, Dict, Optional, cast

from fastapi.responses import StreamingResponse

import litellm
from litellm.files.types import FileContentStreamingResult
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging

from litellm.proxy.openai_files_endpoints.common_utils import (
    prepare_data_with_credentials,
)


class FileContentStreamingHandler:
    @staticmethod
    def should_stream_file_content(
        *,
        custom_llm_provider: str,
        is_base64_unified_file_id: Any,
    ) -> bool:
        return (
            custom_llm_provider == "openai"
            and bool(is_base64_unified_file_id) is False
        )

    @staticmethod
    async def stream_file_content_with_logging(
        stream_iterator: AsyncIterator[bytes],
        proxy_logging_obj: ProxyLogging,
        user_api_key_dict: UserAPIKeyAuth,
        data: Dict[str, Any],
    ):
        try:
            async for chunk in stream_iterator:
                yield chunk
            await proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        except Exception as e:
            await proxy_logging_obj.post_call_failure_hook(
                user_api_key_dict=user_api_key_dict,
                original_exception=e,
                request_data=data,
            )
            raise
        finally:
            if hasattr(stream_iterator, "aclose"):
                await stream_iterator.aclose()  # type: ignore[attr-defined]

    @staticmethod
    async def get_streaming_file_content_response(
        *,
        custom_llm_provider: str,
        file_id: str,
        data: Dict[str, Any],
        should_route: bool,
        original_file_id: Optional[str],
        credentials: Optional[Dict[str, Any]],
        proxy_logging_obj: ProxyLogging,
        user_api_key_dict: UserAPIKeyAuth,
        version: str,
    ) -> StreamingResponse:
        effective_custom_llm_provider = custom_llm_provider
        if should_route:
            prepare_data_with_credentials(
                data=data,
                credentials=credentials,  # type: ignore[arg-type]
                file_id=original_file_id,
            )
            effective_custom_llm_provider = cast(
                str, credentials["custom_llm_provider"]
            )

        stream_result = cast(
            FileContentStreamingResult,
            await litellm.afile_content(
                **{
                    "custom_llm_provider": effective_custom_llm_provider,
                    "file_id": file_id,
                    "stream": True,
                    **data,
                }  # type: ignore
            ),
        )

        stream_iterator = cast(
            AsyncIterator[bytes],
            stream_result.stream_iterator,
        )
        hidden_params = getattr(stream_iterator, "_hidden_params", {}) or {}
        response_headers = {
            **stream_result.headers,
            **ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=hidden_params.get("model_id", "") or "",
                cache_key=hidden_params.get("cache_key", "") or "",
                api_base=hidden_params.get("api_base", "") or "",
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            ),
        }

        return StreamingResponse(
            FileContentStreamingHandler.stream_file_content_with_logging(
                stream_iterator=stream_iterator,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
                data=data,
            ),
            media_type="application/octet-stream",
            headers=response_headers,
        )

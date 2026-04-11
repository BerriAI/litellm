from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional, Tuple, cast

from fastapi.responses import StreamingResponse

import litellm
from litellm.files.types import FileContentProvider, FileContentStreamingResult
from litellm.types.utils import OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging


class FileContentStreamingHandler:
    @staticmethod
    def resolve_streaming_request_params(
        *,
        custom_llm_provider: str,
        file_id: str,
        data: Dict[str, Any],
        should_route: bool,
        original_file_id: Optional[str],
        credentials: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Resolve the provider, file ID, and request payload to use for streaming.

        For model-routed requests, this derives the effective provider from
        credentials, applies `prepare_data_with_credentials()` to a copied
        payload, swaps in the decoded/original file ID, and removes `model`
        so `afile_content()` does not re-resolve the provider. This helper
        does not mutate the passed-in `data` dictionary. Non-routed requests
        return the original provider, file ID, and data unchanged.
        """
        if should_route and credentials is not None:
            from litellm.proxy.openai_files_endpoints.common_utils import (
                prepare_data_with_credentials,
            )

            resolved_streaming_data = dict(data)
            prepare_data_with_credentials(
                data=resolved_streaming_data,
                credentials=credentials,
                file_id=original_file_id,
            )
            resolved_streaming_data.pop("model", None)
            resolved_streaming_provider = cast(
                str, credentials["custom_llm_provider"]
            )
            resolved_custom_llm_provider = resolved_streaming_provider
            resolved_file_id = cast(str, resolved_streaming_data["file_id"])
        else:
            resolved_streaming_data = data
            resolved_custom_llm_provider = custom_llm_provider
            resolved_file_id = file_id

        return (
            resolved_custom_llm_provider,
            resolved_file_id,
            resolved_streaming_data,
        )

    @staticmethod
    def should_stream_file_content(
        *,
        custom_llm_provider: str,
    ) -> bool:
        return (
            custom_llm_provider in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS
        )

    @staticmethod
    async def stream_file_content_with_logging(
        stream_iterator: AsyncIterator[bytes],
        proxy_logging_obj: "ProxyLogging",
        user_api_key_dict: "UserAPIKeyAuth",
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
        proxy_logging_obj: "ProxyLogging",
        user_api_key_dict: "UserAPIKeyAuth",
        version: str,
    ) -> StreamingResponse:
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )
        stream_result = cast(
            FileContentStreamingResult,
            await litellm.afile_content(
                **{
                    "custom_llm_provider": cast(
                        FileContentProvider, custom_llm_provider
                    ),
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

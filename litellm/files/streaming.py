import datetime
import traceback
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, Optional, Union, cast

import anyio
from litellm.files.types import FileContentProvider

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.utils import StandardLoggingHiddenParams, StandardLoggingPayload

class FileContentStreamingResponse:
    """
    Iterator wrapper for file content streaming that carries LiteLLM metadata
    and emits success/failure callbacks once the stream finishes.
    """

    def __init__(
        self,
        stream_iterator: Union[Iterator[bytes], AsyncIterator[bytes]],
        file_id: str,
        model: Optional[str],
        custom_llm_provider: Optional[Union[FileContentProvider, str]],
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> None:
        self.stream_iterator = stream_iterator
        self.file_id = file_id
        self.model = model
        self.custom_llm_provider = custom_llm_provider
        self.logging_obj = logging_obj
        self.standard_logging_object: Optional["StandardLoggingPayload"] = None
        self._hidden_params: Dict[str, Any] = {}
        self._logging_completed = False
        self._close_completed = False
        self._start_time = (
            logging_obj.start_time
            if logging_obj is not None and getattr(logging_obj, "start_time", None)
            else datetime.datetime.now()
        )
        self._sync_hidden_params()

    def __iter__(self) -> "FileContentStreamingResponse":
        if not hasattr(self.stream_iterator, "__next__"):
            raise TypeError("File content stream does not support sync iteration")
        return self

    def __next__(self) -> bytes:
        if not hasattr(self.stream_iterator, "__next__"):
            raise TypeError("File content stream does not support sync iteration")

        try:
            return next(cast(Iterator[bytes], self.stream_iterator))
        except StopIteration:
            self._log_success_sync()
            raise
        except Exception as e:
            self._log_failure_sync(e)
            raise

    def __aiter__(self) -> "FileContentStreamingResponse":
        if not hasattr(self.stream_iterator, "__anext__"):
            raise TypeError("File content stream does not support async iteration")
        return self

    async def __anext__(self) -> bytes:
        if not hasattr(self.stream_iterator, "__anext__"):
            raise TypeError("File content stream does not support async iteration")

        try:
            return await cast(AsyncIterator[bytes], self.stream_iterator).__anext__()
        except StopAsyncIteration:
            await self._log_success_async()
            raise
        except Exception as e:
            await self._log_failure_async(e)
            raise

    async def aclose(self) -> None:
        if self._close_completed:
            return

        self._close_completed = True
        self._logging_completed = True
        stream_to_close = self.stream_iterator
        self.stream_iterator = cast(Union[Iterator[bytes], AsyncIterator[bytes]], iter(()))

        # Shield cleanup from request cancellation so upstream HTTP connections
        # are released promptly on client disconnects.
        with anyio.CancelScope(shield=True):
            if hasattr(stream_to_close, "aclose"):
                await cast(AsyncIterator[bytes], stream_to_close).aclose()  # type: ignore[attr-defined]
            elif hasattr(stream_to_close, "close"):
                result = cast(Iterator[bytes], stream_to_close).close()  # type: ignore[attr-defined]
                if result is not None:
                    await result

    def close(self) -> None:
        if self._close_completed:
            return

        self._close_completed = True
        self._logging_completed = True
        stream_to_close = self.stream_iterator
        self.stream_iterator = cast(Union[Iterator[bytes], AsyncIterator[bytes]], iter(()))

        if hasattr(stream_to_close, "close"):
            cast(Iterator[bytes], stream_to_close).close()  # type: ignore[attr-defined]

    def _build_logging_response(self) -> Dict[str, str]:
        response = {
            "id": self.file_id,
            "object": "file.content",
        }
        if self.model:
            response["model"] = self.model
        return response

    def _sync_hidden_params(self) -> None:
        litellm_params: dict[str, Any] = {}
        if self.logging_obj is not None:
            litellm_params = (
                self.logging_obj.model_call_details.get("litellm_params", {}) or {}
            )

        if "api_base" not in self._hidden_params and litellm_params.get("api_base"):
            self._hidden_params["api_base"] = litellm_params["api_base"]

        # The generic client decorator infers `model` from the first positional arg,
        # which is `file_id` for this API. Correct it before logging callbacks run.
        self._hidden_params["litellm_model_name"] = self.model
        if "response_cost" not in self._hidden_params:
            self._hidden_params["response_cost"] = None

    def _build_standard_logging_object(
        self,
        end_time: datetime.datetime,
    ) -> Optional["StandardLoggingPayload"]:
        if self.standard_logging_object is not None:
            return self.standard_logging_object

        if self.logging_obj is None:
            return None

        from litellm.litellm_core_utils.litellm_logging import (
            get_standard_logging_object_payload,
        )

        self._sync_hidden_params()
        payload = get_standard_logging_object_payload(
            kwargs=self.logging_obj.model_call_details,
            init_response_obj=self._build_logging_response(),
            start_time=self._start_time,
            end_time=end_time,
            logging_obj=self.logging_obj,
            status="success",
        )
        if payload is None:
            return None

        merged_hidden_params = cast(
            "StandardLoggingHiddenParams",
            {
                **cast(Dict[str, Any], payload.get("hidden_params") or {}),
                **self._hidden_params,
            },
        )
        payload["hidden_params"] = merged_hidden_params
        payload["response"] = self._build_logging_response()
        if self.custom_llm_provider is not None:
            payload["custom_llm_provider"] = self.custom_llm_provider
        if self.model is not None:
            payload["model"] = self.model
        if self._hidden_params.get("api_base"):
            payload["api_base"] = cast(str, self._hidden_params["api_base"])

        self.standard_logging_object = payload
        return payload

    async def _log_success_async(self) -> None:
        if self._logging_completed or self.logging_obj is None:
            return

        self._logging_completed = True
        end_time = datetime.datetime.now()
        standard_logging_object = self._build_standard_logging_object(end_time=end_time)
        await self.logging_obj.async_success_handler(
            result=self._build_logging_response(),
            start_time=self._start_time,
            end_time=end_time,
            standard_logging_object=standard_logging_object,
        )
        self.logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=self._build_logging_response(),
            start_time=self._start_time,
            end_time=end_time,
        )

    def _log_success_sync(self) -> None:
        if self._logging_completed or self.logging_obj is None:
            return

        self._logging_completed = True
        end_time = datetime.datetime.now()
        standard_logging_object = self._build_standard_logging_object(end_time=end_time)
        self.logging_obj.success_handler(
            result=self._build_logging_response(),
            start_time=self._start_time,
            end_time=end_time,
            standard_logging_object=standard_logging_object,
        )

    async def _log_failure_async(self, error: Exception) -> None:
        if self._logging_completed or self.logging_obj is None:
            return

        self._logging_completed = True
        end_time = datetime.datetime.now()
        traceback_str = traceback.format_exc()
        self.logging_obj.failure_handler(
            error, traceback_str, self._start_time, end_time
        )
        await self.logging_obj.async_failure_handler(
            error, traceback_str, self._start_time, end_time
        )

    def _log_failure_sync(self, error: Exception) -> None:
        if self._logging_completed or self.logging_obj is None:
            return

        self._logging_completed = True
        end_time = datetime.datetime.now()
        self.logging_obj.failure_handler(
            error, traceback.format_exc(), self._start_time, end_time
        )

from typing import Any, AsyncGenerator, Dict, List, Optional, cast

import litellm
from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponseStream, Usage


class _GenWrapper:
    def __init__(self, agen: AsyncGenerator, metrics: dict):
        self._agen = agen
        self._metrics = metrics
        self._t0 = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        import time
        if self._t0 is None:
            self._t0 = time.perf_counter()
        try:
            item = await self._agen.__anext__()
            if "ttft_ms" not in self._metrics:
                self._metrics["ttft_ms"] = (time.perf_counter() - self._t0) * 1000.0
            return item
        except StopAsyncIteration:
            self._metrics.setdefault("ttft_ms", 0.0)
            self._metrics["total_ms"] = (time.perf_counter() - self._t0) * 1000.0 if self._t0 else 0.0
            raise


async def acompletion_streaming_iterator(
    router: Any,
    model_response: CustomStreamWrapper,
    messages: List[Dict[str, str]],
    initial_kwargs: Dict[str, Any],
) -> CustomStreamWrapper:
    """
    Helper to iterate over a streaming response with fallback support.

    Mirrors Router._acompletion_streaming_iterator but lives here to keep the router slimmer.
    """
    from litellm.exceptions import MidStreamFallbackError
    from litellm.main import stream_chunk_builder
    from litellm.cost_calculator import BaseTokenUsageProcessor

    async def _stream_with_fallbacks():
        buffer = []
        try:
            async for item in model_response:
                buffer.append(item)
            for item in buffer:
                yield item
        except MidStreamFallbackError as e:
            complete_response_object = stream_chunk_builder(chunks=model_response.chunks)
            complete_response_object_usage = cast(
                Optional[Usage], getattr(complete_response_object, "usage", None)
            )
            try:
                model_group = cast(str, initial_kwargs.get("model"))
                fallbacks: Optional[List] = initial_kwargs.get("fallbacks", router.fallbacks)
                context_window_fallbacks: Optional[List] = initial_kwargs.get(
                    "context_window_fallbacks", router.context_window_fallbacks
                )
                content_policy_fallbacks: Optional[List] = initial_kwargs.get(
                    "content_policy_fallbacks", router.content_policy_fallbacks
                )
                initial_kwargs["original_function"] = router._acompletion
                initial_kwargs["messages"] = messages + [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. You are given a message and you need to respond to it. "
                            "You are also given a generated content. You need to respond to the message in continuation "
                            "of the generated content. Do not repeat the same content. Your response should be in continuation "
                            "of this text: "
                        ),
                    },
                    {"role": "assistant", "content": e.generated_content, "prefix": True},
                ]
                router._update_kwargs_before_fallbacks(model=model_group, kwargs=initial_kwargs)
                fallback_response = await router.async_function_with_fallbacks_common_utils(
                    e=e,
                    disable_fallbacks=False,
                    fallbacks=fallbacks,
                    context_window_fallbacks=context_window_fallbacks,
                    content_policy_fallbacks=content_policy_fallbacks,
                    model_group=model_group,
                    args=(),
                    kwargs=initial_kwargs,
                )

                if hasattr(fallback_response, "__aiter__"):
                    async for fallback_item in fallback_response:  # type: ignore
                        if (
                            fallback_item
                            and isinstance(fallback_item, ModelResponseStream)
                            and hasattr(fallback_item, "usage")
                        ):
                            usage = cast(Optional[Usage], getattr(fallback_item, "usage", None))
                            usage_objects = [u for u in [usage, complete_response_object_usage] if u is not None]
                            combined_usage = BaseTokenUsageProcessor.combine_usage_objects(
                                usage_objects=usage_objects
                            )
                            setattr(fallback_item, "usage", combined_usage)
                        yield fallback_item
                else:
                    yield None
            except Exception as fallback_error:
                verbose_router_logger.error(f"Fallback also failed: {fallback_error}")
                raise fallback_error

    metrics: dict = {"ttft_ms": 0.0, "total_ms": 0.0}
    wrapped = CustomStreamWrapper(
        completion_stream=_GenWrapper(_stream_with_fallbacks(), metrics),
        model=model_response.model,
        logging_obj=model_response.logging_obj,
        custom_llm_provider=model_response.custom_llm_provider,
    )
    try:
        hidden = getattr(wrapped, "_hidden_params", None)
        if isinstance(hidden, dict):
            hidden["metrics"] = metrics
        setattr(wrapped, "_litellm_metrics", metrics)
    except Exception:
        pass
    return wrapped

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.llms.vertex_ai.common_utils import get_vertex_model_id_from_url
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution import (
    register_passthrough_batch,
    unified_batch_object_id,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    GeminiSseChunkParser,
    VertexPassthroughLoggingHandler,
)
from litellm.types.utils import (
    Choices,
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    from ..success_handler import PassThroughEndpointLogging
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})


class GeminiPassthroughLoggingHandler:
    @staticmethod
    def gemini_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        if VertexPassthroughLoggingHandler.is_interactions_route(url_route):
            return VertexPassthroughLoggingHandler.interactions_passthrough_handler(
                httpx_response=httpx_response,
                request_body=request_body,
                logging_obj=logging_obj,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                custom_llm_provider="gemini",
                vertex_location=None,
            )
        if "batchGenerateContent" in url_route:
            return GeminiPassthroughLoggingHandler._batch_create_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                start_time=start_time,
                kwargs=_JSON_OBJECT.validate_python(kwargs),
            )
        if "predictLongRunning" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            gemini_video_config: Final = GeminiVideoConfig()
            litellm_video_response: Final = gemini_video_config.transform_video_create_response(
                model=model,
                raw_response=httpx_response,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
                request_data=request_body,
            )
            logging_obj.model = model
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = "gemini"
            logging_obj.custom_llm_provider = "gemini"

            response_cost: Final = litellm.completion_cost(
                completion_response=litellm_video_response,
                model=model,
                custom_llm_provider="gemini",
                call_type="create_video",
            )

            # Set response_cost in _hidden_params to prevent recalculation
            if not hasattr(litellm_video_response, "_hidden_params"):
                litellm_video_response.hidden_params = {}
            litellm_video_response.hidden_params["response_cost"] = response_cost

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = "gemini"
            logging_obj.model_call_details["response_cost"] = response_cost
            return {
                "result": litellm_video_response,
                "kwargs": kwargs,
            }

        if "generateContent" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            # Use Gemini config for transformation
            instance_of_gemini_llm: Final = litellm.GoogleAIStudioGeminiConfig()
            litellm_model_response: Final[ModelResponse] = instance_of_gemini_llm.transform_response(
                model=model,
                messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
                raw_response=httpx_response,
                model_response=litellm.ModelResponse(),
                logging_obj=logging_obj,
                optional_params={},
                litellm_params={},
                api_key="",
                request_data={},
                encoding=getattr(litellm, "encoding", None),
            )
            kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_model_response,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }
        else:
            return {
                "result": None,
                "kwargs": kwargs,
            }

    @staticmethod
    def gemini_deployment_id(model: str) -> str | None:
        """The configured ``gemini/<model>`` deployment CheckBatchCost bills a pass-through batch through, or nothing."""
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            return None
        deployments: Final = tuple(
            _JSON_OBJECT.validate_python(deployment)
            for deployment in llm_router.registry().deployments(model_name=model) or ()
        )
        deployment_ids: Final = (
            _JSON_OBJECT.validate_python(deployment.get("model_info") or _EMPTY).get("id")
            for deployment in deployments
            if _JSON_OBJECT.validate_python(deployment.get("litellm_params") or _EMPTY).get("model")
            == f"gemini/{model}"
        )
        return next((deployment_id for deployment_id in deployment_ids if isinstance(deployment_id, str)), None)

    @staticmethod
    def _batch_create_handler(
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        start_time: datetime,
        kwargs: Mapping[str, object],
    ) -> PassThroughEndpointLoggingTypedDict:
        """A 200 from ``models/{model}:batchGenerateContent`` registers the job for the calling key, so
        CheckBatchCost prices its output file when it completes. Anything else is logged as the plain
        $0 pass-through row it always was."""
        from litellm._uuid import uuid
        from litellm.llms.gemini.batches.transformation import GeminiBatchesConfig

        model: Final = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)
        if httpx_response.status_code != 200 or "name" not in _JSON_OBJECT.validate_json(httpx_response.content):
            return {"result": None, "kwargs": dict(kwargs, model=model, custom_llm_provider="gemini")}
        batch: Final = GeminiBatchesConfig().transform_create_batch_response(
            model=model, raw_response=httpx_response, logging_obj=logging_obj, litellm_params=MappingProxyType({})
        )
        deployment_id: Final = GeminiPassthroughLoggingHandler.gemini_deployment_id(model)
        if deployment_id is None:
            verbose_proxy_logger.warning(
                "No gemini/%s deployment is configured, so batch %s created through the pass-through cannot be billed",
                model,
                batch.id,
            )
            return {
                "result": None,
                "kwargs": dict(kwargs, model=model, custom_llm_provider="gemini", batch_id=batch.id),
            }
        unified_object_id: Final = unified_batch_object_id(deployment_id, batch.id)
        litellm_params: Final = _JSON_OBJECT.validate_python(kwargs.get("litellm_params") or _EMPTY)
        request_metadata: Final = _JSON_OBJECT.validate_python(litellm_params.get("metadata") or _EMPTY)
        register_passthrough_batch("Gemini", unified_object_id, batch, batch.id, request_metadata)

        litellm_model_response: Final = ModelResponse()
        litellm_model_response.id = str(uuid.uuid4())
        litellm_model_response.model = model
        litellm_model_response.object = "batch"
        litellm_model_response.created = int(start_time.timestamp())
        litellm_model_response.choices = [
            Choices(
                finish_reason="stop",
                index=0,
                message={
                    "role": "assistant",
                    "content": f"Batch job {batch.id} created and is pending. Spend lands when the batch completes.",
                    "tool_calls": None,
                    "function_call": None,
                    "provider_specific_fields": {
                        "batch_job_id": batch.id,
                        "batch_job_state": batch.status,
                        "unified_object_id": unified_object_id,
                    },
                },
            )
        ]
        logging_obj.model = model  # rebind-ok: passthrough attribution records the resolved Gemini model
        logging_obj.model_call_details.update(
            {"model": model, "custom_llm_provider": "gemini", "response_cost": 0.0, "batch_id": batch.id}
        )
        return {
            "result": litellm_model_response,
            "kwargs": dict(
                kwargs,
                response_cost=0.0,
                model=model,
                custom_llm_provider="gemini",
                batch_id=batch.id,
                unified_object_id=unified_object_id,
            ),
        }

    @staticmethod
    def _handle_logging_gemini_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: list[str],
        model: str | None,
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Gemini passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """
        kwargs: dict[str, object] = {}
        model = model or GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)
        complete_streaming_response: Final = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
            url_route=url_route,
        )

        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Gemini passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": kwargs,
            }

        kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
            custom_llm_provider="gemini",
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: Sequence[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
    ) -> ModelResponse | TextCompletionResponse | None:
        if "generateContent" in url_route or "streamGenerateContent" in url_route:
            gemini_iterator: Final = GeminiSseChunkParser(
                streaming_response=None,
                sync_stream=False,
                logging_obj=litellm_logging_obj,
            )
            parsed_chunks: Final = [gemini_iterator.parse_sse_chunk(chunk) for chunk in all_chunks]
        else:
            return None

        if len(parsed_chunks) == 0:
            return None

        all_openai_chunks: Final = [parsed_chunk for parsed_chunk in parsed_chunks if parsed_chunk is not None]

        complete_streaming_response: Final = litellm.stream_chunk_builder(chunks=all_openai_chunks)

        return complete_streaming_response

    @staticmethod
    def extract_model_from_url(url: str) -> str:
        return get_vertex_model_id_from_url(url) or "unknown"

    @staticmethod
    def _create_gemini_response_logging_payload_for_generate_content(
        litellm_model_response: ModelResponse | TextCompletionResponse,
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ):
        """
        Create the standard logging object for Gemini passthrough generateContent (streaming and non-streaming)
        """

        response_cost: Final = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider="gemini",
        )

        kwargs["response_cost"] = response_cost
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = custom_llm_provider

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", kwargs)

        # set litellm_call_id to logging response object
        litellm_model_response.id = logging_obj.litellm_call_id
        logging_obj.model = litellm_model_response.model or model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
        return kwargs

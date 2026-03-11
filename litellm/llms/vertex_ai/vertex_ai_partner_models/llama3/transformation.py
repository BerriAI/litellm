import types
from typing import Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
    OpenAIGPTConfig,
)
from litellm.types.llms.openai import AllMessageValues, OpenAIChatCompletionResponse
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from ...common_utils import VertexAIError


class VertexAILlama3Config(OpenAIGPTConfig):
    """
    Reference:https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama#streaming

    The class `VertexAILlama3Config` provides configuration for the VertexAI's Llama API interface. Below are the parameters:

    - `max_tokens` Required (integer) max tokens,

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    max_tokens: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key == "max_tokens" and value is None:
                value = self.max_tokens
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self, model: str):
        supported_params = super().get_supported_openai_params(model=model)
        try:
            supported_params.remove("max_retries")
        except KeyError:
            pass
        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ):
        if "max_completion_tokens" in non_default_params:
            non_default_params["max_tokens"] = non_default_params.pop(
                "max_completion_tokens"
            )
        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return VertexAILlama3StreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = OpenAIChatCompletionResponse(**raw_response.json())  # type: ignore
        except Exception as e:
            response_headers = getattr(raw_response, "headers", None)
            raise VertexAIError(
                message="Unable to get json response - {}, Original Response: {}".format(
                    str(e), raw_response.text
                ),
                status_code=raw_response.status_code,
                headers=response_headers,
            )
        model_response.model = completion_response.get("model", model)
        model_response.id = completion_response.get("id", "")
        model_response.created = completion_response.get("created", 0)
        setattr(model_response, "usage", Usage(**completion_response.get("usage", {})))

        model_response.choices = self._transform_choices(  # type: ignore
            choices=completion_response["choices"],
            json_mode=json_mode,
        )

        return model_response


class VertexAILlama3StreamingHandler(OpenAIChatCompletionStreamingHandler):
    """
    Vertex AI Llama models may not include role in streaming chunk deltas.
    This handler ensures the first chunk always has role="assistant".
    
    When Vertex AI returns a single chunk with both role and finish_reason (empty response),
    this handler splits it into two chunks:
    1. First chunk: role="assistant", content="", finish_reason=None
    2. Second chunk: role=None, content=None, finish_reason="stop"
    
    This matches OpenAI's streaming format where the first chunk has role and
    the final chunk has finish_reason but no role.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sent_role = False
        self._pending_chunk: Optional[ModelResponseStream] = None

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        result = super().chunk_parser(chunk)
        if not self.sent_role and result.choices:
            delta = result.choices[0].delta
            finish_reason = result.choices[0].finish_reason
            
            # If this is both the first chunk AND the final chunk (has finish_reason),
            # we need to split it into two chunks to match OpenAI format
            if finish_reason is not None:
                # Create a pending final chunk with finish_reason but no role
                self._pending_chunk = ModelResponseStream(
                    id=result.id,
                    object="chat.completion.chunk",
                    created=result.created,
                    model=result.model,
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(content=None, role=None),
                            finish_reason=finish_reason,
                        )
                    ],
                )
                # Modify current chunk to be the first chunk with role but no finish_reason
                result.choices[0].finish_reason = None
                delta.role = "assistant"
                # Ensure content is empty string for first chunk, not None
                if delta.content is None:
                    delta.content = ""
                # Prevent downstream stream wrapper from dropping this chunk
                # (it drops empty-content chunks unless special fields are present)
                if delta.provider_specific_fields is None:
                    delta.provider_specific_fields = {}
            elif delta.role is None:
                delta.role = "assistant"
            # If the first chunk has empty content, ensure it's still emitted
            if (delta.content == "" or delta.content is None) and delta.provider_specific_fields is None:
                delta.provider_specific_fields = {}
            self.sent_role = True
        return result

    def __next__(self):
        # First return any pending chunk from a previous split
        if self._pending_chunk is not None:
            chunk = self._pending_chunk
            self._pending_chunk = None
            return chunk
        return super().__next__()

    async def __anext__(self):
        # First return any pending chunk from a previous split
        if self._pending_chunk is not None:
            chunk = self._pending_chunk
            self._pending_chunk = None
            return chunk
        return await super().__anext__()

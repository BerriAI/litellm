from typing import Final

from litellm.types.utils import Choices, ModelResponse, TextChoices, TextCompletionResponse


class TextCompletionStreamWrapper:
    def __init__(
        self,
        completion_stream,
        model,
        stream_options: dict | None = None,
        custom_llm_provider: str | None = None,
    ):
        self.completion_stream = completion_stream
        self.model = model
        self.stream_options = stream_options
        self.custom_llm_provider = custom_llm_provider

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def convert_to_text_completion_object(self, chunk: ModelResponse):
        try:
            response: Final = TextCompletionResponse()
            response["id"] = chunk.get("id", None)
            response["object"] = "text_completion"
            response["created"] = chunk.get("created", None)
            response["model"] = chunk.get("model", None)
            text_choices: Final = TextChoices()
            if isinstance(chunk, Choices):  # chunk should always be of type StreamingChoices
                raise Exception

            if chunk["choices"]:
                delta: Final = chunk["choices"][0]["delta"]
                text_choices["text"] = delta["content"]
                text_choices["reasoning_content"] = delta.get("reasoning_content")
                text_choices["index"] = chunk["choices"][0]["index"]
                text_choices["finish_reason"] = chunk["choices"][0]["finish_reason"]
                response["choices"] = [text_choices]
            else:
                response["choices"] = []

            # only pass usage when stream_options["include_usage"] is True
            if self.stream_options and self.stream_options.get("include_usage", False) is True:
                response.usage = chunk.get("usage", None)
            else:
                response.usage = None

            return response
        except Exception as e:
            raise Exception(f"Error occurred converting to text completion object - chunk: {chunk}; Error: {e}")

    def __next__(self):
        # model_response = ModelResponse(stream=True, model=self.model)
        TextCompletionResponse()
        try:
            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                processed_chunk = self.convert_to_text_completion_object(chunk=chunk)
                return processed_chunk
            raise StopIteration
        except StopIteration:
            raise StopIteration
        except Exception as e:
            from litellm.litellm_core_utils.exception_mapping_utils import exception_type

            raise exception_type(
                model=self.model,
                custom_llm_provider=self.custom_llm_provider or "",
                original_exception=e,
                completion_kwargs={},
                extra_kwargs={},
            )

    async def __anext__(self):
        try:
            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                processed_chunk = self.convert_to_text_completion_object(chunk=chunk)
                return processed_chunk
            raise StopIteration
        except StopIteration:
            raise StopAsyncIteration

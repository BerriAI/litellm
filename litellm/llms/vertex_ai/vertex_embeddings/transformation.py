import types
from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from litellm.types.utils import EmbeddingResponse, Usage

from .types import *


class VertexAITextEmbeddingConfig(BaseModel):
    """
    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#TextEmbeddingInput

    Args:
        auto_truncate: Optional(bool) If True, will truncate input text to fit within the model's max input length.
        task_type: Optional(str) The type of task to be performed. The default is "RETRIEVAL_QUERY".
        title: Optional(str) The title of the document to be embedded. (only valid with task_type=RETRIEVAL_DOCUMENT).
    """

    auto_truncate: Optional[bool] = None
    task_type: Optional[
        Literal[
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
        ]
    ] = None
    title: Optional[str] = None

    def __init__(
        self,
        auto_truncate: Optional[bool] = None,
        task_type: Optional[
            Literal[
                "RETRIEVAL_QUERY",
                "RETRIEVAL_DOCUMENT",
                "SEMANTIC_SIMILARITY",
                "CLASSIFICATION",
                "CLUSTERING",
                "QUESTION_ANSWERING",
                "FACT_VERIFICATION",
            ]
        ] = None,
        title: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
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

    def get_supported_openai_params(self):
        return ["dimensions"]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict, kwargs: dict
    ):
        for param, value in non_default_params.items():
            if param == "dimensions":
                optional_params["outputDimensionality"] = value

        if "input_type" in kwargs:
            optional_params["task_type"] = kwargs.pop("input_type")
        return optional_params, kwargs

    def get_mapped_special_auth_params(self) -> dict:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def transform_openai_request_to_vertex_embedding_request(
        self, input: Union[list, str], optional_params: dict, model: str
    ) -> VertexEmbeddingRequest:
        """
        Transforms an openai request to a vertex embedding request.
        """
        # Import here to avoid circular import issues with litellm.__init__
        from litellm.llms.vertex_ai.vertex_embeddings.bge import VertexBGEConfig
        if model.isdigit():
            return self._transform_openai_request_to_fine_tuned_embedding_request(
                input, optional_params, model
            )
        if VertexBGEConfig.is_bge_model(model):
            return VertexBGEConfig.transform_request(
                input=input, optional_params=optional_params, model=model
            )

        vertex_request: VertexEmbeddingRequest = VertexEmbeddingRequest()
        vertex_text_embedding_input_list: List[TextEmbeddingInput] = []
        task_type: Optional[TaskType] = optional_params.get("task_type")
        title = optional_params.get("title")

        if isinstance(input, str):
            input = [input]  # Convert single string to list for uniform processing

        for text in input:
            embedding_input = self.create_embedding_input(
                content=text, task_type=task_type, title=title
            )
            vertex_text_embedding_input_list.append(embedding_input)

        vertex_request["instances"] = vertex_text_embedding_input_list
        vertex_request["parameters"] = EmbeddingParameters(**optional_params)

        return vertex_request

    def _transform_openai_request_to_fine_tuned_embedding_request(
        self, input: Union[list, str], optional_params: dict, model: str
    ) -> VertexEmbeddingRequest:
        """
        Transforms an openai request to a vertex fine-tuned embedding request.

        Vertex Doc: https://console.cloud.google.com/vertex-ai/model-garden?hl=en&project=adroit-crow-413218&pageState=(%22galleryStateKey%22:(%22f%22:(%22g%22:%5B%5D,%22o%22:%5B%5D),%22s%22:%22%22))
        Sample Request:

        ```json
        {
            "instances" : [
                {
                "inputs": "How would the Future of AI in 10 Years look?",
                "parameters": {
                    "max_new_tokens": 128,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "top_k": 10
                }
                }
            ]
        }
        ```
        """
        vertex_request: VertexEmbeddingRequest = VertexEmbeddingRequest()
        vertex_text_embedding_input_list: List[TextEmbeddingFineTunedInput] = []
        if isinstance(input, str):
            input = [input]  # Convert single string to list for uniform processing

        for text in input:
            embedding_input = TextEmbeddingFineTunedInput(inputs=text)
            vertex_text_embedding_input_list.append(embedding_input)

        vertex_request["instances"] = vertex_text_embedding_input_list
        vertex_request["parameters"] = TextEmbeddingFineTunedParameters(
            **optional_params
        )
        # Remove 'shared_session' from parameters if present
        if vertex_request["parameters"] is not None and "shared_session" in vertex_request["parameters"]:
            del vertex_request["parameters"]["shared_session"]  # type: ignore[typeddict-item]

        return vertex_request

    def create_embedding_input(
        self,
        content: str,
        task_type: Optional[TaskType] = None,
        title: Optional[str] = None,
    ) -> TextEmbeddingInput:
        """
        Creates a TextEmbeddingInput object.

        Vertex requires a List of TextEmbeddingInput objects. This helper function creates a single TextEmbeddingInput object.

        Args:
            content (str): The content to be embedded.
            task_type (Optional[TaskType]): The type of task to be performed.
            title (Optional[str]): The title of the document to be embedded.

        Returns:
            TextEmbeddingInput: A TextEmbeddingInput object.
        """
        text_embedding_input = TextEmbeddingInput(content=content)
        if task_type is not None:
            text_embedding_input["task_type"] = task_type
        if title is not None:
            text_embedding_input["title"] = title
        return text_embedding_input

    def transform_vertex_response_to_openai(
        self, response: dict, model: str, model_response: EmbeddingResponse
    ) -> EmbeddingResponse:
        """
        Transforms a vertex embedding response to an openai response.
        """
        if model.isdigit():
            return self._transform_vertex_response_to_openai_for_fine_tuned_models(
                response, model, model_response
            )
        
        # Import here to avoid circular import issues with litellm.__init__
        from litellm.llms.vertex_ai.vertex_embeddings.bge import VertexBGEConfig
        
        if VertexBGEConfig.is_bge_model(model):
            return VertexBGEConfig.transform_response(
                response=response, model=model, model_response=model_response
            )

        _predictions = response["predictions"]

        embedding_response = []
        input_tokens: int = 0
        for idx, element in enumerate(_predictions):
            embedding = element["embeddings"]
            embedding_response.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding["values"],
                }
            )
            input_tokens += embedding["statistics"]["token_count"]

        model_response.object = "list"
        model_response.data = embedding_response
        model_response.model = model
        usage = Usage(
            prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
        )
        setattr(model_response, "usage", usage)
        return model_response

    def _transform_vertex_response_to_openai_for_fine_tuned_models(
        self, response: dict, model: str, model_response: EmbeddingResponse
    ) -> EmbeddingResponse:
        """
        Transforms a vertex fine-tuned model embedding response to an openai response format.
        """
        _predictions = response["predictions"]

        embedding_response = []
        # For fine-tuned models, we don't get token counts in the response
        input_tokens = 0

        for idx, embedding_values in enumerate(_predictions):
            embedding_response.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding_values[
                        0
                    ],  # The embedding values are nested one level deeper
                }
            )

        model_response.object = "list"
        model_response.data = embedding_response
        model_response.model = model
        usage = Usage(
            prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
        )
        setattr(model_response, "usage", usage)
        return model_response

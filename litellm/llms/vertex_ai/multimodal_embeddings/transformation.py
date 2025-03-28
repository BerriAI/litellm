from typing import List, Optional, Union, cast

from httpx import Headers, Response

from litellm.exceptions import InternalServerError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import LiteLLMLoggingObj
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.llms.vertex_ai import (
    Instance,
    InstanceImage,
    InstanceVideo,
    MultimodalPredictions,
    VertexMultimodalEmbeddingRequest,
)
from litellm.types.utils import Embedding, EmbeddingResponse
from litellm.utils import is_base64_encoded

from ...base_llm.embedding.transformation import BaseEmbeddingConfig
from ..common_utils import VertexAIError


class VertexAIMultimodalEmbedding(BaseEmbeddingConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return ["dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "dimensions":
                optional_params["outputDimensionality"] = value
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        default_headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}",
        }
        headers.update(default_headers)
        return headers

    def _process_input_element(self, input_element: str) -> Instance:
        """
        Process the input element for multimodal embedding requests. checks if the if the input is gcs uri, base64 encoded image or plain text.

        Args:
            input_element (str): The input element to process.

        Returns:
            Dict[str, Any]: A dictionary representing the processed input element.
        """
        if len(input_element) == 0:
            return Instance(text=input_element)
        elif "gs://" in input_element:
            if "mp4" in input_element:
                return Instance(video=InstanceVideo(gcsUri=input_element))
            else:
                return Instance(image=InstanceImage(gcsUri=input_element))
        elif is_base64_encoded(s=input_element):
            return Instance(
                image=InstanceImage(
                    bytesBase64Encoded=(
                        input_element.split(",")[1]
                        if "," in input_element
                        else input_element
                    )
                )
            )
        else:
            return Instance(text=input_element)

    def process_openai_embedding_input(
        self, _input: Union[list, str]
    ) -> List[Instance]:
        """
        Process the input for multimodal embedding requests.

        Args:
            _input (Union[list, str]): The input data to process.

        Returns:
            List[Instance]: A list of processed VertexAI Instance objects.
        """

        _input_list = None
        if not isinstance(_input, list):
            _input_list = [_input]
        else:
            _input_list = _input

        processed_instances = []
        for element in _input_list:
            if isinstance(element, str):
                instance = Instance(**self._process_input_element(element))
            elif isinstance(element, dict):
                instance = Instance(**element)
            else:
                raise ValueError(f"Unsupported input type: {type(element)}")
            processed_instances.append(instance)

        return processed_instances

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        optional_params = optional_params or {}

        request_data = VertexMultimodalEmbeddingRequest()

        if "instances" in optional_params:
            request_data["instances"] = optional_params["instances"]
        elif isinstance(input, list):
            vertex_instances: List[Instance] = self.process_openai_embedding_input(
                _input=input
            )
            request_data["instances"] = vertex_instances

        else:
            # construct instances
            vertex_request_instance = Instance(**optional_params)

            if isinstance(input, str):
                vertex_request_instance = self._process_input_element(input)

            request_data["instances"] = [vertex_request_instance]

        return cast(dict, request_data)

    def transform_embedding_response(
        self,
        model: str,
        raw_response: Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        if raw_response.status_code != 200:
            raise Exception(f"Error: {raw_response.status_code} {raw_response.text}")

        _json_response = raw_response.json()
        if "predictions" not in _json_response:
            raise InternalServerError(
                message=f"embedding response does not contain 'predictions', got {_json_response}",
                llm_provider="vertex_ai",
                model=model,
            )
        _predictions = _json_response["predictions"]
        vertex_predictions = MultimodalPredictions(predictions=_predictions)
        model_response.data = self.transform_embedding_response_to_openai(
            predictions=vertex_predictions
        )
        model_response.model = model

        return model_response

    def transform_embedding_response_to_openai(
        self, predictions: MultimodalPredictions
    ) -> List[Embedding]:

        openai_embeddings: List[Embedding] = []
        if "predictions" in predictions:
            for idx, _prediction in enumerate(predictions["predictions"]):
                if _prediction:
                    if "textEmbedding" in _prediction:
                        openai_embedding_object = Embedding(
                            embedding=_prediction["textEmbedding"],
                            index=idx,
                            object="embedding",
                        )
                        openai_embeddings.append(openai_embedding_object)
                    elif "imageEmbedding" in _prediction:
                        openai_embedding_object = Embedding(
                            embedding=_prediction["imageEmbedding"],
                            index=idx,
                            object="embedding",
                        )
                        openai_embeddings.append(openai_embedding_object)
                    elif "videoEmbeddings" in _prediction:
                        for video_embedding in _prediction["videoEmbeddings"]:
                            openai_embedding_object = Embedding(
                                embedding=video_embedding["embedding"],
                                index=idx,
                                object="embedding",
                            )
                            openai_embeddings.append(openai_embedding_object)
        return openai_embeddings

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return VertexAIError(
            status_code=status_code, message=error_message, headers=headers
        )

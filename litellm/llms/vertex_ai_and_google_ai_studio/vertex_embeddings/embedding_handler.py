import json
import os
import types
from typing import Literal, Optional, Union

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.llms.vertex_ai_and_google_ai_studio.vertex_ai_non_gemini import (
    VertexAIError,
)
from litellm.types.llms.vertex_ai import *
from litellm.utils import Usage


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
        locals_ = locals()
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
                optional_params["output_dimensionality"] = value

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


def embedding(
    model: str,
    input: Union[list, str],
    print_verbose,
    model_response: litellm.EmbeddingResponse,
    optional_params: dict,
    api_key: Optional[str] = None,
    logging_obj=None,
    encoding=None,
    vertex_project=None,
    vertex_location=None,
    vertex_credentials=None,
    aembedding=False,
):
    # logic for parsing in - calling - parsing out model embedding calls
    try:
        import vertexai
    except:
        raise VertexAIError(
            status_code=400,
            message="vertexai import failed please run `pip install google-cloud-aiplatform`",
        )

    import google.auth  # type: ignore
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

    ## Load credentials with the correct quota project ref: https://github.com/googleapis/python-aiplatform/issues/2557#issuecomment-1709284744
    try:
        print_verbose(
            f"VERTEX AI: vertex_project={vertex_project}; vertex_location={vertex_location}"
        )
        if vertex_credentials is not None and isinstance(vertex_credentials, str):
            import google.oauth2.service_account

            json_obj = json.loads(vertex_credentials)

            creds = google.oauth2.service_account.Credentials.from_service_account_info(
                json_obj,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            creds, _ = google.auth.default(quota_project_id=vertex_project)
        print_verbose(
            f"VERTEX AI: creds={creds}; google application credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}"
        )
        vertexai.init(
            project=vertex_project, location=vertex_location, credentials=creds
        )
    except Exception as e:
        raise VertexAIError(status_code=401, message=str(e))

    if isinstance(input, str):
        input = [input]

    if optional_params is not None and isinstance(optional_params, dict):
        if optional_params.get("task_type") or optional_params.get("title"):
            # if user passed task_type or title, cast to TextEmbeddingInput
            _task_type = optional_params.pop("task_type", None)
            _title = optional_params.pop("title", None)
            input = [
                TextEmbeddingInput(text=x, task_type=_task_type, title=_title)
                for x in input
            ]

    try:
        llm_model = TextEmbeddingModel.from_pretrained(model)
    except Exception as e:
        raise VertexAIError(status_code=422, message=str(e))

    if aembedding == True:
        return async_embedding(
            model=model,
            client=llm_model,
            input=input,
            logging_obj=logging_obj,
            model_response=model_response,
            optional_params=optional_params,
            encoding=encoding,
        )

    _input_dict = {"texts": input, **optional_params}
    request_str = f"""embeddings = llm_model.get_embeddings({_input_dict})"""
    ## LOGGING PRE-CALL
    logging_obj.pre_call(
        input=input,
        api_key=None,
        additional_args={
            "complete_input_dict": optional_params,
            "request_str": request_str,
        },
    )

    try:
        embeddings = llm_model.get_embeddings(**_input_dict)
    except Exception as e:
        raise VertexAIError(status_code=500, message=str(e))

    ## LOGGING POST-CALL
    logging_obj.post_call(input=input, api_key=None, original_response=embeddings)
    ## Populate OpenAI compliant dictionary
    embedding_response = []
    input_tokens: int = 0
    for idx, embedding in enumerate(embeddings):
        embedding_response.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": embedding.values,
            }
        )
        input_tokens += embedding.statistics.token_count  # type: ignore
    model_response.object = "list"
    model_response.data = embedding_response
    model_response.model = model

    usage = Usage(
        prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
    )
    setattr(model_response, "usage", usage)

    return model_response


async def async_embedding(
    model: str,
    input: Union[list, str],
    model_response: litellm.EmbeddingResponse,
    logging_obj=None,
    optional_params=None,
    encoding=None,
    client=None,
):
    """
    Async embedding implementation
    """
    _input_dict = {"texts": input, **optional_params}
    request_str = f"""embeddings = llm_model.get_embeddings({_input_dict})"""
    ## LOGGING PRE-CALL
    logging_obj.pre_call(
        input=input,
        api_key=None,
        additional_args={
            "complete_input_dict": optional_params,
            "request_str": request_str,
        },
    )

    try:
        embeddings = await client.get_embeddings_async(**_input_dict)
    except Exception as e:
        raise VertexAIError(status_code=500, message=str(e))

    ## LOGGING POST-CALL
    logging_obj.post_call(input=input, api_key=None, original_response=embeddings)
    ## Populate OpenAI compliant dictionary
    embedding_response = []
    input_tokens: int = 0
    for idx, embedding in enumerate(embeddings):
        embedding_response.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": embedding.values,
            }
        )
        input_tokens += embedding.statistics.token_count

    model_response.object = "list"
    model_response.data = embedding_response
    model_response.model = model
    usage = Usage(
        prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
    )
    setattr(model_response, "usage", usage)
    return model_response


async def transform_vertex_response_to_openai(
    response: dict, model: str, model_response: litellm.EmbeddingResponse
) -> litellm.EmbeddingResponse:

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

import json
import os
import types
from typing import Literal, Optional, Union

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging.Logging import (
    Logging as LiteLLMLoggingObject,
)
from litellm.llms.vertex_ai_and_google_ai_studio.vertex_ai_non_gemini import (
    VertexAIError,
)
from litellm.llms.vertex_ai_and_google_ai_studio.vertex_llm_base import VertexBase
from litellm.types.llms.vertex_ai import *
from litellm.utils import Usage


class VertexEmbedding(VertexBase):
    def __init__(self) -> None:
        super().__init__()

    def embedding(
        self,
        model: str,
        input: Union[list, str],
        print_verbose,
        model_response: litellm.EmbeddingResponse,
        optional_params: dict,
        logging_obj: LiteLLMLoggingObject,
        api_key: Optional[str] = None,
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

                creds = (
                    google.oauth2.service_account.Credentials.from_service_account_info(
                        json_obj,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                )
            else:
                creds, _ = google.auth.default(quota_project_id=vertex_project)
            print_verbose(
                f"VERTEX AI: creds={creds}; google application credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}"
            )
            vertexai.init(
                project=vertex_project, location=vertex_location, credentials=creds  # type: ignore
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
            return self.async_embedding(
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
        self,
        model: str,
        input: Union[list, str],
        model_response: litellm.EmbeddingResponse,
        logging_obj: LiteLLMLoggingObject,
        optional_params: dict,
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
            embeddings = await client.get_embeddings_async(**_input_dict)  # type: ignore
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

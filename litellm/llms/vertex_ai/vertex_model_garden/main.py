"""
API Handler for calling Vertex AI Model Garden Models

Most Vertex Model Garden Models are OpenAI compatible - so this handler calls `openai_like_chat_completions`

Usage:

response = litellm.completion(
    model="vertex_ai/openai/5464397967697903616",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

Sent to this route when `model` is in the format `vertex_ai/openai/{MODEL_ID}`


Vertex Documentation for using the OpenAI /chat/completions endpoint: https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_pytorch_llama3_deployment.ipynb
"""

from typing import Callable, Optional, Union

import httpx  # type: ignore

from litellm.utils import ModelResponse

from ..common_utils import VertexAIError
from ..vertex_llm_base import VertexBase


def create_vertex_url(
    vertex_location: str,
    vertex_project: str,
    stream: Optional[bool],
    model: str,
    api_base: Optional[str] = None,
) -> str:
    """Return the base url for the vertex garden models"""
    #  f"https://{self.endpoint.location}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{self.endpoint.location}"
    return f"https://{vertex_location}-aiplatform.googleapis.com/v1beta1/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}"


class VertexAIModelGardenModels(VertexBase):
    def __init__(self) -> None:
        pass

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        api_base: Optional[str],
        optional_params: dict,
        custom_prompt_dict: dict,
        headers: Optional[dict],
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        logger_fn=None,
        acompletion: bool = False,
        client=None,
    ):
        """
        Handles calling Vertex AI Model Garden Models in OpenAI compatible format

        Sent to this route when `model` is in the format `vertex_ai/openai/{MODEL_ID}`
        """
        try:
            import vertexai

            from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
            from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
                VertexLLM,
            )
        except Exception as e:
            raise VertexAIError(
                status_code=400,
                message=f"""vertexai import failed please run `pip install -U "google-cloud-aiplatform>=1.38"`. Got error: {e}""",
            )

        if not (
            hasattr(vertexai, "preview") or hasattr(vertexai.preview, "language_models")
        ):
            raise VertexAIError(
                status_code=400,
                message="""Upgrade vertex ai. Run `pip install "google-cloud-aiplatform>=1.38"`""",
            )
        try:
            model = model.replace("openai/", "")
            vertex_httpx_logic = VertexLLM()

            access_token, project_id = vertex_httpx_logic._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            openai_like_chat_completions = OpenAILikeChatHandler()

            ## CONSTRUCT API BASE
            stream: bool = optional_params.get("stream", False) or False
            optional_params["stream"] = stream
            default_api_base = create_vertex_url(
                vertex_location=vertex_location or "us-central1",
                vertex_project=vertex_project or project_id,
                stream=stream,
                model=model,
            )

            if len(default_api_base.split(":")) > 1:
                endpoint = default_api_base.split(":")[-1]
            else:
                endpoint = ""

            _, api_base = self._check_custom_proxy(
                api_base=api_base,
                custom_llm_provider="vertex_ai",
                gemini_api_key=None,
                endpoint=endpoint,
                stream=stream,
                auth_header=None,
                url=default_api_base,
            )
            model = ""
            return openai_like_chat_completions.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=access_token,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                logging_obj=logging_obj,
                optional_params=optional_params,
                acompletion=acompletion,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                client=client,
                timeout=timeout,
                encoding=encoding,
                custom_llm_provider="vertex_ai",
            )

        except Exception as e:
            raise VertexAIError(status_code=500, message=str(e))

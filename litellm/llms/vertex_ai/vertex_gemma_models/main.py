"""
API Handler for calling Vertex AI Gemma Models

These models use a custom prediction endpoint format that wraps messages in 'instances'
with @requestFormat: "chatCompletions" and returns responses wrapped in 'predictions'.

Usage:

response = litellm.completion(
    model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    vertex_project="your-project-id",
    vertex_location="us-central1",
)

Sent to this route when `model` is in the format `vertex_ai/gemma/{MODEL_NAME}`

The API expects a custom endpoint URL format:
https://{ENDPOINT_NUMBER}.{location}-{REGION_NUMBER}.prediction.vertexai.goog/v1/projects/{PROJECT_ID}/locations/{location}/endpoints/{ENDPOINT_ID}:predict
"""

from typing import Callable, Optional, Union

import httpx  # type: ignore

from litellm.utils import ModelResponse

from ..common_utils import VertexAIError
from ..vertex_llm_base import VertexBase


class VertexAIGemmaModels(VertexBase):
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
        Handles calling Vertex AI Gemma Models

        Sent to this route when `model` is in the format `vertex_ai/gemma/{MODEL_NAME}`
        """
        try:
            import vertexai

            from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
                VertexLLM,
            )
            from litellm.llms.vertex_ai.vertex_gemma_models.transformation import (
                VertexGemmaConfig,
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
            model = model.replace("gemma/", "")
            vertex_httpx_logic = VertexLLM()

            access_token, project_id = vertex_httpx_logic._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            gemma_transformation = VertexGemmaConfig()

            ## CONSTRUCT API BASE
            stream: bool = optional_params.get("stream", False) or False
            optional_params["stream"] = stream

            # If api_base is not provided, it should be set as an environment variable
            # or passed explicitly because the endpoint URL is unique per deployment
            if api_base is None:
                raise VertexAIError(
                    status_code=400,
                    message="api_base is required for Vertex AI Gemma models. Please provide the full endpoint URL.",
                )

            # Check if we need to append :predict
            if not api_base.endswith(":predict"):
                _, api_base = self._check_custom_proxy(
                    api_base=api_base,
                    custom_llm_provider="vertex_ai",
                    gemini_api_key=None,
                    endpoint="predict",
                    stream=stream,
                    auth_header=None,
                    url=api_base,
                )
            # If api_base already ends with :predict, use it as-is

            # Use the custom transformation handler for gemma models
            return gemma_transformation.completion(
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
            if hasattr(e, "status_code"):
                raise e
            raise VertexAIError(status_code=500, message=str(e))


# What is this?
## API Handler for calling Vertex AI Partner Models
from enum import Enum
from typing import Callable, Optional, Union

import httpx  # type: ignore

import litellm
from litellm import LlmProviders
from litellm.types.llms.vertex_ai import VertexPartnerProvider
from litellm.utils import ModelResponse

from ...custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from ..vertex_llm_base import VertexBase

base_llm_http_handler = BaseLLMHTTPHandler()


class VertexAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url=" https://cloud.google.com/vertex-ai/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class PartnerModelPrefixes(str, Enum):
    META_PREFIX = "meta/"
    DEEPSEEK_PREFIX = "deepseek-ai"
    MISTRAL_PREFIX = "mistral"
    CODERESTAL_PREFIX = "codestral"
    JAMBA_PREFIX = "jamba"
    CLAUDE_PREFIX = "claude"
    QWEN_PREFIX = "qwen"
    GPT_OSS_PREFIX = "openai/gpt-oss-"


class VertexAIPartnerModels(VertexBase):
    def __init__(self) -> None:
        pass

    @staticmethod
    def is_vertex_partner_model(model: str):
        """
        Check if the model string is a Vertex AI Partner Model
        Only use this once you have confirmed that custom_llm_provider is vertex_ai

        Returns:
            bool: True if the model string is a Vertex AI Partner Model, False otherwise
        """
        if (
            model.startswith(PartnerModelPrefixes.META_PREFIX)
            or model.startswith(PartnerModelPrefixes.DEEPSEEK_PREFIX)
            or model.startswith(PartnerModelPrefixes.MISTRAL_PREFIX)
            or model.startswith(PartnerModelPrefixes.CODERESTAL_PREFIX)
            or model.startswith(PartnerModelPrefixes.JAMBA_PREFIX)
            or model.startswith(PartnerModelPrefixes.CLAUDE_PREFIX)
            or model.startswith(PartnerModelPrefixes.QWEN_PREFIX)
            or model.startswith(PartnerModelPrefixes.GPT_OSS_PREFIX)
        ):
            return True
        return False
    
    @staticmethod
    def should_use_openai_handler(model: str):
        OPENAI_LIKE_VERTEX_PROVIDERS = [
            "llama",
            PartnerModelPrefixes.DEEPSEEK_PREFIX,
            PartnerModelPrefixes.QWEN_PREFIX,
            PartnerModelPrefixes.GPT_OSS_PREFIX,
        ]
        if any(provider in model for provider in OPENAI_LIKE_VERTEX_PROVIDERS):
            return True
        return False

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
        try:
            import vertexai

            from litellm.llms.anthropic.chat import AnthropicChatCompletion
            from litellm.llms.codestral.completion.handler import (
                CodestralTextCompletion,
            )
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
            vertex_httpx_logic = VertexLLM()

            access_token, project_id = vertex_httpx_logic._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            openai_like_chat_completions = OpenAILikeChatHandler()
            codestral_fim_completions = CodestralTextCompletion()
            anthropic_chat_completions = AnthropicChatCompletion()

            ## CONSTRUCT API BASE
            stream: bool = optional_params.get("stream", False) or False

            optional_params["stream"] = stream

            if self.should_use_openai_handler(model):
                partner = VertexPartnerProvider.llama
            elif "mistral" in model or "codestral" in model:
                partner = VertexPartnerProvider.mistralai
            elif "jamba" in model:
                partner = VertexPartnerProvider.ai21
            elif "claude" in model:
                partner = VertexPartnerProvider.claude
            else:
                raise ValueError(f"Unknown partner model: {model}")

            api_base = self.get_complete_vertex_url(
                custom_api_base=api_base,
                vertex_location=vertex_location,
                vertex_project=vertex_project,
                project_id=project_id,
                partner=partner,
                stream=stream,
                model=model,
            )

            if "codestral" in model or "mistral" in model:
                model = model.split("@")[0]

            if "codestral" in model and litellm_params.get("text_completion") is True:
                optional_params["model"] = model
                text_completion_model_response = litellm.TextCompletionResponse(
                    stream=stream
                )
                return codestral_fim_completions.completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    api_key=access_token,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=text_completion_model_response,
                    print_verbose=print_verbose,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    acompletion=acompletion,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,
                    encoding=encoding,
                )
            elif "claude" in model:
                if headers is None:
                    headers = {}
                headers.update({"Authorization": "Bearer {}".format(access_token)})

                optional_params.update(
                    {
                        "anthropic_version": "vertex-2023-10-16",
                        "is_vertex_request": True,
                    }
                )

                return anthropic_chat_completions.completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    acompletion=acompletion,
                    custom_prompt_dict=litellm.custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,  # for calculating input/output tokens
                    api_key=access_token,
                    logging_obj=logging_obj,
                    headers=headers,
                    timeout=timeout,
                    client=client,
                    custom_llm_provider=LlmProviders.VERTEX_AI.value,
                )
            elif self.should_use_openai_handler(model):
                return base_llm_http_handler.completion(
                    model=model,
                    stream=stream,
                    messages=messages,
                    acompletion=acompletion,
                    api_base=api_base,
                    model_response=model_response,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    custom_llm_provider="vertex_ai",
                    timeout=timeout,
                    headers=headers,
                    encoding=encoding,
                    api_key=access_token,
                    logging_obj=logging_obj,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
                    client=client,
                )
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
                custom_endpoint=True,
            )

        except Exception as e:
            if hasattr(e, "status_code"):
                raise e
            raise VertexAIError(status_code=500, message=str(e))

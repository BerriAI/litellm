from typing import Optional, Tuple

import httpx

import litellm
from litellm.constants import REPLICATE_MODEL_NAME_WITH_ID_LENGTH
from litellm.secret_managers.main import get_secret, get_secret_str

from ..types.router import LiteLLM_Params


def _is_non_openai_azure_model(model: str) -> bool:
    try:
        model_name = model.split("/", 1)[1]
        if (
            model_name in litellm.cohere_chat_models
            or f"mistral/{model_name}" in litellm.mistral_chat_models
        ):
            return True
    except Exception:
        return False
    return False


def handle_cohere_chat_model_custom_llm_provider(
    model: str, custom_llm_provider: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    if user sets model = "cohere/command-r" -> use custom_llm_provider = "cohere_chat"

    Args:
        model:
        custom_llm_provider:

    Returns:
        model, custom_llm_provider
    """

    if custom_llm_provider:
        if custom_llm_provider == "cohere" and model in litellm.cohere_chat_models:
            return model, "cohere_chat"

    if "/" in model:
        _custom_llm_provider, _model = model.split("/", 1)
        if (
            _custom_llm_provider
            and _custom_llm_provider == "cohere"
            and _model in litellm.cohere_chat_models
        ):
            return _model, "cohere_chat"

    return model, custom_llm_provider


def handle_anthropic_text_model_custom_llm_provider(
    model: str, custom_llm_provider: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    if user sets model = "anthropic/claude-2" -> use custom_llm_provider = "anthropic_text"

    Args:
        model:
        custom_llm_provider:

    Returns:
        model, custom_llm_provider
    """

    if custom_llm_provider:
        if (
            custom_llm_provider == "anthropic"
            and litellm.AnthropicTextConfig._is_anthropic_text_model(model)
        ):
            return model, "anthropic_text"

    if "/" in model:
        _custom_llm_provider, _model = model.split("/", 1)
        if (
            _custom_llm_provider
            and _custom_llm_provider == "anthropic"
            and litellm.AnthropicTextConfig._is_anthropic_text_model(_model)
        ):
            return _model, "anthropic_text"

    return model, custom_llm_provider


def get_llm_provider(  # noqa: PLR0915
    model: str,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    litellm_params: Optional[LiteLLM_Params] = None,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Returns the provider for a given model name - e.g. 'azure/chatgpt-v-2' -> 'azure'

    For router -> Can also give the whole litellm param dict -> this function will extract the relevant details

    Raises Error - if unable to map model to a provider

    Return model, custom_llm_provider, dynamic_api_key, api_base
    """
    try:
        if litellm.LiteLLMProxyChatConfig._should_use_litellm_proxy_by_default(
            litellm_params=litellm_params
        ):
            return litellm.LiteLLMProxyChatConfig.litellm_proxy_get_custom_llm_provider_info(
                model=model, api_base=api_base, api_key=api_key
            )

        ## IF LITELLM PARAMS GIVEN ##
        if litellm_params:
            assert (
                custom_llm_provider is None and api_base is None and api_key is None
            ), "Either pass in litellm_params or the custom_llm_provider/api_base/api_key. Otherwise, these values will be overriden."
            custom_llm_provider = litellm_params.custom_llm_provider
            api_base = litellm_params.api_base
            api_key = litellm_params.api_key

        dynamic_api_key = None
        # check if llm provider provided
        # AZURE AI-Studio Logic - Azure AI Studio supports AZURE/Cohere
        # If User passes azure/command-r-plus -> we should send it to cohere_chat/command-r-plus
        if model.split("/", 1)[0] == "azure":
            if _is_non_openai_azure_model(model):
                custom_llm_provider = "openai"
                return model, custom_llm_provider, dynamic_api_key, api_base

        ### Handle cases when custom_llm_provider is set to cohere/command-r-plus but it should use cohere_chat route
        model, custom_llm_provider = handle_cohere_chat_model_custom_llm_provider(
            model, custom_llm_provider
        )

        model, custom_llm_provider = handle_anthropic_text_model_custom_llm_provider(
            model, custom_llm_provider
        )

        if custom_llm_provider and (
            model.split("/")[0] != custom_llm_provider
        ):  # handle scenario where model="azure/*" and custom_llm_provider="azure"
            model = custom_llm_provider + "/" + model

        if api_key and api_key.startswith("os.environ/"):
            dynamic_api_key = get_secret_str(api_key)
        # check if llm provider part of model name

        if (
            model.split("/", 1)[0] in litellm.provider_list
            and model.split("/", 1)[0] not in litellm.model_list_set
            and len(model.split("/"))
            > 1  # handle edge case where user passes in `litellm --model mistral` https://github.com/BerriAI/litellm/issues/1351
        ):
            return _get_openai_compatible_provider_info(
                model=model,
                api_base=api_base,
                api_key=api_key,
                dynamic_api_key=dynamic_api_key,
            )
        elif model.split("/", 1)[0] in litellm.provider_list:
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]
            if api_base is not None and not isinstance(api_base, str):
                raise Exception(
                    "api base needs to be a string. api_base={}".format(api_base)
                )
            if dynamic_api_key is not None and not isinstance(dynamic_api_key, str):
                raise Exception(
                    "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                        dynamic_api_key
                    )
                )
            return model, custom_llm_provider, dynamic_api_key, api_base
        # check if api base is a known openai compatible endpoint
        if api_base:
            for endpoint in litellm.openai_compatible_endpoints:
                if endpoint in api_base:
                    if endpoint == "api.perplexity.ai":
                        custom_llm_provider = "perplexity"
                        dynamic_api_key = get_secret_str("PERPLEXITYAI_API_KEY")
                    elif endpoint == "api.endpoints.anyscale.com/v1":
                        custom_llm_provider = "anyscale"
                        dynamic_api_key = get_secret_str("ANYSCALE_API_KEY")
                    elif endpoint == "api.deepinfra.com/v1/openai":
                        custom_llm_provider = "deepinfra"
                        dynamic_api_key = get_secret_str("DEEPINFRA_API_KEY")
                    elif endpoint == "api.mistral.ai/v1":
                        custom_llm_provider = "mistral"
                        dynamic_api_key = get_secret_str("MISTRAL_API_KEY")
                    elif endpoint == "api.groq.com/openai/v1":
                        custom_llm_provider = "groq"
                        dynamic_api_key = get_secret_str("GROQ_API_KEY")
                    elif endpoint == "https://integrate.api.nvidia.com/v1":
                        custom_llm_provider = "nvidia_nim"
                        dynamic_api_key = get_secret_str("NVIDIA_NIM_API_KEY")
                    elif endpoint == "https://api.cerebras.ai/v1":
                        custom_llm_provider = "cerebras"
                        dynamic_api_key = get_secret_str("CEREBRAS_API_KEY")
                    elif endpoint == "https://api.sambanova.ai/v1":
                        custom_llm_provider = "sambanova"
                        dynamic_api_key = get_secret_str("SAMBANOVA_API_KEY")
                    elif endpoint == "https://api.ai21.com/studio/v1":
                        custom_llm_provider = "ai21_chat"
                        dynamic_api_key = get_secret_str("AI21_API_KEY")
                    elif endpoint == "https://codestral.mistral.ai/v1":
                        custom_llm_provider = "codestral"
                        dynamic_api_key = get_secret_str("CODESTRAL_API_KEY")
                    elif endpoint == "https://codestral.mistral.ai/v1":
                        custom_llm_provider = "text-completion-codestral"
                        dynamic_api_key = get_secret_str("CODESTRAL_API_KEY")
                    elif endpoint == "app.empower.dev/api/v1":
                        custom_llm_provider = "empower"
                        dynamic_api_key = get_secret_str("EMPOWER_API_KEY")
                    elif endpoint == "api.deepseek.com/v1":
                        custom_llm_provider = "deepseek"
                        dynamic_api_key = get_secret_str("DEEPSEEK_API_KEY")
                    elif endpoint == "https://api.friendli.ai/serverless/v1":
                        custom_llm_provider = "friendliai"
                        dynamic_api_key = get_secret_str(
                            "FRIENDLIAI_API_KEY"
                        ) or get_secret("FRIENDLI_TOKEN")
                    elif endpoint == "api.galadriel.com/v1":
                        custom_llm_provider = "galadriel"
                        dynamic_api_key = get_secret_str("GALADRIEL_API_KEY")
                    elif endpoint == "https://api.llama.com/compat/v1":
                        custom_llm_provider = "meta_llama"
                        dynamic_api_key = api_key or get_secret_str("LLAMA_API_KEY")
                    elif endpoint == "https://api.featherless.ai/v1":
                        custom_llm_provider = "featherless_ai"
                        dynamic_api_key = get_secret_str("FEATHERLESS_AI_API_KEY")
                    elif endpoint == litellm.NscaleConfig.API_BASE_URL:
                        custom_llm_provider = "nscale"
                        dynamic_api_key = litellm.NscaleConfig.get_api_key()
                    elif endpoint == "dashscope-intl.aliyuncs.com/compatible-mode/v1":
                        custom_llm_provider = "dashscope"
                        dynamic_api_key = get_secret_str("DASHSCOPE_API_KEY")
                    elif endpoint == "api.moonshot.ai/v1":
                        custom_llm_provider = "moonshot"
                        dynamic_api_key = get_secret_str("MOONSHOT_API_KEY")
                    elif endpoint == "https://api.v0.dev/v1":
                        custom_llm_provider = "v0"
                        dynamic_api_key = get_secret_str("V0_API_KEY")
                    elif endpoint == "https://api.lambda.ai/v1":
                        custom_llm_provider = "lambda_ai"
                        dynamic_api_key = get_secret_str("LAMBDA_API_KEY")
                    elif endpoint == "https://api.hyperbolic.xyz/v1":
                        custom_llm_provider = "hyperbolic"
                        dynamic_api_key = get_secret_str("HYPERBOLIC_API_KEY")

                    if api_base is not None and not isinstance(api_base, str):
                        raise Exception(
                            "api base needs to be a string. api_base={}".format(
                                api_base
                            )
                        )
                    if dynamic_api_key is not None and not isinstance(
                        dynamic_api_key, str
                    ):
                        raise Exception(
                            "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                                dynamic_api_key
                            )
                        )
                    return model, custom_llm_provider, dynamic_api_key, api_base  # type: ignore

        # check if model in known model provider list  -> for huggingface models, raise exception as they don't have a fixed provider (can be togetherai, anyscale, baseten, runpod, et.)
        ## openai - chatcompletion + text completion
        if (
            model in litellm.open_ai_chat_completion_models
            or "ft:gpt-3.5-turbo" in model
            or "ft:gpt-4" in model  # catches ft:gpt-4-0613, ft:gpt-4o
            or model in litellm.openai_image_generation_models
        ):
            custom_llm_provider = "openai"
        elif model in litellm.open_ai_text_completion_models:
            custom_llm_provider = "text-completion-openai"
        ## anthropic
        elif model in litellm.anthropic_models:
            if litellm.AnthropicTextConfig._is_anthropic_text_model(model):
                custom_llm_provider = "anthropic_text"
            else:
                custom_llm_provider = "anthropic"
        ## cohere
        elif model in litellm.cohere_models or model in litellm.cohere_embedding_models:
            custom_llm_provider = "cohere"
        ## cohere chat models
        elif model in litellm.cohere_chat_models:
            custom_llm_provider = "cohere_chat"
        ## replicate
        elif model in litellm.replicate_models or (
            ":" in model and len(model) > REPLICATE_MODEL_NAME_WITH_ID_LENGTH
        ):
            model_parts = model.split(":")
            if (
                len(model_parts) > 1
                and len(model_parts[1]) == REPLICATE_MODEL_NAME_WITH_ID_LENGTH
            ):  ## checks if model name has a 64 digit code - e.g. "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
                custom_llm_provider = "replicate"
            elif model in litellm.replicate_models:
                custom_llm_provider = "replicate"
        ## openrouter
        elif model in litellm.openrouter_models:
            custom_llm_provider = "openrouter"
        ## maritalk
        elif model in litellm.maritalk_models:
            custom_llm_provider = "maritalk"
        ## vertex - text + chat + language (gemini) models
        elif (
            model in litellm.vertex_chat_models
            or model in litellm.vertex_code_chat_models
            or model in litellm.vertex_text_models
            or model in litellm.vertex_code_text_models
            or model in litellm.vertex_language_models
            or model in litellm.vertex_embedding_models
            or model in litellm.vertex_vision_models
            or model in litellm.vertex_ai_image_models
        ):
            custom_llm_provider = "vertex_ai"
        ## ai21
        elif model in litellm.ai21_chat_models or model in litellm.ai21_models:
            custom_llm_provider = "ai21_chat"
            api_base = (
                api_base
                or get_secret("AI21_API_BASE")
                or "https://api.ai21.com/studio/v1"
            )  # type: ignore
            dynamic_api_key = api_key or get_secret("AI21_API_KEY")
        ## aleph_alpha
        elif model in litellm.aleph_alpha_models:
            custom_llm_provider = "aleph_alpha"
        ## baseten
        elif model in litellm.baseten_models:
            custom_llm_provider = "baseten"
        ## nlp_cloud
        elif model in litellm.nlp_cloud_models:
            custom_llm_provider = "nlp_cloud"
        ## petals
        elif model in litellm.petals_models:
            custom_llm_provider = "petals"
        ## bedrock
        elif (
            model in litellm.bedrock_models
            or model in litellm.bedrock_embedding_models
            or model in litellm.bedrock_converse_models
        ):
            custom_llm_provider = "bedrock"
        elif model in litellm.watsonx_models:
            custom_llm_provider = "watsonx"
        # openai embeddings
        elif model in litellm.open_ai_embedding_models:
            custom_llm_provider = "openai"
        elif model in litellm.empower_models:
            custom_llm_provider = "empower"
        elif model in litellm.gradient_ai_models:
            custom_llm_provider = "gradient_ai"
        elif model == "*":
            custom_llm_provider = "openai"
        # bytez models
        elif model.startswith("bytez/"):
            custom_llm_provider = "bytez"
        # cometapi models
        elif model.startswith("cometapi/"):
            custom_llm_provider = "cometapi"
        elif model.startswith("oci/"):
            custom_llm_provider = "oci"
        if not custom_llm_provider:
            if litellm.suppress_debug_info is False:
                print()  # noqa
                print(  # noqa
                    "\033[1;31mProvider List: https://docs.litellm.ai/docs/providers\033[0m"  # noqa
                )  # noqa
                print()  # noqa
            error_str = f"LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model={model}\n Pass model as E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/starcoder',..)` Learn more: https://docs.litellm.ai/docs/providers"
            # maps to openai.NotFoundError, this is raised when openai does not recognize the llm
            raise litellm.exceptions.BadRequestError(  # type: ignore
                message=error_str,
                model=model,
                response=httpx.Response(
                    status_code=400,
                    content=error_str,
                    request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
                llm_provider="",
            )
        if api_base is not None and not isinstance(api_base, str):
            raise Exception(
                "api base needs to be a string. api_base={}".format(api_base)
            )
        if dynamic_api_key is not None and not isinstance(dynamic_api_key, str):
            raise Exception(
                "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                    dynamic_api_key
                )
            )
        return model, custom_llm_provider, dynamic_api_key, api_base
    except Exception as e:
        if isinstance(e, litellm.exceptions.BadRequestError):
            raise e
        else:
            error_str = (
                f"GetLLMProvider Exception - {str(e)}\n\noriginal model: {model}"
            )
            raise litellm.exceptions.BadRequestError(  # type: ignore
                message=f"GetLLMProvider Exception - {str(e)}\n\noriginal model: {model}",
                model=model,
                response=httpx.Response(
                    status_code=400,
                    content=error_str,
                    request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
                llm_provider="",
            )


def _get_openai_compatible_provider_info(  # noqa: PLR0915
    model: str,
    api_base: Optional[str],
    api_key: Optional[str],
    dynamic_api_key: Optional[str],
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Returns:
        Tuple[str, str, Optional[str], Optional[str]]:
            model: str
            custom_llm_provider: str
            dynamic_api_key: Optional[str]
            api_base: Optional[str]
    """

    custom_llm_provider = model.split("/", 1)[0]
    model = model.split("/", 1)[1]

    if custom_llm_provider == "perplexity":
        # perplexity is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.perplexity.ai
        (
            api_base,
            dynamic_api_key,
        ) = litellm.PerplexityChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "aiohttp_openai":
        return model, "aiohttp_openai", api_key, api_base
    elif custom_llm_provider == "anyscale":
        # anyscale is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = api_base or get_secret_str("ANYSCALE_API_BASE") or "https://api.endpoints.anyscale.com/v1"  # type: ignore
        dynamic_api_key = api_key or get_secret_str("ANYSCALE_API_KEY")
    elif custom_llm_provider == "deepinfra":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.DeepInfraConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "empower":
        api_base = (
            api_base
            or get_secret("EMPOWER_API_BASE")
            or "https://app.empower.dev/api/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("EMPOWER_API_KEY")
    elif custom_llm_provider == "groq":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.GroqChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "nvidia_nim":
        # nvidia_nim is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = (
            api_base
            or get_secret("NVIDIA_NIM_API_BASE")
            or "https://integrate.api.nvidia.com/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("NVIDIA_NIM_API_KEY")
    elif custom_llm_provider == "cerebras":
        api_base = (
            api_base or get_secret("CEREBRAS_API_BASE") or "https://api.cerebras.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("CEREBRAS_API_KEY")
    elif custom_llm_provider == "sambanova":
        api_base = (
            api_base
            or get_secret("SAMBANOVA_API_BASE")
            or "https://api.sambanova.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("SAMBANOVA_API_KEY")
    elif custom_llm_provider == "meta_llama":
        api_base = (
            api_base
            or get_secret("LLAMA_API_BASE")
            or "https://api.llama.com/compat/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("LLAMA_API_KEY")
    elif custom_llm_provider == "nebius":
        api_base = (
            api_base
            or get_secret("NEBIUS_API_BASE")
            or "https://api.studio.nebius.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("NEBIUS_API_KEY")
    elif (custom_llm_provider == "ai21_chat") or (
        custom_llm_provider == "ai21" and model in litellm.ai21_chat_models
    ):
        api_base = (
            api_base or get_secret("AI21_API_BASE") or "https://api.ai21.com/studio/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("AI21_API_KEY")
        custom_llm_provider = "ai21_chat"
    elif custom_llm_provider == "volcengine":
        # volcengine is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = (
            api_base
            or get_secret("VOLCENGINE_API_BASE")
            or "https://ark.cn-beijing.volces.com/api/v3"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("VOLCENGINE_API_KEY")
    elif custom_llm_provider == "codestral":
        # codestral is openai compatible, we just need to set this to custom_openai and have the api_base be https://codestral.mistral.ai/v1
        api_base = (
            api_base
            or get_secret("CODESTRAL_API_BASE")
            or "https://codestral.mistral.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("CODESTRAL_API_KEY")
    elif custom_llm_provider == "hosted_vllm":
        # vllm is openai compatible, we just need to set this to custom_openai
        (
            api_base,
            dynamic_api_key,
        ) = litellm.HostedVLLMChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "llamafile":
        # llamafile is OpenAI compatible.
        (
            api_base,
            dynamic_api_key,
        ) = litellm.LlamafileChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "datarobot":
        # DataRobot is OpenAI compatible.
        (
            api_base,
            dynamic_api_key,
        ) = litellm.DataRobotConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "lm_studio":
        # lm_studio is openai compatible, we just need to set this to custom_openai
        (
            api_base,
            dynamic_api_key,
        ) = litellm.LMStudioChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "deepseek":
        # deepseek is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.deepseek.com/v1
        api_base = (
            api_base
            or get_secret("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/beta"
        )  # type: ignore

        dynamic_api_key = api_key or get_secret_str("DEEPSEEK_API_KEY")
    elif custom_llm_provider == "fireworks_ai":
        # fireworks is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.fireworks.ai/inference/v1
        (
            api_base,
            dynamic_api_key,
        ) = litellm.FireworksAIConfig()._get_openai_compatible_provider_info(
            api_base=api_base, api_key=api_key
        )
    elif custom_llm_provider == "azure_ai":
        (
            api_base,
            dynamic_api_key,
            custom_llm_provider,
        ) = litellm.AzureAIStudioConfig()._get_openai_compatible_provider_info(
            model, api_base, api_key, custom_llm_provider
        )
    elif custom_llm_provider == "github":
        api_base = (
            api_base
            or get_secret_str("GITHUB_API_BASE")
            or "https://models.inference.ai.azure.com"  # This is github's default base url
        )
        dynamic_api_key = api_key or get_secret_str("GITHUB_API_KEY")
    elif custom_llm_provider == "litellm_proxy":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.LiteLLMProxyChatConfig()._get_openai_compatible_provider_info(
            api_base=api_base, api_key=api_key
        )

    elif custom_llm_provider == "mistral":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.MistralConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "jina_ai":
        (
            custom_llm_provider,
            api_base,
            dynamic_api_key,
        ) = litellm.JinaAIEmbeddingConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "xai":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.XAIChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "together_ai":
        api_base = (
            api_base
            or get_secret_str("TOGETHER_AI_API_BASE")
            or "https://api.together.xyz/v1"
        )  # type: ignore
        dynamic_api_key = api_key or (
            get_secret_str("TOGETHER_API_KEY")
            or get_secret_str("TOGETHER_AI_API_KEY")
            or get_secret_str("TOGETHERAI_API_KEY")
            or get_secret_str("TOGETHER_AI_TOKEN")
        )
    elif custom_llm_provider == "friendliai":
        api_base = (
            api_base
            or get_secret("FRIENDLI_API_BASE")
            or "https://api.friendli.ai/serverless/v1"
        )  # type: ignore
        dynamic_api_key = (
            api_key
            or get_secret_str("FRIENDLIAI_API_KEY")
            or get_secret_str("FRIENDLI_TOKEN")
        )
    elif custom_llm_provider == "galadriel":
        api_base = (
            api_base
            or get_secret("GALADRIEL_API_BASE")
            or "https://api.galadriel.com/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("GALADRIEL_API_KEY")
    elif custom_llm_provider == "github_copilot":
        (
            api_base,
            dynamic_api_key,
            custom_llm_provider,
        ) = litellm.GithubCopilotConfig()._get_openai_compatible_provider_info(
            model, api_base, api_key, custom_llm_provider
        )
    elif custom_llm_provider == "novita":
        api_base = (
            api_base
            or get_secret("NOVITA_API_BASE")
            or "https://api.novita.ai/v3/openai"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("NOVITA_API_KEY")
    elif custom_llm_provider == "snowflake":
        api_base = (
            api_base
            or get_secret_str("SNOWFLAKE_API_BASE")
            or f"https://{get_secret('SNOWFLAKE_ACCOUNT_ID')}.snowflakecomputing.com/api/v2/cortex/inference:complete"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("SNOWFLAKE_JWT")
    elif custom_llm_provider == "gradient_ai":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.GradientAIConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "featherless_ai":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.FeatherlessAIConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "nscale":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.NscaleConfig()._get_openai_compatible_provider_info(
            api_base=api_base, api_key=api_key
        )
    elif custom_llm_provider == "dashscope":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.DashScopeChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "moonshot":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.MoonshotChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "v0":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.V0ChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "morph":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.MorphChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "lambda_ai":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.LambdaAIChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )
    elif custom_llm_provider == "hyperbolic":
        (
            api_base,
            dynamic_api_key,
        ) = litellm.HyperbolicChatConfig()._get_openai_compatible_provider_info(
            api_base, api_key
        )

    if api_base is not None and not isinstance(api_base, str):
        raise Exception("api base needs to be a string. api_base={}".format(api_base))
    if dynamic_api_key is not None and not isinstance(dynamic_api_key, str):
        raise Exception(
            "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                dynamic_api_key
            )
        )
    if dynamic_api_key is None and api_key is not None:
        dynamic_api_key = api_key
    return model, custom_llm_provider, dynamic_api_key, api_base

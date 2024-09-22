from typing import Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.secret_managers.main import get_secret

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


def _is_azure_openai_model(model: str) -> bool:
    try:
        if "/" in model:
            model = model.split("/", 1)[1]
        if (
            model in litellm.open_ai_chat_completion_models
            or model in litellm.open_ai_text_completion_models
            or model in litellm.open_ai_embedding_models
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


def get_llm_provider(
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
        ## IF LITELLM PARAMS GIVEN ##
        if litellm_params is not None:
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

        if custom_llm_provider:
            if (
                model.split("/")[0] == custom_llm_provider
            ):  # handle scenario where model="azure/*" and custom_llm_provider="azure"
                model = model.replace("{}/".format(custom_llm_provider), "")

            return model, custom_llm_provider, dynamic_api_key, api_base

        if api_key and api_key.startswith("os.environ/"):
            dynamic_api_key = get_secret(api_key)
        # check if llm provider part of model name
        if (
            model.split("/", 1)[0] in litellm.provider_list
            and model.split("/", 1)[0] not in litellm.model_list
            and len(model.split("/"))
            > 1  # handle edge case where user passes in `litellm --model mistral` https://github.com/BerriAI/litellm/issues/1351
        ):
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]

            if custom_llm_provider == "perplexity":
                # perplexity is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.perplexity.ai
                api_base = api_base or get_secret("PERPLEXITY_API_BASE") or "https://api.perplexity.ai"  # type: ignore
                dynamic_api_key = (
                    api_key
                    or get_secret("PERPLEXITYAI_API_KEY")
                    or get_secret("PERPLEXITY_API_KEY")
                )
            elif custom_llm_provider == "anyscale":
                # anyscale is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = api_base or get_secret("ANYSCALE_API_BASE") or "https://api.endpoints.anyscale.com/v1"  # type: ignore
                dynamic_api_key = api_key or get_secret("ANYSCALE_API_KEY")
            elif custom_llm_provider == "deepinfra":
                # deepinfra is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = api_base or get_secret("DEEPINFRA_API_BASE") or "https://api.deepinfra.com/v1/openai"  # type: ignore
                dynamic_api_key = api_key or get_secret("DEEPINFRA_API_KEY")
            elif custom_llm_provider == "empower":
                api_base = (
                    api_base
                    or get_secret("EMPOWER_API_BASE")
                    or "https://app.empower.dev/api/v1"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("EMPOWER_API_KEY")
            elif custom_llm_provider == "groq":
                # groq is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.groq.com/openai/v1
                api_base = (
                    api_base
                    or get_secret("GROQ_API_BASE")
                    or "https://api.groq.com/openai/v1"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("GROQ_API_KEY")
            elif custom_llm_provider == "nvidia_nim":
                # nvidia_nim is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = (
                    api_base
                    or get_secret("NVIDIA_NIM_API_BASE")
                    or "https://integrate.api.nvidia.com/v1"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("NVIDIA_NIM_API_KEY")
            elif custom_llm_provider == "cerebras":
                api_base = (
                    api_base
                    or get_secret("CEREBRAS_API_BASE")
                    or "https://api.cerebras.ai/v1"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("CEREBRAS_API_KEY")
            elif custom_llm_provider == "sambanova":
                api_base = (
                    api_base
                    or get_secret("SAMBANOVA_API_BASE")
                    or "https://api.sambanova.ai/v1"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("SAMBANOVA_API_KEY")
            elif (custom_llm_provider == "ai21_chat") or (
                custom_llm_provider == "ai21" and model in litellm.ai21_chat_models
            ):
                api_base = (
                    api_base
                    or get_secret("AI21_API_BASE")
                    or "https://api.ai21.com/studio/v1"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("AI21_API_KEY")
                custom_llm_provider = "ai21_chat"
            elif custom_llm_provider == "volcengine":
                # volcengine is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = (
                    api_base
                    or get_secret("VOLCENGINE_API_BASE")
                    or "https://ark.cn-beijing.volces.com/api/v3"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("VOLCENGINE_API_KEY")
            elif custom_llm_provider == "codestral":
                # codestral is openai compatible, we just need to set this to custom_openai and have the api_base be https://codestral.mistral.ai/v1
                api_base = (
                    api_base
                    or get_secret("CODESTRAL_API_BASE")
                    or "https://codestral.mistral.ai/v1"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("CODESTRAL_API_KEY")
            elif custom_llm_provider == "deepseek":
                # deepseek is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.deepseek.com/v1
                api_base = (
                    api_base
                    or get_secret("DEEPSEEK_API_BASE")
                    or "https://api.deepseek.com/beta"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("DEEPSEEK_API_KEY")
            elif custom_llm_provider == "fireworks_ai":
                # fireworks is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.fireworks.ai/inference/v1
                if litellm.FireworksAIEmbeddingConfig().is_fireworks_embedding_model(
                    model=model
                ):
                    # fireworks embeddings models do no require accounts/fireworks prefix https://docs.fireworks.ai/api-reference/creates-an-embedding-vector-representing-the-input-text
                    pass
                elif not model.startswith("accounts/"):
                    model = f"accounts/fireworks/models/{model}"
                api_base = (
                    api_base
                    or get_secret("FIREWORKS_API_BASE")
                    or "https://api.fireworks.ai/inference/v1"
                )  # type: ignore
                dynamic_api_key = api_key or (
                    get_secret("FIREWORKS_API_KEY")
                    or get_secret("FIREWORKS_AI_API_KEY")
                    or get_secret("FIREWORKSAI_API_KEY")
                    or get_secret("FIREWORKS_AI_TOKEN")
                )
            elif custom_llm_provider == "azure_ai":
                api_base = api_base or get_secret("AZURE_AI_API_BASE")  # type: ignore
                dynamic_api_key = api_key or get_secret("AZURE_AI_API_KEY")

                if _is_azure_openai_model(model=model):
                    verbose_logger.debug(
                        "Model={} is Azure OpenAI model. Setting custom_llm_provider='azure'.".format(
                            model
                        )
                    )
                    custom_llm_provider = "azure"
            elif custom_llm_provider == "github":
                api_base = api_base or get_secret("GITHUB_API_BASE") or "https://models.inference.ai.azure.com"  # type: ignore
                dynamic_api_key = api_key or get_secret("GITHUB_API_KEY")
            elif custom_llm_provider == "litellm_proxy":
                api_base = api_base or get_secret("LITELLM_PROXY_API_BASE") or "https://models.inference.ai.azure.com"  # type: ignore
                dynamic_api_key = api_key or get_secret("LITELLM_PROXY_API_KEY")

            elif custom_llm_provider == "mistral":
                # mistral is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.mistral.ai
                api_base = (
                    api_base
                    or get_secret("MISTRAL_AZURE_API_BASE")  # for Azure AI Mistral
                    or "https://api.mistral.ai/v1"
                )  # type: ignore

                # if api_base does not end with /v1 we add it
                if api_base is not None and not api_base.endswith(
                    "/v1"
                ):  # Mistral always needs a /v1 at the end
                    api_base = api_base + "/v1"
                dynamic_api_key = (
                    api_key
                    or get_secret("MISTRAL_AZURE_API_KEY")  # for Azure AI Mistral
                    or get_secret("MISTRAL_API_KEY")
                )
            elif custom_llm_provider == "voyage":
                # voyage is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.voyageai.com/v1
                api_base = (
                    api_base
                    or get_secret("VOYAGE_API_BASE")
                    or "https://api.voyageai.com/v1"
                )  # type: ignore
                dynamic_api_key = api_key or get_secret("VOYAGE_API_KEY")
            elif custom_llm_provider == "together_ai":
                api_base = (
                    api_base
                    or get_secret("TOGETHER_AI_API_BASE")
                    or "https://api.together.xyz/v1"
                )  # type: ignore
                dynamic_api_key = api_key or (
                    get_secret("TOGETHER_API_KEY")
                    or get_secret("TOGETHER_AI_API_KEY")
                    or get_secret("TOGETHERAI_API_KEY")
                    or get_secret("TOGETHER_AI_TOKEN")
                )
            elif custom_llm_provider == "friendliai":
                api_base = (
                    api_base
                    or get_secret("FRIENDLI_API_BASE")
                    or "https://inference.friendli.ai/v1"
                )  # type: ignore
                dynamic_api_key = (
                    api_key
                    or get_secret("FRIENDLIAI_API_KEY")
                    or get_secret("FRIENDLI_TOKEN")
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
                        dynamic_api_key = get_secret("PERPLEXITYAI_API_KEY")
                    elif endpoint == "api.endpoints.anyscale.com/v1":
                        custom_llm_provider = "anyscale"
                        dynamic_api_key = get_secret("ANYSCALE_API_KEY")
                    elif endpoint == "api.deepinfra.com/v1/openai":
                        custom_llm_provider = "deepinfra"
                        dynamic_api_key = get_secret("DEEPINFRA_API_KEY")
                    elif endpoint == "api.mistral.ai/v1":
                        custom_llm_provider = "mistral"
                        dynamic_api_key = get_secret("MISTRAL_API_KEY")
                    elif endpoint == "api.groq.com/openai/v1":
                        custom_llm_provider = "groq"
                        dynamic_api_key = get_secret("GROQ_API_KEY")
                    elif endpoint == "https://integrate.api.nvidia.com/v1":
                        custom_llm_provider = "nvidia_nim"
                        dynamic_api_key = get_secret("NVIDIA_NIM_API_KEY")
                    elif endpoint == "https://api.cerebras.ai/v1":
                        custom_llm_provider = "cerebras"
                        dynamic_api_key = get_secret("CEREBRAS_API_KEY")
                    elif endpoint == "https://api.sambanova.ai/v1":
                        custom_llm_provider = "sambanova"
                        dynamic_api_key = get_secret("SAMBANOVA_API_KEY")
                    elif endpoint == "https://api.ai21.com/studio/v1":
                        custom_llm_provider = "ai21_chat"
                        dynamic_api_key = get_secret("AI21_API_KEY")
                    elif endpoint == "https://codestral.mistral.ai/v1":
                        custom_llm_provider = "codestral"
                        dynamic_api_key = get_secret("CODESTRAL_API_KEY")
                    elif endpoint == "https://codestral.mistral.ai/v1":
                        custom_llm_provider = "text-completion-codestral"
                        dynamic_api_key = get_secret("CODESTRAL_API_KEY")
                    elif endpoint == "app.empower.dev/api/v1":
                        custom_llm_provider = "empower"
                        dynamic_api_key = get_secret("EMPOWER_API_KEY")
                    elif endpoint == "api.deepseek.com/v1":
                        custom_llm_provider = "deepseek"
                        dynamic_api_key = get_secret("DEEPSEEK_API_KEY")
                    elif endpoint == "inference.friendli.ai/v1":
                        custom_llm_provider = "friendliai"
                        dynamic_api_key = get_secret(
                            "FRIENDLIAI_API_KEY"
                        ) or get_secret("FRIENDLI_TOKEN")

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
            custom_llm_provider = "anthropic"
        ## cohere
        elif model in litellm.cohere_models or model in litellm.cohere_embedding_models:
            custom_llm_provider = "cohere"
        ## cohere chat models
        elif model in litellm.cohere_chat_models:
            custom_llm_provider = "cohere_chat"
        ## replicate
        elif model in litellm.replicate_models or (":" in model and len(model) > 64):
            model_parts = model.split(":")
            if (
                len(model_parts) > 1 and len(model_parts[1]) == 64
            ):  ## checks if model name has a 64 digit code - e.g. "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
                custom_llm_provider = "replicate"
            elif model in litellm.replicate_models:
                custom_llm_provider = "replicate"
        ## openrouter
        elif model in litellm.openrouter_models:
            custom_llm_provider = "openrouter"
        ## openrouter
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
        elif model in litellm.ai21_models:
            custom_llm_provider = "ai21"
        elif model in litellm.ai21_chat_models:
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
            model in litellm.bedrock_models or model in litellm.bedrock_embedding_models
        ):
            custom_llm_provider = "bedrock"
        elif model in litellm.watsonx_models:
            custom_llm_provider = "watsonx"
        # openai embeddings
        elif model in litellm.open_ai_embedding_models:
            custom_llm_provider = "openai"
        elif model in litellm.empower_models:
            custom_llm_provider = "empower"
        elif model == "*":
            custom_llm_provider = "openai"
        if custom_llm_provider is None or custom_llm_provider == "":
            if litellm.suppress_debug_info == False:
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

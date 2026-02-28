from litellm.llms.base_llm.chat.transformation import BaseLLMException

# Native OpenRouter models whose IDs start with "openrouter/".
# When used via LiteLLM (openrouter/openrouter/free), get_llm_provider()
# must not strip the inner "openrouter/" prefix on its second invocation.
# See: https://github.com/BerriAI/litellm/issues/16353
NATIVE_OPENROUTER_MODELS = {
    "openrouter/auto",
    "openrouter/free",
    "openrouter/bodybuilder",
}


class OpenRouterException(BaseLLMException):
    pass

import random
from collections.abc import Iterator
from itertools import islice, product
from typing import Final

import pytest

import litellm
from litellm.constants import openai_compatible_endpoints
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider, resolve_llm_provider
from litellm.litellm_core_utils.model_latest_aliases import latest_alias_rows
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.rust_bridge import _native

Case = tuple[str, str | None, str | None, str | None]
Outcome = tuple[object, ...]

SECRET_NAMES: Final = (
    "USE_LITELLM_PROXY",
    "LITELLM_PROXY_API_BASE",
    "LITELLM_PROXY_API_KEY",
    "AI21_API_BASE",
    "AI21_API_KEY",
    "PERPLEXITY_API_BASE",
    "PERPLEXITYAI_API_KEY",
    "PERPLEXITY_API_KEY",
    "ANYSCALE_API_BASE",
    "ANYSCALE_API_KEY",
    "DEEPINFRA_API_BASE",
    "DEEPINFRA_API_KEY",
    "CLF_AI_GATEWAY_API_BASE",
    "CLF_AI_GATEWAY_API_KEY",
    "EMPOWER_API_BASE",
    "EMPOWER_API_KEY",
    "GROQ_API_BASE",
    "GROQ_API_KEY",
    "NVIDIA_NIM_API_BASE",
    "NVIDIA_NIM_API_KEY",
    "NVIDIA_RIVA_API_BASE",
    "NVIDIA_RIVA_API_KEY",
    "SONIOX_API_BASE",
    "SONIOX_API_KEY",
    "CEREBRAS_API_BASE",
    "CEREBRAS_API_KEY",
    "NADIR_API_BASE",
    "NADIR_API_KEY",
    "SAMBANOVA_API_BASE",
    "SAMBANOVA_API_KEY",
    "LLAMA_API_BASE",
    "LLAMA_API_KEY",
    "NEBIUS_API_BASE",
    "NEBIUS_API_KEY",
    "OLLAMA_API_BASE",
    "OLLAMA_API_KEY",
    "VOLCENGINE_API_BASE",
    "VOLCENGINE_API_KEY",
    "CODESTRAL_API_BASE",
    "CODESTRAL_API_KEY",
    "HOSTED_VLLM_API_BASE",
    "HOSTED_VLLM_API_KEY",
    "LM_STUDIO_API_BASE",
    "LM_STUDIO_API_KEY",
    "DEEPSEEK_API_BASE",
    "DEEPSEEK_API_KEY",
    "TENCENT_API_BASE",
    "TENCENT_API_KEY",
    "FIREWORKS_API_BASE",
    "FIREWORKS_API_KEY",
    "FIREWORKS_AI_API_KEY",
    "GITHUB_API_BASE",
    "GITHUB_API_KEY",
    "JINA_AI_API_BASE",
    "JINA_AI_API_KEY",
    "JINA_API_KEY",
    "TOGETHER_AI_API_BASE",
    "TOGETHER_API_KEY",
    "TOGETHERAI_API_KEY",
    "FRIENDLI_API_BASE",
    "FRIENDLIAI_API_KEY",
    "FRIENDLI_TOKEN",
    "GALADRIEL_API_BASE",
    "GALADRIEL_API_KEY",
    "NOVITA_API_BASE",
    "NOVITA_API_KEY",
    "FEATHERLESS_AI_API_BASE",
    "FEATHERLESS_API_BASE",
    "FEATHERLESS_AI_API_KEY",
    "FEATHERLESS_API_KEY",
    "HEROKU_API_BASE",
    "HEROKU_API_KEY",
    "MOONSHOT_API_BASE",
    "MOONSHOT_API_KEY",
    "DOCKER_MODEL_RUNNER_API_BASE",
    "DOCKER_MODEL_RUNNER_API_KEY",
    "V0_API_BASE",
    "V0_API_KEY",
    "MORPH_API_BASE",
    "MORPH_API_KEY",
    "LAMBDA_API_BASE",
    "LAMBDA_API_KEY",
    "HYPERBOLIC_API_BASE",
    "HYPERBOLIC_API_KEY",
    "VERCEL_AI_GATEWAY_API_BASE",
    "VERCEL_AI_GATEWAY_API_KEY",
    "VERCEL_OIDC_TOKEN",
    "WANDB_API_BASE",
    "WANDB_API_KEY",
    "LANGGRAPH_API_BASE",
    "LANGGRAPH_API_KEY",
    "MANUS_API_BASE",
    "MANUS_API_KEY",
    "GIGACHAT_API_BASE",
    "GIGACHAT_API_KEY",
    "MISTRAL_API_KEY",
    "BASETEN_API_KEY",
    "DASHSCOPE_API_KEY",
    "MODELSCOPE_API_KEY",
    "MINIMAX_API_KEY",
    "PUBLICAI_API_KEY",
    "PUBLICAI_API_BASE",
    "SYNTHETIC_API_KEY",
    "STIMA_API_KEY",
    "NANOGPT_API_KEY",
    "POE_API_KEY",
    "CHUTES_API_KEY",
    "INCEPTION_API_KEY",
    "EDENAI_API_KEY",
    "PINSTRIPES_API_KEY",
    "META_API_KEY",
)
ENVIRONMENTS: Final = {
    "unset": {},
    "text": {"USE_LITELLM_PROXY": "no", **{name: f"env-{name.lower()}" for name in SECRET_NAMES[1:]}},
    "flags": {name: ("True", "false", "", " TRUE ")[index % 4] for index, name in enumerate(SECRET_NAMES[1:])},
    "proxy": {"USE_LITELLM_PROXY": "true", "LITELLM_PROXY_API_BASE": "https://proxy.example"},
}
NAMES: Final = (
    "",
    "*",
    "/",
    "x/",
    "/x",
    "mistral",
    "openai",
    "gpt-4o",
    "ft:gpt-4o-mini:org::id",
    "ft:gpt-3.5-turbo:x",
    "gpt-image-1",
    "sora-2",
    "claude-2",
    "claude-instant-1",
    "claude-sonnet-4-5",
    "claude-opus-latest",
    "claude-future-9-1",
    "claude-sonnet-4-5\n",
    "Claude-Sonnet-4-5",
    "clàude-sonnet-4-5",
    "anthropic.claude-v2",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "command-r",
    "command-r-plus",
    "embed-english-v3.0",
    "meta/llama-2-70b-chat:" + "a" * 64,
    "owner/model:" + "b" * 63,
    "x" * 70 + ":y",
    "gemini-pro",
    "gemini-2.5-pro",
    "j2-ultra",
    "jamba-1.5-large",
    "luminous-base",
    "llama-3-1-8b",
    "chat-bison",
    "amazon.titan-embed-text-v1",
    "text-embedding-3-small",
    "jev-latest",
    "unknown-model",
    "bytez/x",
    "gdc/x",
    "lemonade/x",
    "heroku/x",
    "cometapi/x",
    "oci/x",
    "compactifai/x",
    "ovhcloud/x",
    "clarifai/x",
    "amazon_nova-pro",
    "sap/x",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-instruct",
    "openrouter/auto",
    "openrouter/anthropic/claude-3.5-sonnet",
    "azure/command-r-plus",
    "azure/mistral-large",
    "azure/gpt-4o",
    "azure",
    "cohere/command-r",
    "anthropic/claude-2",
    "litellm_proxy/gpt-4o",
    "test-onprem-llm/model",
    "my-custom/model",
    "a/b/c",
)
API_BASES: Final = (
    None,
    "",
    "https://example.com/v1",
    "http://localhost:4000",
    "https://attacker.com/api.groq.com/openai/v1",
    "https://API.GROQ.COM/openai/v1/",
    "https://api.groq.com/openai/v10",
    "https://u:p@api.groq.com:443/openai/v1;x",
    "https://[::1]:8080/v1",
    " https://api.groq.com/openai/v1",
    "https://api.getnadir.com/v1",
    "http://api.getnadir.com/v1",
    "api.together.xyz/v1?x=1",
    "https://ai-gateway.helicone.ai/",
    "https://inference.api.nscale.com/v1",
)
API_KEYS: Final = (None, "", "sk-caller")


def _providers() -> tuple[str, ...]:
    return (*(str(provider.value) for provider in litellm.LlmProviders), *JSONProviderRegistry.list_providers())


def _listed_models() -> tuple[str, ...]:
    return tuple(sorted({model for models in litellm.models_by_provider.values() for model in models}))


def _models(rng: random.Random) -> tuple[str, ...]:
    providers: Final = _providers()
    listed: Final = _listed_models()
    prefixed: Final = tuple(f"{provider}/{rng.choice(listed)}" for provider in providers)
    doubly: Final = tuple(f"{provider}/{provider}/{rng.choice(NAMES)}" for provider in providers)
    aliases: Final = tuple(key for key, _, _ in latest_alias_rows())
    return (*NAMES, *listed, *prefixed, *doubly, *aliases)


def _api_bases(rng: random.Random) -> tuple[str | None, ...]:
    endpoints: Final = tuple(openai_compatible_endpoints)
    variants: Final = tuple(
        variant
        for endpoint in endpoints
        for variant in (endpoint, f"https://{endpoint.removeprefix('https://')}/chat", endpoint.upper())
    )
    return (*API_BASES, *variants, *(rng.choice(variants) + "/x" for _ in range(20)))


def corpus(seed: int, size: int) -> Iterator[Case]:
    rng: Final = random.Random(seed)
    models: Final = _models(rng)
    providers: Final = (None, None, None, "", "foo", "cohere", "anthropic", "openrouter", *_providers())
    api_bases: Final = _api_bases(rng)
    exhaustive: Final = product(NAMES, (None, "", "anthropic", "cohere", "openai", "foo"), API_BASES[:6], API_KEYS)
    sampled: Final = (
        (rng.choice(models), rng.choice(providers), rng.choice(api_bases), rng.choice(API_KEYS)) for _ in range(size)
    )
    yield from exhaustive
    yield from sampled


def _outcome(resolve: object, case: Case, capsys: pytest.CaptureFixture[str]) -> Outcome:
    try:
        resolved: Final = resolve(*case)  # pyright: ignore[reportCallIssue]  # both resolvers share this signature
    except litellm.BadRequestError as error:
        return ("raised", type(error), str(error), error.model, error.llm_provider, capsys.readouterr().out)
    return ("returned", resolved, capsys.readouterr().out)


def _native_outcome(case: Case, capsys: pytest.CaptureFixture[str]) -> Outcome | None:
    if _native.get_llm_provider(*case) is None:
        return None
    return _outcome(resolve_llm_provider, case, capsys)


@pytest.mark.parametrize("environment", tuple(ENVIRONMENTS))
def test_native_get_llm_provider_matches_python_on_a_generated_corpus(
    environment: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in ENVIRONMENTS[environment].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "test-onprem-llm", "custom_handler": object()}])
    compared: Final = tuple(
        (case, native, _outcome(get_llm_provider, case, capsys))
        for case in corpus(seed=len(environment), size=12_000)
        if (native := _native_outcome(case, capsys)) is not None
    )
    mismatches: Final = tuple(
        islice(((case, native, python) for case, native, python in compared if native != python), 10)
    )
    assert mismatches == ()
    assert len(compared) > 12_000


@pytest.mark.parametrize(
    ("model", "api_base"),
    (("claude-sonnet-4-5", None), ("groq/llama-3.3-70b", "https://api.groq.com/openai/v1"), ("unknown-model", None)),
)
def test_resolve_llm_provider_follows_a_runtime_switch_to_the_litellm_proxy(
    model: str, api_base: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("USE_LITELLM_PROXY", raising=False)
    resolve_llm_provider("claude-sonnet-4-5")
    monkeypatch.setattr(litellm, "use_litellm_proxy", True)
    assert resolve_llm_provider(model, api_base=api_base) == get_llm_provider(model, api_base=api_base)
    assert resolve_llm_provider(model, api_base=api_base)[1] == "litellm_proxy"


def test_resolve_llm_provider_sees_a_custom_provider_registered_after_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    with pytest.raises(litellm.BadRequestError):
        resolve_llm_provider("late-provider/model")
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "late-provider", "custom_handler": object()}])
    assert resolve_llm_provider("late-provider/model") == ("model", "late-provider", None, None)


def test_resolve_llm_provider_sees_a_model_registered_after_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    with pytest.raises(litellm.BadRequestError):
        resolve_llm_provider("native-parity-late-model")
    litellm.register_model(
        {"native-parity-late-model": {"litellm_provider": "anthropic", "mode": "chat", "max_tokens": 8}}
    )
    assert resolve_llm_provider("native-parity-late-model") == get_llm_provider("native-parity-late-model")
    assert resolve_llm_provider("native-parity-late-model")[1] == "anthropic"

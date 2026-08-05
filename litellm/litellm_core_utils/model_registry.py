"""Immutable provider/model registry derived from the litellm cost map.

Every legacy ``litellm.<provider>_models`` collection, ``models_by_provider``,
``model_list`` and ``all_embedding_models`` is a view over one snapshot built
here, so a cost-map reload can never leave them stale relative to each other.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import groupby
from typing import assert_never

from litellm.constants import (
    BEDROCK_CONVERSE_MODELS,
    WANDB_MODELS,
    baseten_models,
    bedrock_embedding_models,
    clarifai_models,
    cohere_embedding_models,
    empower_models,
    huggingface_models,
    modelscope_models,
    open_ai_embedding_models,
    replicate_models,
    together_ai_models,
)

_BEDROCK_PRICING_ONLY_PATTERN = re.compile(r"^bedrock/[a-zA-Z0-9_-]+/.+$")
_VERTEX_KEY_PREFIX = "vertex_ai/"


def is_bedrock_pricing_only_model(key: str) -> bool:
    """Keys shaped ``bedrock/<region>/<model>`` exist for pricing only, not for routing."""
    if "month-commitment" in key:
        return True
    return _BEDROCK_PRICING_ONLY_PATTERN.match(key) is not None


def is_openai_finetune_model(key: str) -> bool:
    """Keys shaped ``ft:<model>`` exist for pricing only, not for routing."""
    return key.startswith("ft:") and not key.count(":") > 1


def _is_enumerable_openai_model(key: str) -> bool:
    return not is_openai_finetune_model(key)


def _is_enumerable_bedrock_model(key: str) -> bool:
    return not is_bedrock_pricing_only_model(key)


def _is_enumerable_fireworks_model(key: str) -> bool:
    return "-to-" not in key and "fireworks-ai-default" not in key


def _is_enumerable_fireworks_embedding_model(key: str) -> bool:
    return "-to-" not in key


@dataclass(frozen=True, slots=True)
class _LegacyTarget:
    set_name: str
    strip_vertex_prefix: bool = False
    key_filter: Callable[[str], bool] | None = None


@dataclass(frozen=True, slots=True)
class _ModeSplitTarget:
    chat_set_name: str
    default_set_name: str


_ProviderRule = _LegacyTarget | _ModeSplitTarget


@dataclass(frozen=True, slots=True)
class _LegacyMember:
    set_name: str
    model: str


@dataclass(frozen=True, slots=True)
class _ProviderMember:
    provider: str
    model: str


_DerivedMember = _LegacyMember | _ProviderMember


@dataclass(frozen=True, slots=True)
class ModelRegistrySnapshot:
    """One atomically-swappable view of everything derived from the cost map."""

    legacy_sets: Mapping[str, frozenset[str]]
    models_by_provider: Mapping[str, frozenset[str]]
    model_list: tuple[str, ...]
    model_list_set: frozenset[str]
    all_embedding_models: frozenset[str]


_PROVIDER_RULES: Mapping[str, _ProviderRule] = {
    "openai": _LegacyTarget(set_name="open_ai_chat_completion_models", key_filter=_is_enumerable_openai_model),
    "text-completion-openai": _LegacyTarget(set_name="open_ai_text_completion_models"),
    "azure_text": _LegacyTarget(set_name="azure_text_models"),
    "cohere": _LegacyTarget(set_name="cohere_models"),
    "cohere_chat": _LegacyTarget(set_name="cohere_chat_models"),
    "mistral": _LegacyTarget(set_name="mistral_chat_models"),
    "anthropic": _LegacyTarget(set_name="anthropic_models"),
    "empower": _LegacyTarget(set_name="empower_models"),
    "openrouter": _LegacyTarget(set_name="openrouter_models"),
    "vercel_ai_gateway": _LegacyTarget(set_name="vercel_ai_gateway_models"),
    "datarobot": _LegacyTarget(set_name="datarobot_models"),
    "vertex_ai-text-models": _LegacyTarget(set_name="vertex_text_models"),
    "vertex_ai-code-text-models": _LegacyTarget(set_name="vertex_code_text_models"),
    "vertex_ai-language-models": _LegacyTarget(set_name="vertex_language_models"),
    "vertex_ai-vision-models": _LegacyTarget(set_name="vertex_vision_models"),
    "vertex_ai-chat-models": _LegacyTarget(set_name="vertex_chat_models"),
    "vertex_ai-code-chat-models": _LegacyTarget(set_name="vertex_code_chat_models"),
    "vertex_ai-embedding-models": _LegacyTarget(set_name="vertex_embedding_models"),
    "vertex_ai-anthropic_models": _LegacyTarget(set_name="vertex_anthropic_models", strip_vertex_prefix=True),
    "vertex_ai-llama_models": _LegacyTarget(set_name="vertex_llama3_models", strip_vertex_prefix=True),
    "vertex_ai-deepseek_models": _LegacyTarget(set_name="vertex_deepseek_models", strip_vertex_prefix=True),
    "vertex_ai-mistral_models": _LegacyTarget(set_name="vertex_mistral_models", strip_vertex_prefix=True),
    "vertex_ai-ai21_models": _LegacyTarget(set_name="vertex_ai_ai21_models", strip_vertex_prefix=True),
    "vertex_ai-image-models": _LegacyTarget(set_name="vertex_ai_image_models", strip_vertex_prefix=True),
    "vertex_ai-video-models": _LegacyTarget(set_name="vertex_ai_video_models", strip_vertex_prefix=True),
    "vertex_ai-openai_models": _LegacyTarget(set_name="vertex_openai_models", strip_vertex_prefix=True),
    "vertex_ai-minimax_models": _LegacyTarget(set_name="vertex_minimax_models", strip_vertex_prefix=True),
    "vertex_ai-moonshot_models": _LegacyTarget(set_name="vertex_moonshot_models", strip_vertex_prefix=True),
    "vertex_ai-zai_models": _LegacyTarget(set_name="vertex_zai_models", strip_vertex_prefix=True),
    "ai21": _ModeSplitTarget(chat_set_name="ai21_chat_models", default_set_name="ai21_models"),
    "nlp_cloud": _LegacyTarget(set_name="nlp_cloud_models"),
    "aleph_alpha": _LegacyTarget(set_name="aleph_alpha_models"),
    "bedrock": _LegacyTarget(set_name="bedrock_models", key_filter=_is_enumerable_bedrock_model),
    "bedrock_converse": _LegacyTarget(set_name="bedrock_converse_models"),
    "deepinfra": _LegacyTarget(set_name="deepinfra_models"),
    "perplexity": _LegacyTarget(set_name="perplexity_models"),
    "watsonx": _LegacyTarget(set_name="watsonx_models"),
    "gemini": _LegacyTarget(set_name="gemini_models"),
    "fireworks_ai": _LegacyTarget(set_name="fireworks_ai_models", key_filter=_is_enumerable_fireworks_model),
    "fireworks_ai-embedding-models": _LegacyTarget(
        set_name="fireworks_ai_embedding_models", key_filter=_is_enumerable_fireworks_embedding_model
    ),
    "text-completion-codestral": _LegacyTarget(set_name="text_completion_codestral_models"),
    "text-completion-inception": _LegacyTarget(set_name="text_completion_inception_models"),
    "xai": _LegacyTarget(set_name="xai_models"),
    "zai": _LegacyTarget(set_name="zai_models"),
    "fal_ai": _LegacyTarget(set_name="fal_ai_models"),
    "deepseek": _LegacyTarget(set_name="deepseek_models"),
    "tencent": _LegacyTarget(set_name="tencent_models"),
    "runwayml": _LegacyTarget(set_name="runwayml_models"),
    "meta_llama": _LegacyTarget(set_name="llama_models"),
    "nscale": _LegacyTarget(set_name="nscale_models"),
    "azure_ai": _LegacyTarget(set_name="azure_ai_models"),
    "voyage": _LegacyTarget(set_name="voyage_models"),
    "infinity": _LegacyTarget(set_name="infinity_models"),
    "databricks": _LegacyTarget(set_name="databricks_models"),
    "cloudflare": _LegacyTarget(set_name="cloudflare_models"),
    "codestral": _LegacyTarget(set_name="codestral_models"),
    "friendliai": _LegacyTarget(set_name="friendliai_models"),
    "palm": _LegacyTarget(set_name="palm_models"),
    "groq": _LegacyTarget(set_name="groq_models"),
    "azure": _LegacyTarget(set_name="azure_models"),
    "azure_anthropic": _LegacyTarget(set_name="azure_anthropic_models"),
    "anyscale": _LegacyTarget(set_name="anyscale_models"),
    "cerebras": _LegacyTarget(set_name="cerebras_models"),
    "galadriel": _LegacyTarget(set_name="galadriel_models"),
    "nvidia_nim": _LegacyTarget(set_name="nvidia_nim_models"),
    "nvidia_riva": _LegacyTarget(set_name="nvidia_riva_models"),
    "soniox": _LegacyTarget(set_name="soniox_models"),
    "sambanova": _LegacyTarget(set_name="sambanova_models"),
    "sambanova-embedding-models": _LegacyTarget(set_name="sambanova_embedding_models"),
    "novita": _LegacyTarget(set_name="novita_models"),
    "nebius-chat-models": _LegacyTarget(set_name="nebius_models"),
    "nebius-embedding-models": _LegacyTarget(set_name="nebius_embedding_models"),
    "aiml": _LegacyTarget(set_name="aiml_models"),
    "assemblyai": _LegacyTarget(set_name="assemblyai_models"),
    "jina_ai": _LegacyTarget(set_name="jina_ai_models"),
    "snowflake": _LegacyTarget(set_name="snowflake_models"),
    "gradient_ai": _LegacyTarget(set_name="gradient_ai_models"),
    "featherless_ai": _LegacyTarget(set_name="featherless_ai_models"),
    "deepgram": _LegacyTarget(set_name="deepgram_models"),
    "elevenlabs": _LegacyTarget(set_name="elevenlabs_models"),
    "heroku": _LegacyTarget(set_name="heroku_models"),
    "dashscope": _LegacyTarget(set_name="dashscope_models"),
    "modelscope": _LegacyTarget(set_name="modelscope_models"),
    "moonshot": _LegacyTarget(set_name="moonshot_models"),
    "publicai": _LegacyTarget(set_name="publicai_models"),
    "darkbloom": _LegacyTarget(set_name="darkbloom_models"),
    "v0": _LegacyTarget(set_name="v0_models"),
    "morph": _LegacyTarget(set_name="morph_models"),
    "lambda_ai": _LegacyTarget(set_name="lambda_ai_models"),
    "inception": _LegacyTarget(set_name="inception_models"),
    "hyperbolic": _LegacyTarget(set_name="hyperbolic_models"),
    "black_forest_labs": _LegacyTarget(set_name="black_forest_labs_models"),
    "recraft": _LegacyTarget(set_name="recraft_models"),
    "cometapi": _LegacyTarget(set_name="cometapi_models"),
    "oci": _LegacyTarget(set_name="oci_models"),
    "volcengine": _LegacyTarget(set_name="volcengine_models"),
    "wandb": _LegacyTarget(set_name="wandb_models"),
    "ovhcloud": _LegacyTarget(set_name="ovhcloud_models"),
    "ovhcloud-embedding-models": _LegacyTarget(set_name="ovhcloud_embedding_models"),
    "lemonade": _LegacyTarget(set_name="lemonade_models"),
    "docker_model_runner": _LegacyTarget(set_name="docker_model_runner_models"),
    "amazon_nova": _LegacyTarget(set_name="amazon_nova_models"),
    "stability": _LegacyTarget(set_name="stability_models"),
    "github_copilot": _LegacyTarget(set_name="github_copilot_models"),
    "chatgpt": _LegacyTarget(set_name="chatgpt_models"),
    "minimax": _LegacyTarget(set_name="minimax_models"),
    "aws_polly": _LegacyTarget(set_name="aws_polly_models"),
    "gigachat": _LegacyTarget(set_name="gigachat_models"),
    "llamagate": _LegacyTarget(set_name="llamagate_models"),
    "reducto": _LegacyTarget(set_name="reducto_models"),
    "bedrock_mantle": _LegacyTarget(set_name="bedrock_mantle_models"),
}

_PROVIDER_COMPOSITION: Mapping[str, tuple[str, ...]] = {
    "openai": ("open_ai_chat_completion_models", "open_ai_text_completion_models"),
    "text-completion-openai": ("open_ai_text_completion_models",),
    "cohere": ("cohere_models", "cohere_chat_models"),
    "cohere_chat": ("cohere_chat_models",),
    "anthropic": ("anthropic_models",),
    "replicate": ("replicate_models",),
    "huggingface": ("huggingface_models",),
    "together_ai": ("together_ai_models",),
    "baseten": ("baseten_models",),
    "openrouter": ("openrouter_models",),
    "vercel_ai_gateway": ("vercel_ai_gateway_models",),
    "datarobot": ("datarobot_models",),
    "vertex_ai": (
        "vertex_chat_models",
        "vertex_text_models",
        "vertex_anthropic_models",
        "vertex_vision_models",
        "vertex_language_models",
        "vertex_deepseek_models",
        "vertex_minimax_models",
        "vertex_moonshot_models",
        "vertex_zai_models",
    ),
    "ai21": ("ai21_models",),
    "bedrock": ("bedrock_models", "bedrock_converse_models"),
    "petals": ("petals_models",),
    "ollama": ("ollama_models",),
    "ollama_chat": ("ollama_models",),
    "deepinfra": ("deepinfra_models",),
    "perplexity": ("perplexity_models",),
    "maritalk": ("maritalk_models",),
    "watsonx": ("watsonx_models",),
    "gemini": ("gemini_models",),
    "fireworks_ai": ("fireworks_ai_models", "fireworks_ai_embedding_models"),
    "aleph_alpha": ("aleph_alpha_models",),
    "text-completion-codestral": ("text_completion_codestral_models",),
    "text-completion-inception": ("text_completion_inception_models",),
    "xai": ("xai_models",),
    "zai": ("zai_models",),
    "fal_ai": ("fal_ai_models",),
    "deepseek": ("deepseek_models",),
    "tencent": ("tencent_models",),
    "runwayml": ("runwayml_models",),
    "mistral": ("mistral_chat_models",),
    "azure_ai": ("azure_ai_models",),
    "voyage": ("voyage_models",),
    "infinity": ("infinity_models",),
    "databricks": ("databricks_models",),
    "cloudflare": ("cloudflare_models",),
    "codestral": ("codestral_models",),
    "nlp_cloud": ("nlp_cloud_models",),
    "friendliai": ("friendliai_models",),
    "palm": ("palm_models",),
    "groq": ("groq_models",),
    "azure": ("azure_models", "azure_text_models"),
    "azure_anthropic": ("azure_anthropic_models",),
    "azure_text": ("azure_text_models",),
    "anyscale": ("anyscale_models",),
    "cerebras": ("cerebras_models",),
    "galadriel": ("galadriel_models",),
    "nvidia_nim": ("nvidia_nim_models",),
    "nvidia_riva": ("nvidia_riva_models",),
    "soniox": ("soniox_models",),
    "sambanova": ("sambanova_models", "sambanova_embedding_models"),
    "novita": ("novita_models",),
    "nebius": ("nebius_models", "nebius_embedding_models"),
    "aiml": ("aiml_models",),
    "assemblyai": ("assemblyai_models",),
    "jina_ai": ("jina_ai_models",),
    "snowflake": ("snowflake_models",),
    "gradient_ai": ("gradient_ai_models",),
    "meta_llama": ("llama_models",),
    "nscale": ("nscale_models",),
    "featherless_ai": ("featherless_ai_models",),
    "deepgram": ("deepgram_models",),
    "elevenlabs": ("elevenlabs_models",),
    "heroku": ("heroku_models",),
    "dashscope": ("dashscope_models",),
    "modelscope": ("modelscope_models",),
    "moonshot": ("moonshot_models",),
    "publicai": ("publicai_models",),
    "darkbloom": ("darkbloom_models",),
    "v0": ("v0_models",),
    "morph": ("morph_models",),
    "lambda_ai": ("lambda_ai_models",),
    "inception": ("inception_models",),
    "hyperbolic": ("hyperbolic_models",),
    "black_forest_labs": ("black_forest_labs_models",),
    "recraft": ("recraft_models",),
    "cometapi": ("cometapi_models",),
    "oci": ("oci_models",),
    "volcengine": ("volcengine_models",),
    "wandb": ("wandb_models",),
    "ovhcloud": ("ovhcloud_models", "ovhcloud_embedding_models"),
    "lemonade": ("lemonade_models",),
    "clarifai": ("clarifai_models",),
    "amazon_nova": ("amazon_nova_models",),
    "stability": ("stability_models",),
    "github_copilot": ("github_copilot_models",),
    "chatgpt": ("chatgpt_models",),
    "minimax": ("minimax_models",),
    "aws_polly": ("aws_polly_models",),
    "gigachat": ("gigachat_models",),
    "llamagate": ("llamagate_models",),
    "reducto": ("reducto_models",),
    "bedrock_mantle": ("bedrock_mantle_models",),
}

_MODEL_LIST_SOURCES: tuple[str, ...] = (
    "open_ai_chat_completion_models",
    "open_ai_text_completion_models",
    "cohere_models",
    "cohere_chat_models",
    "anthropic_models",
    "replicate_models",
    "openrouter_models",
    "datarobot_models",
    "huggingface_models",
    "vertex_chat_models",
    "vertex_text_models",
    "ai21_models",
    "ai21_chat_models",
    "together_ai_models",
    "baseten_models",
    "aleph_alpha_models",
    "nlp_cloud_models",
    "ollama_models",
    "bedrock_models",
    "deepinfra_models",
    "perplexity_models",
    "maritalk_models",
    "runwayml_models",
    "vertex_language_models",
    "watsonx_models",
    "gemini_models",
    "text_completion_codestral_models",
    "text_completion_inception_models",
    "xai_models",
    "zai_models",
    "fal_ai_models",
    "deepseek_models",
    "modelscope_models",
    "azure_ai_models",
    "voyage_models",
    "infinity_models",
    "databricks_models",
    "cloudflare_models",
    "codestral_models",
    "friendliai_models",
    "palm_models",
    "groq_models",
    "azure_models",
    "azure_anthropic_models",
    "anyscale_models",
    "cerebras_models",
    "galadriel_models",
    "nvidia_nim_models",
    "nvidia_riva_models",
    "soniox_models",
    "sambanova_models",
    "azure_text_models",
    "novita_models",
    "assemblyai_models",
    "jina_ai_models",
    "snowflake_models",
    "gradient_ai_models",
    "llama_models",
    "featherless_ai_models",
    "nscale_models",
    "deepgram_models",
    "elevenlabs_models",
    "dashscope_models",
    "moonshot_models",
    "publicai_models",
    "darkbloom_models",
    "v0_models",
    "morph_models",
    "lambda_ai_models",
    "inception_models",
    "black_forest_labs_models",
    "recraft_models",
    "cometapi_models",
    "oci_models",
    "heroku_models",
    "vercel_ai_gateway_models",
    "volcengine_models",
    "wandb_models",
    "ovhcloud_models",
    "lemonade_models",
    "docker_model_runner_models",
    "reducto_models",
    "bedrock_mantle_models",
    "clarifai_models",
)

_EMBEDDING_SOURCES: tuple[str, ...] = (
    "open_ai_embedding_models",
    "cohere_embedding_models",
    "bedrock_embedding_models",
    "vertex_embedding_models",
    "fireworks_ai_embedding_models",
    "nebius_embedding_models",
    "sambanova_embedding_models",
    "ovhcloud_embedding_models",
)

_STATIC_SEEDS: Mapping[str, frozenset[str]] = {
    "bedrock_converse_models": frozenset(BEDROCK_CONVERSE_MODELS),
    "wandb_models": frozenset(WANDB_MODELS),
    "empower_models": frozenset(empower_models),
    "modelscope_models": frozenset(modelscope_models),
}

_CONSTANT_SETS: Mapping[str, frozenset[str]] = {
    "replicate_models": frozenset(replicate_models),
    "clarifai_models": frozenset(clarifai_models),
    "huggingface_models": frozenset(huggingface_models),
    "together_ai_models": frozenset(together_ai_models),
    "baseten_models": frozenset(baseten_models),
    "open_ai_embedding_models": frozenset(open_ai_embedding_models),
    "cohere_embedding_models": frozenset(cohere_embedding_models),
    "bedrock_embedding_models": frozenset(bedrock_embedding_models),
}


def _rule_set_names(rule: _ProviderRule) -> tuple[str, ...]:
    match rule:
        case _LegacyTarget(set_name=set_name):
            return (set_name,)
        case _ModeSplitTarget(chat_set_name=chat_set_name, default_set_name=default_set_name):
            return (chat_set_name, default_set_name)
        case _:
            assert_never(rule)


REGISTRY_SET_NAMES: frozenset[str] = frozenset(
    name for rule in _PROVIDER_RULES.values() for name in _rule_set_names(rule)
)

_OPENROUTER_SET_NAME = _rule_set_names(_PROVIDER_RULES["openrouter"])[0]

_PROVIDERS_BY_SET: Mapping[str, tuple[str, ...]] = {
    set_name: tuple(provider for provider, sources in _PROVIDER_COMPOSITION.items() if set_name in sources)
    for set_name in {source for sources in _PROVIDER_COMPOSITION.values() for source in sources}
}


def _first(pair: tuple[str, str]) -> str:
    return pair[0]


def _prefix_roots(provider: str) -> tuple[str, ...]:
    return tuple(provider[:index] for index, char in enumerate(provider) if char == "-")


def _fallback_provider(provider: str, known_providers: frozenset[str]) -> str | None:
    if provider in known_providers:
        return provider
    return max((root for root in _prefix_roots(provider) if root in known_providers), key=len, default=None)


def _classify(
    key: str,
    value: Mapping[str, object],
    known_providers: frozenset[str],
) -> _DerivedMember | None:
    """Place one cost-map entry, or drop it.

    Providers with no table rule fall back to provider-bucket-only placement, and only for
    prefixed keys: unprefixed keys under such a provider are pricing tiers (``together-ai-up-to-4b``),
    not callable models.
    """
    provider = value.get("litellm_provider")
    if not isinstance(provider, str):
        return None
    rule = _PROVIDER_RULES.get(provider)
    if rule is None:
        if "/" not in key:
            return None
        bucket = _fallback_provider(provider, known_providers)
        return None if bucket is None else _ProviderMember(provider=bucket, model=key)
    match rule:
        case _LegacyTarget(set_name=set_name, strip_vertex_prefix=strip_vertex_prefix, key_filter=key_filter):
            if key_filter is not None and not key_filter(key):
                return None
            model = key.replace(_VERTEX_KEY_PREFIX, "") if strip_vertex_prefix else key
            return _LegacyMember(set_name=set_name, model=model)
        case _ModeSplitTarget(chat_set_name=chat_set_name, default_set_name=default_set_name):
            return _LegacyMember(
                set_name=chat_set_name if value.get("mode") == "chat" else default_set_name,
                model=key,
            )
        case _:
            assert_never(rule)


def _derive(
    model_cost: Mapping[str, Mapping[str, object]],
    known_providers: frozenset[str],
) -> tuple[_DerivedMember, ...]:
    return tuple(
        member for key, value in model_cost.items() if (member := _classify(key, value, known_providers)) is not None
    )


def _openrouter_register_aliases(
    model_cost_additions: Mapping[str, Mapping[str, object]],
) -> tuple[_DerivedMember, ...]:
    """Prefix-stripped twins for openrouter models registered at runtime.

    ``register_model`` has always stored the unprefixed name as well, so ``get_llm_provider`` can
    infer openrouter from a bare model name. The shipped cost map carries only verbatim keys, so
    this stays out of ``build_snapshot`` and cannot move import-time membership.
    """
    return tuple(
        _LegacyMember(set_name=_OPENROUTER_SET_NAME, model=key.split("/", 1)[-1])
        for key, value in model_cost_additions.items()
        if value.get("litellm_provider") == "openrouter" and "/" in key
    )


def _grouped(pairs: Iterable[tuple[str, str]]) -> Mapping[str, frozenset[str]]:
    return {name: frozenset(model for _, model in group) for name, group in groupby(sorted(pairs), key=_first)}


def _legacy_groups(members: tuple[_DerivedMember, ...]) -> Mapping[str, frozenset[str]]:
    return _grouped((member.set_name, member.model) for member in members if isinstance(member, _LegacyMember))


def _provider_groups(members: tuple[_DerivedMember, ...]) -> Mapping[str, frozenset[str]]:
    return _grouped((member.provider, member.model) for member in members if isinstance(member, _ProviderMember))


def _union(source_names: Iterable[str], resolved: Mapping[str, frozenset[str]]) -> frozenset[str]:
    return frozenset(model for name in source_names for model in resolved.get(name, frozenset()))


def _assemble(
    legacy_sets: Mapping[str, frozenset[str]],
    provider_extras: Mapping[str, frozenset[str]],
    static_model_names: Mapping[str, frozenset[str]],
) -> ModelRegistrySnapshot:
    resolved = {**_CONSTANT_SETS, **static_model_names, **legacy_sets}
    composed = {
        provider: _union(sources, resolved) | provider_extras.get(provider, frozenset())
        for provider, sources in _PROVIDER_COMPOSITION.items()
    }
    extra_only = {
        provider: models for provider, models in provider_extras.items() if provider not in _PROVIDER_COMPOSITION
    }
    model_list_set = _union(_MODEL_LIST_SOURCES, resolved)
    return ModelRegistrySnapshot(
        legacy_sets=legacy_sets,
        models_by_provider={**composed, **extra_only},
        model_list=tuple(sorted(model_list_set)),
        model_list_set=model_list_set,
        all_embedding_models=_union(_EMBEDDING_SOURCES, resolved),
    )


def build_snapshot(
    model_cost: Mapping[str, Mapping[str, object]],
    known_providers: frozenset[str],
    static_model_names: Mapping[str, frozenset[str]],
) -> ModelRegistrySnapshot:
    """Derive a complete snapshot from ``model_cost``, dropping models it no longer contains."""
    members = _derive(model_cost, known_providers)
    derived = _legacy_groups(members)
    legacy_sets = {
        name: _STATIC_SEEDS.get(name, frozenset()) | derived.get(name, frozenset()) for name in REGISTRY_SET_NAMES
    }
    return _assemble(legacy_sets, _provider_groups(members), static_model_names)


def extend_snapshot(
    snapshot: ModelRegistrySnapshot,
    model_cost_additions: Mapping[str, Mapping[str, object]],
    known_providers: frozenset[str],
) -> ModelRegistrySnapshot:
    """Add ``model_cost_additions`` to ``snapshot`` using the same rules as ``build_snapshot``.

    Additive counterpart of a full rebuild, for callers that register a handful of models at a
    time (``litellm.register_model``) and cannot pay for a full re-derivation per call.
    """
    members = _derive(model_cost_additions, known_providers) + _openrouter_register_aliases(model_cost_additions)
    if not members:
        return snapshot
    added = _legacy_groups(members)
    provider_extras = _provider_groups(members)
    legacy_sets = {
        **snapshot.legacy_sets,
        **{name: snapshot.legacy_sets.get(name, frozenset()) | models for name, models in added.items()},
    }
    touched_providers = frozenset(
        provider for name in added for provider in _PROVIDERS_BY_SET.get(name, ())
    ) | frozenset(provider_extras)
    models_by_provider = {
        **snapshot.models_by_provider,
        **{
            provider: snapshot.models_by_provider.get(provider, frozenset())
            | _union(_PROVIDER_COMPOSITION.get(provider, ()), added)
            | provider_extras.get(provider, frozenset())
            for provider in touched_providers
        },
    }
    listed = _union(_MODEL_LIST_SOURCES, added) - snapshot.model_list_set
    model_list_set = snapshot.model_list_set | listed
    return ModelRegistrySnapshot(
        legacy_sets=legacy_sets,
        models_by_provider=models_by_provider,
        model_list=tuple(sorted(model_list_set)) if listed else snapshot.model_list,
        model_list_set=model_list_set,
        all_embedding_models=snapshot.all_embedding_models | _union(_EMBEDDING_SOURCES, added),
    )

"""The provider model sets, derived from the live map on read (decision 18). The only place that knows provider names.
``LEGACY_NAMES`` names each of main's module-level sets and the raw ``litellm_provider`` strings it holds, ``FILTERS``
and ``BARE_KEYS`` carry the rules main's branch chain applied, and ``ALIASES`` is main's ``models_by_provider``"""

import datetime
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Final

from litellm._catalog import CATALOG

LEGACY_NAMES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "open_ai_chat_completion_models": ("openai",),
        "open_ai_text_completion_models": ("text-completion-openai",),
        "cohere_models": ("cohere",),
        "cohere_chat_models": ("cohere_chat",),
        "mistral_chat_models": ("mistral",),
        "text_completion_codestral_models": ("text-completion-codestral",),
        "text_completion_inception_models": ("text-completion-inception",),
        "anthropic_models": ("anthropic",),
        "openrouter_models": ("openrouter",),
        "datarobot_models": ("datarobot",),
        "vertex_language_models": ("vertex_ai-language-models",),
        "vertex_vision_models": ("vertex_ai-vision-models",),
        "vertex_chat_models": ("vertex_ai-chat-models",),
        "vertex_code_chat_models": ("vertex_ai-code-chat-models",),
        "vertex_ai_image_models": ("vertex_ai-image-models",),
        "vertex_ai_video_models": ("vertex_ai-video-models",),
        "vertex_text_models": ("vertex_ai-text-models",),
        "vertex_code_text_models": ("vertex_ai-code-text-models",),
        "vertex_embedding_models": ("vertex_ai-embedding-models",),
        "vertex_anthropic_models": ("vertex_ai-anthropic_models",),
        "vertex_llama3_models": ("vertex_ai-llama_models",),
        "vertex_deepseek_models": ("vertex_ai-deepseek_models",),
        "vertex_ai_ai21_models": ("vertex_ai-ai21_models",),
        "vertex_mistral_models": ("vertex_ai-mistral_models",),
        "vertex_openai_models": ("vertex_ai-openai_models",),
        "vertex_minimax_models": ("vertex_ai-minimax_models",),
        "vertex_moonshot_models": ("vertex_ai-moonshot_models",),
        "vertex_zai_models": ("vertex_ai-zai_models",),
        "ai21_models": ("ai21",),
        "ai21_chat_models": ("ai21",),
        "nlp_cloud_models": ("nlp_cloud",),
        "aleph_alpha_models": ("aleph_alpha",),
        "bedrock_models": ("bedrock",),
        "bedrock_converse_models": ("bedrock_converse",),
        "fal_ai_models": ("fal_ai",),
        "fireworks_ai_models": ("fireworks_ai",),
        "baseten_models": ("baseten",),
        "fireworks_ai_embedding_models": ("fireworks_ai-embedding-models",),
        "deepinfra_models": ("deepinfra",),
        "perplexity_models": ("perplexity",),
        "watsonx_models": ("watsonx",),
        "gemini_models": ("gemini",),
        "xai_models": ("xai",),
        "zai_models": ("zai",),
        "deepseek_models": ("deepseek",),
        "tencent_models": ("tencent",),
        "runwayml_models": ("runwayml",),
        "azure_ai_models": ("azure_ai",),
        "jina_ai_models": ("jina_ai",),
        "voyage_models": ("voyage",),
        "infinity_models": ("infinity",),
        "heroku_models": ("heroku",),
        "databricks_models": ("databricks",),
        "cloudflare_models": ("cloudflare",),
        "codestral_models": ("codestral",),
        "friendliai_models": ("friendliai",),
        "featherless_ai_models": ("featherless_ai",),
        "palm_models": ("palm",),
        "groq_models": ("groq",),
        "azure_models": ("azure",),
        "azure_anthropic_models": ("azure_anthropic",),
        "azure_text_models": ("azure_text",),
        "anyscale_models": ("anyscale",),
        "cerebras_models": ("cerebras",),
        "nadir_models": ("nadir",),
        "galadriel_models": ("galadriel",),
        "nvidia_nim_models": ("nvidia_nim",),
        "nvidia_riva_models": ("nvidia_riva",),
        "soniox_models": ("soniox",),
        "sambanova_models": ("sambanova",),
        "scaledown_models": ("scaledown",),
        "sambanova_embedding_models": ("sambanova-embedding-models",),
        "novita_models": ("novita",),
        "assemblyai_models": ("assemblyai",),
        "snowflake_models": ("snowflake",),
        "gradient_ai_models": ("gradient_ai",),
        "llama_models": ("meta_llama",),
        "nscale_models": ("nscale",),
        "nebius_models": ("nebius-chat-models",),
        "nebius_embedding_models": ("nebius-embedding-models",),
        "aiml_models": ("aiml",),
        "deepgram_models": ("deepgram",),
        "elevenlabs_models": ("elevenlabs",),
        "dashscope_models": ("dashscope",),
        "qwencloud_models": ("qwencloud",),
        "qwen_ai_platform_models": ("qwen_ai_platform",),
        "moonshot_models": ("moonshot",),
        "publicai_models": ("publicai",),
        "darkbloom_models": ("darkbloom",),
        "openinfer_models": ("openinfer",),
        "v0_models": ("v0",),
        "morph_models": ("morph",),
        "opencode_go_models": ("opencode_go",),
        "lambda_ai_models": ("lambda_ai",),
        "inception_models": ("inception",),
        "hyperbolic_models": ("hyperbolic",),
        "black_forest_labs_models": ("black_forest_labs",),
        "recraft_models": ("recraft",),
        "cometapi_models": ("cometapi",),
        "oci_models": ("oci",),
        "vercel_ai_gateway_models": ("vercel_ai_gateway",),
        "edenai_models": ("edenai",),
        "volcengine_models": ("volcengine",),
        "wandb_models": ("wandb",),
        "ovhcloud_models": ("ovhcloud",),
        "ovhcloud_embedding_models": ("ovhcloud-embedding-models",),
        "lemonade_models": ("lemonade",),
        "docker_model_runner_models": ("docker_model_runner",),
        "amazon_nova_models": ("amazon_nova",),
        "stability_models": ("stability",),
        "github_copilot_models": ("github_copilot",),
        "microsoft_365_copilot_models": ("microsoft_365_copilot",),
        "chatgpt_models": ("chatgpt",),
        "minimax_models": ("minimax",),
        "aws_polly_models": ("aws_polly",),
        "transcribe_models": ("transcribe",),
        "gigachat_models": ("gigachat",),
        "llamagate_models": ("llamagate",),
        "coralbricks_models": ("coralbricks",),
        "reducto_models": ("reducto",),
        "bedrock_mantle_models": ("bedrock_mantle",),
    }
)

_BEDROCK_PRICING_ONLY: Final = re.compile(r"^bedrock/[a-zA-Z0-9_-]+/.+$")


def is_bedrock_pricing_only_model(key: str) -> bool:
    """Keys of the pattern ``bedrock/<region>/<model>`` and commitment keys, priced for lookup only"""
    return "month-commitment" in key or _BEDROCK_PRICING_ONLY.match(key) is not None


def is_openai_finetune_model(key: str) -> bool:
    """``ft:<model>`` keys, priced for lookup only"""
    return key.startswith("ft:") and not key.count(":") > 1


def _is_past_deprecation_date(row: Mapping[str, object]) -> bool:
    deprecation_date: Final = row.get("deprecation_date")
    if not isinstance(deprecation_date, str):
        return False
    try:
        return datetime.date.fromisoformat(deprecation_date) <= datetime.datetime.now(datetime.timezone.utc).astimezone().date()
    except ValueError:
        return False


FILTERS: Final[Mapping[str, Callable[[str, Mapping[str, object]], bool]]] = MappingProxyType(
    {
        "open_ai_chat_completion_models": lambda key, row: not is_openai_finetune_model(key),
        "ai21_chat_models": lambda key, row: row.get("mode") == "chat",
        "ai21_models": lambda key, row: row.get("mode") != "chat",
        "bedrock_models": lambda key, row: row.get("mode") != "guardrail" and not is_bedrock_pricing_only_model(key),
        "fireworks_ai_models": lambda key, row: "-to-" not in key and "fireworks-ai-default" not in key,
        "fireworks_ai_embedding_models": lambda key, row: "-to-" not in key,
        "baseten_models": lambda key, row: not _is_past_deprecation_date(row),
    }
)

BARE_KEYS: Final = frozenset(
    {
        "vertex_ai_ai21_models",
        "vertex_ai_image_models",
        "vertex_ai_video_models",
        "vertex_anthropic_models",
        "vertex_deepseek_models",
        "vertex_llama3_models",
        "vertex_minimax_models",
        "vertex_mistral_models",
        "vertex_moonshot_models",
        "vertex_openai_models",
        "vertex_zai_models",
    }
)
"""Sets main filled with the ``vertex_ai/`` prefix stripped from each key"""

STATIC: Final = frozenset(
    {
        "clarifai_models",
        "huggingface_models",
        "maritalk_models",
        "ollama_models",
        "petals_models",
        "replicate_models",
        "together_ai_models",
    }
)
"""Hand lists of models the map has no rows for, read from ``litellm`` as they are"""

ALIASES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
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
        "edenai": ("edenai_models",),
        "datarobot": ("datarobot_models",),
        "vertex_ai": (
            "vertex_chat_models",
            "vertex_code_chat_models",
            "vertex_text_models",
            "vertex_code_text_models",
            "vertex_anthropic_models",
            "vertex_vision_models",
            "vertex_language_models",
            "vertex_deepseek_models",
            "vertex_minimax_models",
            "vertex_moonshot_models",
            "vertex_zai_models",
        ),
        "ai21": ("ai21_models", "ai21_chat_models"),
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
        "nadir": ("nadir_models",),
        "galadriel": ("galadriel_models",),
        "nvidia_nim": ("nvidia_nim_models",),
        "nvidia_riva": ("nvidia_riva_models",),
        "soniox": ("soniox_models",),
        "sambanova": ("sambanova_models", "sambanova_embedding_models"),
        "scaledown": ("scaledown_models",),
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
        "qwencloud": ("qwencloud_models",),
        "qwen_ai_platform": ("qwen_ai_platform_models",),
        "modelscope": ("modelscope_models",),
        "moonshot": ("moonshot_models",),
        "publicai": ("publicai_models",),
        "darkbloom": ("darkbloom_models",),
        "openinfer": ("openinfer_models",),
        "v0": ("v0_models",),
        "morph": ("morph_models",),
        "opencode_go": ("opencode_go_models",),
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
        "microsoft_365_copilot": ("microsoft_365_copilot_models",),
        "chatgpt": ("chatgpt_models",),
        "minimax": ("minimax_models",),
        "aws_polly": ("aws_polly_models",),
        "transcribe": ("transcribe_models",),
        "gigachat": ("gigachat_models",),
        "llamagate": ("llamagate_models",),
        "coralbricks": ("coralbricks_models",),
        "reducto": ("reducto_models",),
        "bedrock_mantle": ("bedrock_mantle_models",),
    }
)
"""main's ``models_by_provider``, each provider the union of the named sets"""

_SEEDS: Final[Mapping[str, str]] = MappingProxyType(
    {"bedrock_converse_models": "BEDROCK_CONVERSE_MODELS", "wandb_models": "WANDB_MODELS"}
)


OPAQUE_KEYS: Final[set[str]] = set()  # mutable-ok: register_model adds router deployment ids, never removed
"""Keys registered as opaque ids (``add_to_provider_model_sets=False``), which no provider set lists"""


def _litellm() -> object:
    return sys.modules["litellm"]


def _rows() -> Mapping[str, Mapping[str, object]]:
    import litellm

    return litellm.model_cost


def _derived(name: str) -> frozenset[str]:
    rows: Final = _rows()
    keep: Final = FILTERS.get(name, lambda key, row: True)
    bare: Final = name in BARE_KEYS
    keys: Final = (
        key
        for provider in LEGACY_NAMES[name]
        for key in CATALOG.models_for(provider)  # comprehension-ok: a set's providers, then each provider's keys
        if keep(key, rows.get(key) or {}) and key not in OPAQUE_KEYS
    )
    seed: Final = _SEEDS.get(name)
    seeded: Final = frozenset(getattr(_litellm(), seed)) if seed is not None else frozenset()
    return frozenset(key.replace("vertex_ai/", "") if bare else key for key in keys) | seeded


def legacy_set(name: str) -> frozenset[str]:
    """One of main's 121 module-level provider sets, as the live map has it now. Read only"""
    return _derived(name)


def contains(name: str, model: str) -> bool:
    """``model in litellm.<name>`` without building the set, for the request path. The same rows, filters, bare keys
    and seeds as ``legacy_set``. A set assigned to ``litellm.<name>`` answers instead"""
    assigned: Final = vars(_litellm()).get(name)
    if assigned is not None:
        return model in assigned
    rows: Final = _rows()
    keep: Final = FILTERS.get(name, lambda key, row: True)
    bare: Final = name in BARE_KEYS
    if bare and "vertex_ai/" in model:
        return False
    keys: Final = (model, f"vertex_ai/{model}") if bare else (model,)
    in_map: Final = any(
        isinstance(row := rows.get(key), Mapping)
        and row.get("litellm_provider") in LEGACY_NAMES[name]
        and key not in OPAQUE_KEYS
        and keep(key, row)
        for key in keys
    )
    seed: Final = _SEEDS.get(name)
    return in_map or (seed is not None and model in getattr(_litellm(), seed))


def models_for(provider: str) -> frozenset[str]:
    """The models of ``provider`` as ``models_by_provider`` lists them"""
    names: Final = ALIASES.get(provider)
    if names is None:
        return CATALOG.models_for(provider) - OPAQUE_KEYS
    return frozenset().union(
        *(legacy_set(name) if name in LEGACY_NAMES else frozenset(getattr(_litellm(), name)) for name in names)
    )


class ModelsByProvider(Mapping[str, frozenset[str]]):
    """``litellm.models_by_provider``, read only. Main's providers, then every raw provider string in the map"""

    def __getitem__(self, provider: str) -> frozenset[str]:
        if provider in ALIASES or provider in CATALOG.providers():
            return models_for(provider)
        raise KeyError(provider)

    def __iter__(self) -> Iterator[str]:
        return iter((*ALIASES, *(provider for provider in CATALOG.providers() if provider not in ALIASES)))

    def __len__(self) -> int:
        return len(ALIASES) + sum(1 for provider in CATALOG.providers() if provider not in ALIASES)

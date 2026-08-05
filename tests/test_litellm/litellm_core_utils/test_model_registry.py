from collections.abc import Iterable

import pytest

from litellm import constants
from litellm.litellm_core_utils.model_registry import (
    _PROVIDER_COMPOSITION,
    ModelRegistrySnapshot,
    build_snapshot,
)
from litellm.types.utils import LlmProviders

KNOWN_PROVIDERS = frozenset(provider.value for provider in LlmProviders)


def snapshot(
    model_cost: dict[str, dict[str, object]],
    static_model_names: dict[str, frozenset[str]] | None = None,
) -> ModelRegistrySnapshot:
    return build_snapshot(
        model_cost=model_cost,
        known_providers=KNOWN_PROVIDERS,
        static_model_names=static_model_names or {},
    )


def entry(provider: str, mode: str = "chat") -> dict[str, object]:
    return {"litellm_provider": provider, "mode": mode}


@pytest.mark.parametrize(
    "provider, key, expected_set, expected_member",
    [
        ("vertex_ai-anthropic_models", "vertex_ai/claude-sonnet-4-5", "vertex_anthropic_models", "claude-sonnet-4-5"),
        ("vertex_ai-llama_models", "vertex_ai/llama-3.1-8b", "vertex_llama3_models", "llama-3.1-8b"),
        ("vertex_ai-deepseek_models", "vertex_ai/deepseek-v3", "vertex_deepseek_models", "deepseek-v3"),
        ("vertex_ai-mistral_models", "vertex_ai/mistral-large", "vertex_mistral_models", "mistral-large"),
        ("vertex_ai-ai21_models", "vertex_ai/jamba-large", "vertex_ai_ai21_models", "jamba-large"),
        ("vertex_ai-image-models", "vertex_ai/imagen-4", "vertex_ai_image_models", "imagen-4"),
        ("vertex_ai-video-models", "vertex_ai/veo-3", "vertex_ai_video_models", "veo-3"),
        ("vertex_ai-openai_models", "vertex_ai/gpt-oss-120b", "vertex_openai_models", "gpt-oss-120b"),
        ("vertex_ai-minimax_models", "vertex_ai/minimax-m2", "vertex_minimax_models", "minimax-m2"),
        ("vertex_ai-moonshot_models", "vertex_ai/kimi-k2", "vertex_moonshot_models", "kimi-k2"),
        ("vertex_ai-zai_models", "vertex_ai/glm-4.6", "vertex_zai_models", "glm-4.6"),
    ],
)
def test_vertex_families_strip_the_vertex_ai_key_prefix(
    provider: str, key: str, expected_set: str, expected_member: str
) -> None:
    snap = snapshot({key: entry(provider)})

    assert snap.legacy_sets[expected_set] == frozenset({expected_member})


@pytest.mark.parametrize(
    "provider, expected_set",
    [
        ("vertex_ai-language-models", "vertex_language_models"),
        ("vertex_ai-chat-models", "vertex_chat_models"),
        ("vertex_ai-text-models", "vertex_text_models"),
        ("vertex_ai-embedding-models", "vertex_embedding_models"),
    ],
)
def test_vertex_families_without_a_strip_rule_keep_the_whole_key(provider: str, expected_set: str) -> None:
    snap = snapshot({"vertex_ai/gemini-3-pro": entry(provider)})

    assert snap.legacy_sets[expected_set] == frozenset({"vertex_ai/gemini-3-pro"})


def test_openai_finetune_keys_are_priced_but_not_enumerable() -> None:
    snap = snapshot(
        {
            "gpt-5.2": entry("openai"),
            "ft:gpt-5.2": entry("openai"),
            "ft:gpt-5.2:acme::abc123": entry("openai"),
        }
    )

    assert snap.legacy_sets["open_ai_chat_completion_models"] == frozenset({"gpt-5.2", "ft:gpt-5.2:acme::abc123"})


def test_bedrock_pricing_only_keys_are_not_enumerable() -> None:
    snap = snapshot(
        {
            "anthropic.claude-sonnet-4-5-v1:0": entry("bedrock"),
            "bedrock/us-east-1/anthropic.claude-sonnet-4-5-v1:0": entry("bedrock"),
            "anthropic.claude-sonnet-4-5-v1:0-month-commitment": entry("bedrock"),
        }
    )

    assert snap.legacy_sets["bedrock_models"] == frozenset({"anthropic.claude-sonnet-4-5-v1:0"})


def test_fireworks_pricing_tier_keys_are_not_enumerable() -> None:
    snap = snapshot(
        {
            "accounts/fireworks/models/kimi-k2": entry("fireworks_ai"),
            "fireworks-ai-up-to-16b": entry("fireworks_ai"),
            "fireworks-ai-default": entry("fireworks_ai"),
            "nomic-ai/nomic-embed-text-v1.5": entry("fireworks_ai-embedding-models"),
            "fireworks-ai-up-to-150m": entry("fireworks_ai-embedding-models"),
        }
    )

    assert snap.legacy_sets["fireworks_ai_models"] == frozenset({"accounts/fireworks/models/kimi-k2"})
    assert snap.legacy_sets["fireworks_ai_embedding_models"] == frozenset({"nomic-ai/nomic-embed-text-v1.5"})


def test_ai21_splits_on_chat_mode() -> None:
    snap = snapshot(
        {
            "jamba-large-1.7": entry("ai21", mode="chat"),
            "j2-ultra": entry("ai21", mode="completion"),
        }
    )

    assert snap.legacy_sets["ai21_chat_models"] == frozenset({"jamba-large-1.7"})
    assert snap.legacy_sets["ai21_models"] == frozenset({"j2-ultra"})


def test_meta_llama_provider_feeds_the_llama_models_set() -> None:
    snap = snapshot({"Llama-4-Maverick-17B": entry("meta_llama")})

    assert snap.legacy_sets["llama_models"] == frozenset({"Llama-4-Maverick-17B"})
    assert snap.models_by_provider["meta_llama"] == frozenset({"Llama-4-Maverick-17B"})


def test_fallback_buckets_prefixed_keys_whose_provider_is_a_known_provider() -> None:
    snap = snapshot({"vertex_ai/orphaned-model": entry("vertex_ai")})

    assert "vertex_ai/orphaned-model" in snap.models_by_provider["vertex_ai"]


def test_fallback_routes_a_suffixed_provider_to_its_known_root() -> None:
    snap = snapshot({"vertex_ai/qwen3-coder": entry("vertex_ai-qwen_models")})

    assert "vertex_ai/qwen3-coder" in snap.models_by_provider["vertex_ai"]


def test_fallback_drops_providers_that_are_not_known_providers() -> None:
    snap = snapshot({"tavily/search": entry("tavily"), "serper/search": entry("serper")})

    assert "tavily" not in snap.models_by_provider
    assert "serper" not in snap.models_by_provider


def test_fallback_drops_slashless_keys_because_they_are_pricing_tiers() -> None:
    snap = snapshot({"together-ai-up-to-4b": entry("together_ai"), "nebius-up-to-8b": entry("nebius")})

    assert "together-ai-up-to-4b" not in snap.models_by_provider["together_ai"]
    assert "nebius-up-to-8b" not in snap.models_by_provider["nebius"]


def test_fallback_entries_never_reach_model_list() -> None:
    snap = snapshot(
        {
            "vertex_ai/orphaned-model": entry("vertex_ai"),
            "nebius/Qwen/Qwen3-4B": entry("nebius"),
            "claude-sonnet-4-5": entry("anthropic"),
        }
    )

    assert "vertex_ai/orphaned-model" in snap.models_by_provider["vertex_ai"]
    assert "nebius/Qwen/Qwen3-4B" in snap.models_by_provider["nebius"]
    assert "vertex_ai/orphaned-model" not in snap.model_list_set
    assert "nebius/Qwen/Qwen3-4B" not in snap.model_list_set
    assert "vertex_ai/orphaned-model" not in snap.model_list
    assert "nebius/Qwen/Qwen3-4B" not in snap.model_list
    assert "claude-sonnet-4-5" in snap.model_list_set


def test_rebuild_prunes_models_dropped_from_the_cost_map() -> None:
    before = snapshot({"claude-keep": entry("anthropic"), "claude-drop": entry("anthropic")})
    after = snapshot({"claude-keep": entry("anthropic")})

    assert {"claude-keep", "claude-drop"} <= before.legacy_sets["anthropic_models"]
    assert "claude-drop" in before.model_list_set
    assert after.legacy_sets["anthropic_models"] == frozenset({"claude-keep"})
    assert "claude-drop" not in after.models_by_provider["anthropic"]
    assert "claude-drop" not in after.model_list_set
    assert "claude-drop" not in after.model_list


def test_rebuild_prunes_fallback_bucketed_models_too() -> None:
    before = snapshot({"vertex_ai/keep": entry("vertex_ai"), "vertex_ai/drop": entry("vertex_ai")})
    after = snapshot({"vertex_ai/keep": entry("vertex_ai")})

    assert "vertex_ai/drop" in before.models_by_provider["vertex_ai"]
    assert "vertex_ai/drop" not in after.models_by_provider["vertex_ai"]


@pytest.mark.parametrize(
    "set_name, seed",
    [
        ("bedrock_converse_models", constants.BEDROCK_CONVERSE_MODELS),
        ("wandb_models", constants.WANDB_MODELS),
        ("empower_models", constants.empower_models),
        ("modelscope_models", constants.modelscope_models),
    ],
)
def test_static_seeds_survive_a_build_from_an_empty_cost_map(set_name: str, seed: Iterable[str]) -> None:
    snap = snapshot({})

    assert snap.legacy_sets[set_name] == frozenset(seed)


def test_an_empty_cost_map_leaves_unseeded_sets_empty() -> None:
    snap = snapshot({})

    assert snap.legacy_sets["anthropic_models"] == frozenset()
    assert snap.legacy_sets["groq_models"] == frozenset()


def test_building_does_not_mutate_the_constants_sets() -> None:
    tracked = (
        "empower_models",
        "modelscope_models",
        "WANDB_MODELS",
        "BEDROCK_CONVERSE_MODELS",
        "replicate_models",
        "clarifai_models",
        "huggingface_models",
        "together_ai_models",
        "baseten_models",
        "open_ai_embedding_models",
        "cohere_embedding_models",
        "bedrock_embedding_models",
    )
    before = {name: frozenset(getattr(constants, name)) for name in tracked}

    snapshot(
        {
            "empower/empower-brand-new": entry("empower"),
            "Qwen/Qwen4-Brand-New": entry("modelscope"),
            "openai/gpt-oss-brand-new": entry("wandb"),
            "anthropic.claude-brand-new-v1:0": entry("bedrock_converse"),
        }
    )

    assert {name: frozenset(getattr(constants, name)) for name in tracked} == before


def test_static_model_names_are_injected_into_their_provider_buckets() -> None:
    snap = snapshot({}, static_model_names={"petals_models": frozenset({"petals-team/StableBeluga2"})})

    assert snap.models_by_provider["petals"] == frozenset({"petals-team/StableBeluga2"})


def test_models_by_provider_unions_every_source_set_for_a_provider() -> None:
    snap = snapshot(
        {
            "gpt-5.2": entry("openai"),
            "gpt-3.5-turbo-instruct": entry("text-completion-openai"),
            "anthropic.claude-sonnet-4-5-v1:0": entry("bedrock"),
            "anthropic.claude-converse-v1:0": entry("bedrock_converse"),
        }
    )

    assert {"gpt-5.2", "gpt-3.5-turbo-instruct"} <= snap.models_by_provider["openai"]
    assert snap.models_by_provider["text-completion-openai"] == frozenset({"gpt-3.5-turbo-instruct"})
    assert {"anthropic.claude-sonnet-4-5-v1:0", "anthropic.claude-converse-v1:0"} <= snap.models_by_provider["bedrock"]


def test_all_embedding_models_unions_every_embedding_source() -> None:
    snap = snapshot(
        {
            "vertex_ai/text-embedding-005": entry("vertex_ai-embedding-models", mode="embedding"),
            "nomic-ai/nomic-embed-text-v1.5": entry("fireworks_ai-embedding-models", mode="embedding"),
            "BAAI/bge-en-icl": entry("nebius-embedding-models", mode="embedding"),
            "E5-Mistral-7B-Instruct": entry("sambanova-embedding-models", mode="embedding"),
            "bge-multilingual-gemma2": entry("ovhcloud-embedding-models", mode="embedding"),
        }
    )

    assert {
        "vertex_ai/text-embedding-005",
        "nomic-ai/nomic-embed-text-v1.5",
        "BAAI/bge-en-icl",
        "E5-Mistral-7B-Instruct",
        "bge-multilingual-gemma2",
    } <= snap.all_embedding_models
    assert frozenset(constants.open_ai_embedding_models) <= snap.all_embedding_models


def test_model_list_is_sorted_and_matches_model_list_set() -> None:
    snap = snapshot({"zzz-model": entry("anthropic"), "aaa-model": entry("anthropic")})

    assert snap.model_list == tuple(sorted(snap.model_list))
    assert frozenset(snap.model_list) == snap.model_list_set


def test_snapshot_collections_are_immutable() -> None:
    snap = snapshot({"claude-sonnet-4-5": entry("anthropic")})

    assert isinstance(snap.legacy_sets["anthropic_models"], frozenset)
    assert isinstance(snap.models_by_provider["anthropic"], frozenset)
    assert isinstance(snap.model_list_set, frozenset)
    assert isinstance(snap.all_embedding_models, frozenset)
    assert isinstance(snap.model_list, tuple)


def test_fallback_only_provider_buckets_are_frozensets() -> None:
    """These buckets skip the composition union, so nothing else re-freezes them."""
    assert "sagemaker" not in _PROVIDER_COMPOSITION

    snap = snapshot({"sagemaker/my-endpoint": entry("sagemaker")})

    assert snap.models_by_provider["sagemaker"] == frozenset({"sagemaker/my-endpoint"})
    assert isinstance(snap.models_by_provider["sagemaker"], frozenset)
    assert all(isinstance(models, frozenset) for models in snap.models_by_provider.values())


@pytest.mark.parametrize("mutator", ["add", "discard", "clear", "update"])
def test_legacy_sets_reject_in_place_mutation(mutator: str) -> None:
    """The pre-refactor sets were mutable; silent in-place edits are what went stale."""
    snap = snapshot({"claude-sonnet-4-5": entry("anthropic")})

    with pytest.raises(AttributeError):
        getattr(snap.legacy_sets["anthropic_models"], mutator)


def test_served_names_are_frozensets_and_model_list_is_a_fresh_list() -> None:
    """litellm.model_list stayed a list for compat, but callers must not be able to poison it."""
    import litellm

    assert isinstance(litellm.anthropic_models, frozenset)
    assert isinstance(litellm.models_by_provider["anthropic"], frozenset)
    assert isinstance(litellm.model_list_set, frozenset)

    borrowed = litellm.model_list
    assert isinstance(borrowed, list)
    borrowed.append("not-a-real-model")
    assert "not-a-real-model" not in litellm.model_list
    assert "not-a-real-model" not in litellm.model_list_set


def test_entries_without_a_string_provider_are_dropped() -> None:
    broken = {"no-provider": {"mode": "chat"}, "null-provider": {"litellm_provider": None, "mode": "chat"}}
    baseline = snapshot({})
    snap = snapshot({**broken, "claude-sonnet-4-5": entry("anthropic")})

    assert snap.model_list_set - baseline.model_list_set == frozenset({"claude-sonnet-4-5"})
    assert not any(models & broken.keys() for models in snap.legacy_sets.values())
    assert not any(models & broken.keys() for models in snap.models_by_provider.values())

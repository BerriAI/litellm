"""The ``litellm.model_cost`` questions Rust asks while registering models and pricing calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from litellm.litellm_core_utils import fallback_generalizations, model_cost_table
from litellm.litellm_core_utils.core_helpers import as_object_list
from litellm.litellm_core_utils.model_cost_table import AliasedCostEntry, Landing


def entry(fields: Mapping[str, object], names: tuple[str, ...]) -> AliasedCostEntry:
    """The row object every name in ``names`` shares, the owner first."""
    return AliasedCostEntry(fields, names)


published: Final = model_cost_table.published
write: Final = model_cost_table.write


def generation() -> tuple[
    Mapping[str, object],
    tuple[int, int, int, int],
    tuple[Mapping[str, object], Mapping[str, object], Sequence[object]],
]:
    """The published table, what a pricing snapshot of it is keyed on (its ``ModelCostVersion`` and the version of the
    installed ``fallback_generalizations`` rules), then the cost discount and margin configs as they read now, which
    the snapshot compares by value, and the installed rules."""
    import litellm

    table: Final = model_cost_table.published()
    version: Final = model_cost_table.model_cost_version()
    installed: Final = as_object_list(fallback_generalizations.get_fallback_generalization_rules())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # the rule registry keeps the raw untyped list
    rules: Final = installed if installed is not None else ()
    return (
        table,
        (version.table, version.size, version.writes, fallback_generalizations.fallback_generalizations_version()),
        (litellm.cost_discount_config, litellm.cost_margin_config, rules),
    )


def inputs() -> object:
    """The ``NativeModelCostMap`` behind ``litellm.model_cost``, whose snapshot a Rust route prices its call over."""
    from litellm.litellm_core_utils.get_model_cost_map import model_cost_inputs

    return model_cost_inputs


def known_names() -> tuple[frozenset[str], Mapping[str, str], Mapping[str, str], frozenset[str]]:
    """Every provider name, the model names ``get_model_info`` renames before it looks one up (Azure deployment names
    for every provider, then the Vertex AI Llama, Mistral and AI21 names for Vertex AI) and the Replicate models
    ``completion_cost`` bills by run time."""
    import litellm
    from litellm.types.utils import LlmProvidersSet

    latest: Final = {
        name.removesuffix("@latest"): name
        for name in (*litellm.vertex_mistral_models, *litellm.vertex_ai_ai21_models)
        if name.endswith("@latest")
    }
    llama: Final = {
        name.removeprefix("meta/"): name for name in litellm.vertex_llama3_models if name.startswith("meta/")
    }
    return (
        frozenset(LlmProvidersSet),
        {**litellm.azure_llms, **litellm.azure_embedding_models},
        {**latest, **llama},
        frozenset(litellm.replicate_models),
    )


def infer_provider(model: str) -> tuple[str, str] | None:
    """The model and provider ``get_llm_provider`` reads off ``model``, ``None`` where it raises."""
    import litellm

    try:
        inferred_model, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:  # noqa: BLE001  # get_llm_provider raises a bare Exception for a name it cannot place
        return None
    return inferred_model, provider


def served_row(model: str, provider: str) -> Mapping[str, object] | None:
    """The row ``get_model_info`` answers for a provider that serves its own (Ollama and Lemonade read it live, a
    Hugging Face row is built), through the cache Python's own lookups share, ``None`` where the lookup raises."""
    from litellm.utils import cached_get_model_info_helper

    try:
        return cached_get_model_info_helper(model=model, custom_llm_provider=provider)
    except Exception:  # noqa: BLE001  # get_model_info raises a bare Exception for an unmapped model
        return None


def resolve(
    rows: Mapping[str, Mapping[str, object]] | None,
    key: str,
    registration: Mapping[str, object],
    custom_llm_provider: str | None,
) -> Landing | None:
    """Where a ``register_model`` entry lands in ``rows`` and the row it starts from, ``None`` when ``rows`` is no
    longer the published table. ``rows`` of ``None`` is the published table."""
    import litellm
    from litellm.utils import registration_base

    if rows is not None and rows is not litellm.model_cost:
        return None
    return registration_base(key, registration, custom_llm_provider)


def merged(
    current: Mapping[str, object] | None, base: Mapping[str, object], registration: Mapping[str, object]
) -> AliasedCostEntry:
    """The row a registration writes over ``current``, the row now under its landing key."""
    from litellm.utils import model_cost_registration_write

    return model_cost_registration_write(current, base, registration)


def kept(previous: dict[str, object] | None, registration: Mapping[str, object]) -> dict[str, object]:
    """The registration ``register_model(persist_across_reloads=True)`` keeps for a key: ``registration`` merged
    into the one kept before it."""
    from litellm.utils import base_fields_replaced_by, update_dictionary

    replaced: Final = base_fields_replaced_by(registration)
    return {
        name: value
        for name, value in update_dictionary(dict(previous or {}), registration).items()
        if name not in replaced
    }

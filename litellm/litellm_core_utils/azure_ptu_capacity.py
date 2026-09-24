"""How much throughput one Azure OpenAI provisioned throughput unit (PTU) serves per model.

Azure sizes a provisioned deployment in normalized tokens per minute:
``input TPM x (1 - cache hit rate) + output-to-input ratio x output TPM``, divided by the
model's "Input TPM per PTU" to get the PTUs required. The same two numbers turn a team's
PTU share into a per-minute token ceiling and a request's usage back into PTU-hours.

Table read from
https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-sizing#deployment-parameters-and-throughput-values-by-model
on 2026-09-24 (page dated 2026-09-23). Azure deducts cached input tokens in full for every
model except the GPT-6 family, where a cached input token costs a tenth of an uncached one.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol


class NormalizedTokenWeights(Protocol):
    @property
    def output_to_input_ratio(self) -> float: ...

    @property
    def cached_input_ratio(self) -> float: ...


@dataclass(frozen=True, slots=True)
class PTUCapacity:
    input_tpm_per_ptu: int
    output_to_input_ratio: float
    cached_input_ratio: float = 0.0

    @property
    def normalized_tokens_per_ptu_hour(self) -> int:
        return self.input_tpm_per_ptu * 60


AZURE_PTU_CAPACITY: Final[Mapping[str, PTUCapacity]] = MappingProxyType(
    {
        "gpt-6-sol": PTUCapacity(3_000, 5.0, cached_input_ratio=0.1),
        "gpt-6-astra": PTUCapacity(600, 5.0, cached_input_ratio=0.1),
        "gpt-5.6-luna": PTUCapacity(30_000, 6.0),
        "gpt-5.6-terra": PTUCapacity(3_000, 6.0),
        "gpt-5.6-sol": PTUCapacity(1_200, 6.0),
        "gpt-5.5": PTUCapacity(1_200, 6.0),
        "gpt-5.4": PTUCapacity(2_400, 6.0),
        "gpt-5.4-mini": PTUCapacity(7_900, 6.0),
        "gpt-5.3-codex": PTUCapacity(3_400, 8.0),
        "gpt-5.2": PTUCapacity(3_400, 8.0),
        "gpt-5.2-codex": PTUCapacity(3_400, 8.0),
        "gpt-5.1": PTUCapacity(4_750, 8.0),
        "gpt-5.1-codex": PTUCapacity(4_750, 8.0),
        "gpt-5": PTUCapacity(4_750, 8.0),
        "gpt-5-mini": PTUCapacity(23_750, 8.0),
        "gpt-4.1": PTUCapacity(3_000, 4.0),
        "gpt-4.1-mini": PTUCapacity(14_900, 4.0),
        "gpt-4.1-nano": PTUCapacity(59_400, 4.0),
        "o3": PTUCapacity(3_000, 4.0),
        "o4-mini": PTUCapacity(5_400, 4.0),
        "gpt-4o": PTUCapacity(2_500, 4.0),
        "gpt-4o-mini": PTUCapacity(37_000, 4.0),
        "o3-mini": PTUCapacity(2_500, 4.0),
        "o1": PTUCapacity(230, 4.0),
        "llama-3.3-70b-instruct": PTUCapacity(8_450, 4.0),
    }
)

_VERSION_SUFFIX: Final = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def azure_ptu_capacity(model: str) -> PTUCapacity | None:
    """The sizing row for ``model``, read as its last path segment with a dated version dropped.

    ``azure/gpt-4.1-2025-04-14`` and ``gpt-4.1`` both resolve to the ``gpt-4.1`` row; a
    deployment name that is not a model name resolves to nothing, which is why callers
    prefer ``model_info.base_model``.
    """
    name: Final = model.rsplit("/", 1)[-1].strip().lower()
    return AZURE_PTU_CAPACITY.get(name) or AZURE_PTU_CAPACITY.get(_VERSION_SUFFIX.sub("", name))


def deployment_ptu_capacity(deployment: Mapping[str, object]) -> PTUCapacity | None:
    """The sizing row a deployment resolves to: ``model_info.base_model`` first, since an
    Azure deployment name is arbitrary, then ``litellm_params.model``."""
    model_info: Final = deployment.get("model_info")
    litellm_params: Final = deployment.get("litellm_params")
    candidates: Final = tuple(
        value
        for value in (
            model_info.get("base_model") if isinstance(model_info, Mapping) else None,
            litellm_params.get("model") if isinstance(litellm_params, Mapping) else None,
        )
        if isinstance(value, str) and value
    )
    return next((capacity for capacity in map(azure_ptu_capacity, candidates) if capacity is not None), None)


def normalized_tokens(
    weights: NormalizedTokenWeights, *, prompt_tokens: int, completion_tokens: int, cache_read_tokens: int = 0
) -> float:
    """Azure's normalized token count for one request: uncached input in full, cached input
    at the model's cached ratio, output weighted by the output-to-input ratio."""
    cached: Final = min(max(cache_read_tokens, 0), max(prompt_tokens, 0))
    uncached: Final = max(prompt_tokens, 0) - cached
    return uncached + weights.cached_input_ratio * cached + weights.output_to_input_ratio * max(completion_tokens, 0)


def ptu_hours(capacity: PTUCapacity, normalized: float) -> float:
    """PTU-hours ``normalized`` tokens amount to: one PTU serves its input TPM for sixty minutes."""
    return normalized / capacity.normalized_tokens_per_ptu_hour

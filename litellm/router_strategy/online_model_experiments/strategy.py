from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from litellm.router_strategy.online_model_experiments.assignment import (
    ExperimentAssignment,
    ExperimentAssignmentError,
    ExperimentVariant,
    assign_variant,
    derive_assignment_key,
)
from litellm.types.router import PreRoutingHookResponse

_DEFAULT_IDENTITY_KEYS: Final[tuple[str, ...]] = (
    "user_api_key_hash",
    "user",
    "end_user",
    "litellm_session_id",
)


@dataclass(frozen=True, slots=True)
class OnlineModelExperimentConfig:
    experiment_id: str
    secret: str
    variants: tuple[ExperimentVariant, ...]
    identity_metadata_key: str | None = None

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, object]) -> OnlineModelExperimentConfig:
        experiment_id: Final = _required_string(raw_config, "experiment_id")
        secret: Final = _required_string(raw_config, "secret")
        raw_variants: Final = raw_config.get("variants")
        if not isinstance(raw_variants, (list, tuple)):
            raise ValueError("online_model_experiment_config.variants must be a list")
        variants: Final = tuple(_parse_variant(raw_variant) for raw_variant in raw_variants)
        identity_metadata_key: Final = raw_config.get("identity_metadata_key")
        if identity_metadata_key is not None and not isinstance(identity_metadata_key, str):
            raise ValueError("online_model_experiment_config.identity_metadata_key must be a string")
        return cls(
            experiment_id=experiment_id,
            secret=secret,
            variants=variants,
            identity_metadata_key=identity_metadata_key,
        )


class OnlineModelExperimentRouter:
    def __init__(self, config: OnlineModelExperimentConfig) -> None:
        self.config = config

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict[str, object],
        messages: list[dict[str, object]] | None = None,
        input: str | list[object] | None = None,
        specific_deployment: bool | None = False,
    ) -> PreRoutingHookResponse:
        del model, input, specific_deployment
        identity: Final = _resolve_identity(
            request_kwargs=request_kwargs,
            configured_key=self.config.identity_metadata_key,
        )
        assignment_key: Final = derive_assignment_key(
            experiment_id=self.config.experiment_id,
            identity=identity,
            secret=self.config.secret,
        )
        _raise_on_assignment_error(assignment_key)
        assignment: Final = assign_variant(
            experiment_id=self.config.experiment_id,
            assignment_key=assignment_key,
            variants=self.config.variants,
        )
        _raise_on_assignment_error(assignment)
        selected_variant: Final = next(
            variant for variant in self.config.variants if variant.name == assignment.variant_name
        )
        _stamp_experiment_metadata(
            request_kwargs=request_kwargs,
            experiment_id=self.config.experiment_id,
            variant_name=selected_variant.name,
            assignment_key=assignment_key,
            bucket=assignment.bucket,
        )
        return PreRoutingHookResponse(
            model=selected_variant.model_name,
            messages=messages,
        )


def _required_string(raw_config: Mapping[str, object], key: str) -> str:
    value: Final = raw_config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"online_model_experiment_config.{key} must be a non-empty string")
    return value


def _parse_variant(raw_variant: object) -> ExperimentVariant:
    if not isinstance(raw_variant, Mapping):
        raise ValueError("online_model_experiment_config.variants entries must be mappings")
    name: Final = raw_variant.get("name")
    model_name: Final = raw_variant.get("model")
    weight_basis_points: Final = raw_variant.get("weight_basis_points")
    if not isinstance(name, str) or not name:
        raise ValueError("online_model_experiment_config variant name must be a non-empty string")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("online_model_experiment_config variant model must be a non-empty string")
    if isinstance(weight_basis_points, bool) or not isinstance(weight_basis_points, int):
        raise ValueError("online_model_experiment_config variant weight_basis_points must be an integer")
    return ExperimentVariant(name=name, model_name=model_name, weight_basis_points=weight_basis_points)


def _resolve_identity(request_kwargs: Mapping[str, object], configured_key: str | None) -> str:
    metadata_values: Final = tuple(
        value
        for metadata_name in ("litellm_metadata", "metadata")
        for metadata in (request_kwargs.get(metadata_name),)
        if isinstance(metadata, Mapping)
        for key in ((configured_key,) if configured_key is not None else _DEFAULT_IDENTITY_KEYS)
        for value in (metadata.get(key),)
        if isinstance(value, str) and value
    )
    if metadata_values:
        return metadata_values[0]
    raise ValueError("online_model_experiment requires a stable identity in request metadata")


def _raise_on_assignment_error(result: object) -> None:
    if isinstance(result, ExperimentAssignmentError):
        raise ValueError(result.message)
    if not isinstance(result, (str, ExperimentAssignment)):
        raise ValueError("online_model_experiment assignment failed")


def _stamp_experiment_metadata(
    request_kwargs: dict[str, object],
    experiment_id: str,
    variant_name: str,
    assignment_key: str,
    bucket: int,
) -> None:
    experiment_metadata: Final = {
        "online_model_experiment_id": experiment_id,
        "online_model_experiment_variant": variant_name,
        "online_model_experiment_assignment_key": assignment_key,
        "online_model_experiment_bucket": bucket,
    }
    metadata: Final = request_kwargs.get("metadata")
    metadata_values: Final = dict(metadata) if isinstance(metadata, Mapping) else {}
    request_kwargs["metadata"] = {**metadata_values, **experiment_metadata}
    litellm_params: Final = request_kwargs.get("litellm_params")
    litellm_params_values: Final = dict(litellm_params) if isinstance(litellm_params, Mapping) else {}
    litellm_metadata: Final = litellm_params_values.get("metadata")
    litellm_metadata_values: Final = dict(litellm_metadata) if isinstance(litellm_metadata, Mapping) else {}
    request_kwargs["litellm_params"] = {
        **litellm_params_values,
        "metadata": {**litellm_metadata_values, **experiment_metadata},
    }

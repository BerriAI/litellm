from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class DiagnosticPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    minimum_level: Literal["TRACE", "DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    target_prefixes: tuple[str, ...] = ()
    sample_rate: float = Field(default=1.0, ge=0, le=1, strict=True)


class _OtlpDestination(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    transport: Literal["otlp"] = "otlp"
    name: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    headers: Mapping[str, str] = Field(default_factory=dict, repr=False)
    policy: DiagnosticPolicy | None = None


class _PostHogDestination(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    transport: Literal["posthog"] = "posthog"
    name: str = Field(min_length=1)
    endpoint: str = "https://us.i.posthog.com"
    api_key: str = Field(min_length=1, repr=False)
    policy: DiagnosticPolicy | None = None


_Destination: TypeAlias = Annotated[_OtlpDestination | _PostHogDestination, Field(discriminator="transport")]


class DiagnosticsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(default=False, strict=True)
    service_name: str = "litellm"
    policy: DiagnosticPolicy = Field(default_factory=DiagnosticPolicy)
    destinations: tuple[_Destination, ...] = ()

    @model_validator(mode="after")
    def unique_destination_names(self) -> Self:
        names: Final = tuple(destination.name for destination in self.destinations)
        if len(frozenset(names)) != len(names) or any(not name.strip() for name in names):
            raise ValueError("diagnostic destination names must be nonempty and unique")
        return self

    @classmethod
    def from_sources(
        cls,
        settings: Mapping[str, object] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> DiagnosticsConfig:
        value: Final = (os.environ if environment is None else environment).get("LITELLM_DIAGNOSTICS")
        return (
            cls.model_validate_json(value)
            if value is not None
            else cls.model_validate({} if settings is None else settings)
        )


def configure(
    configuration: DiagnosticsConfig | Mapping[str, object],
    *,
    loggers: tuple[logging.Logger, ...] | None = None,
) -> bool:
    from litellm.rust_bridge import forwarding

    config: Final = DiagnosticsConfig.model_validate(configuration)
    return forwarding.configure(config.model_dump_json(), loggers)


def force_flush() -> bool:
    from litellm.rust_bridge import forwarding

    return forwarding.force_flush()


def shutdown() -> bool:
    from litellm.rust_bridge import forwarding

    return forwarding.shutdown()

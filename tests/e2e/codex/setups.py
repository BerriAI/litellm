"""The credentials are `os.environ/` references the gateway resolves, the same names the Claude Code matrix's
`test_config.yaml` uses, and Bedrock rides the gateway's default AWS identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import pytest
from _pytest.mark.structures import ParameterSet
from cli_harness.rate_limiter import RateBucket
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from models import LiteLLMParamsBody

GPT_ON_BEDROCK_MANTLE: Final = "bedrock_mantle/openai.gpt-6.1-sol"
GPT_ON_BEDROCK_RUNTIME: Final = "bedrock/us.openai.gpt-6.1-sol"
GPT_ON_AZURE_OPENAI: Final = "azure/gpt-6.1-sol"
GPT_ON_OPENAI: Final = "openai/gpt-6.1-sol"
CLAUDE_ON_BEDROCK: Final = "bedrock/us.anthropic.claude-sonnet-5-5"
_BEDROCK_REGION: Final = "us-east-1"

type SetupId = Literal["bedrock_mantle", "bedrock_runtime", "azure_openai", "openai", "claude_bedrock"]


@dataclass(frozen=True, slots=True)
class Deployment:
    litellm_params: LiteLLMParamsBody
    rate_bucket: RateBucket


@dataclass(frozen=True, slots=True)
class ModelSetup:
    model: str
    rate_bucket: RateBucket


def _drives(provider: Provider, model: str) -> pytest.MarkDecorator:
    return meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(provider,),
            models=(model,),
            mode=Mode.STREAM,
        )
    )


SETUPS: Final = (
    pytest.param(
        Deployment(
            LiteLLMParamsBody(model=GPT_ON_BEDROCK_MANTLE, aws_region_name=_BEDROCK_REGION), RateBucket.BEDROCK_MANTLE
        ),
        id="bedrock_mantle",
        marks=_drives(Provider.BEDROCK_MANTLE, GPT_ON_BEDROCK_MANTLE),
    ),
    pytest.param(
        Deployment(
            LiteLLMParamsBody(model=GPT_ON_BEDROCK_RUNTIME, aws_region_name=_BEDROCK_REGION),
            RateBucket.BEDROCK_CONVERSE,
        ),
        id="bedrock_runtime",
        marks=_drives(Provider.BEDROCK, GPT_ON_BEDROCK_RUNTIME),
    ),
    pytest.param(
        Deployment(
            LiteLLMParamsBody(
                model=GPT_ON_AZURE_OPENAI, api_base="os.environ/AZURE_API_BASE", api_key="os.environ/AZURE_API_KEY"
            ),
            RateBucket.AZURE_OPENAI,
        ),
        id="azure_openai",
        marks=_drives(Provider.AZURE, GPT_ON_AZURE_OPENAI),
    ),
    pytest.param(
        Deployment(LiteLLMParamsBody(model=GPT_ON_OPENAI, api_key="os.environ/OPENAI_API_KEY"), RateBucket.OPENAI),
        id="openai",
        marks=_drives(Provider.OPENAI, GPT_ON_OPENAI),
    ),
    pytest.param(
        Deployment(
            LiteLLMParamsBody(model=CLAUDE_ON_BEDROCK, aws_region_name=_BEDROCK_REGION), RateBucket.BEDROCK_CONVERSE
        ),
        id="claude_bedrock",
        marks=_drives(Provider.BEDROCK, CLAUDE_ON_BEDROCK),
    ),
)


def with_expected_failure(setup_id: SetupId, *, reason: str) -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(*setup.values, id=setup.id, marks=(*setup.marks, pytest.mark.xfail(strict=True, reason=reason)))
        if setup.id == setup_id
        else setup
        for setup in SETUPS
    )

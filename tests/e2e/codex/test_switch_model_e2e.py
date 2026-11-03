"""Azure has a single GPT deployment on the test stack, so the Azure setup switches to another GPT on OpenAI."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import (
    BEDROCK_REGION,
    CLAUDE_ON_BEDROCK,
    GPT_ON_AZURE_OPENAI,
    GPT_ON_BEDROCK_MANTLE,
    GPT_ON_BEDROCK_RUNTIME,
    GPT_ON_OPENAI,
    Deployment,
    ExtraModel,
    ModelSetup,
    also_driving,
    with_expected_failure,
)
from e2e_config import unique_marker
from e2e_metadata import Provider
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

TERRA_ON_OPENAI: Final = ExtraModel(
    Provider.OPENAI, LiteLLMParamsBody(model="openai/gpt-6.1-terra", api_key="os.environ/OPENAI_API_KEY")
)
SWITCH_TO: Final = MappingProxyType(
    {
        GPT_ON_BEDROCK_MANTLE: ExtraModel(
            Provider.BEDROCK_MANTLE,
            LiteLLMParamsBody(model="bedrock_mantle/openai.gpt-6-luna", aws_region_name=BEDROCK_REGION),
        ),
        GPT_ON_BEDROCK_RUNTIME: ExtraModel(
            Provider.BEDROCK, LiteLLMParamsBody(model="bedrock/us.openai.gpt-6-luna", aws_region_name=BEDROCK_REGION)
        ),
        GPT_ON_AZURE_OPENAI: TERRA_ON_OPENAI,
        GPT_ON_OPENAI: TERRA_ON_OPENAI,
        CLAUDE_ON_BEDROCK: ExtraModel(
            Provider.BEDROCK,
            LiteLLMParamsBody(model="bedrock/us.anthropic.claude-haiku-5-5", aws_region_name=BEDROCK_REGION),
        ),
    }
)


@pytest.fixture
def second_model(deployment: Deployment, register_model: Callable[[LiteLLMParamsBody], str]) -> str:
    return register_model(SWITCH_TO[deployment.litellm_params.model].litellm_params)


@pytest.fixture
def key_models(setup: ModelSetup, second_model: str) -> tuple[str, ...]:
    return (setup.model, second_model)


@pytest.mark.parametrize(
    "deployment",
    also_driving(
        with_expected_failure(
            "azure_openai",
            reason="LiteLLM forwards the reasoning Azure encrypted to OpenAI, which cannot decrypt another account's"
            " reasoning and rejects the turn with invalid_encrypted_content (#41792)",
        ),
        SWITCH_TO,
    ),
)
class TestCodexSwitchModel:
    def test_remembers_a_value_after_resuming_on_another_model(
        self, codex: CodexHarness, setup: ModelSetup, second_model: str
    ) -> None:
        marker: Final = unique_marker()
        workspace: Final = codex.new_workspace(files={"secret.txt": marker})
        first: Final = codex.exec(
            "Run `cat secret.txt`. Remember the value it prints, then work out how many of its characters are digits"
            " and reply with that count.",
            model=setup.model,
            cwd=workspace,
            reasoning_effort="high",
        )
        assert first.thread_id is not None, first
        assert any(marker in command.output for command in first.commands), first
        assert first.usage is not None and first.usage.reasoning_output_tokens > 0, first
        (workspace / "secret.txt").unlink()
        second: Final = codex.resume(
            first.thread_id,
            "What value did secret.txt print earlier? Reply with just that value.",
            model=second_model,
            cwd=workspace,
            reasoning_effort="high",
        )
        assert second.exit_code == 0, second
        assert codex.turn_models(first.thread_id) == (setup.model, second_model), second
        assert marker in second.answer, second

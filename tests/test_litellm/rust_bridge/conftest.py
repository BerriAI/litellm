from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.route_parity.fixture_recorder import fixture_id, parametrize_recorded_fixtures
from tests.test_litellm.rust_bridge.chat_completions_fixture_models import ChatCompletionParityCase

FIXTURE_DIR_ENV: Final = "LITELLM_CHAT_COMPLETIONS_FIXTURE_DIR"


def _fixture_id(fixture: ChatCompletionParityCase) -> str:
    case_input: Final = fixture.litellm_input
    mode: Final = "stream" if case_input.stream else "nonstream"
    return fixture_id(case_input, f"{case_input.model}-{mode}")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    default_directory: Final = Path(__file__).with_name("chat_completions_fixtures")
    parametrize_recorded_fixtures(
        metafunc,
        fixture_name="chat_completion_fixture",
        case_type=ChatCompletionParityCase,
        env_var=FIXTURE_DIR_ENV,
        default_directory=default_directory,
        regeneration_command=(
            "uv run python tests/test_litellm/rust_bridge/generate_chat_completions_fixtures.py "
            f"--fixture-dir {default_directory} --examples 2"
        ),
        id_builder=_fixture_id,
    )

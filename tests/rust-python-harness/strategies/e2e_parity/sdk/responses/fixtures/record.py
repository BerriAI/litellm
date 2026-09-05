from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dotenv import load_dotenv
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

import litellm
from litellm.rust_bridge.configuration import use_litellm_rust

from ......shared.parity.fixtures.cli import parse_recording_args
from ......shared.parity.fixtures.pipeline import RecordingTarget, record_fixtures
from ......shared.parity.fixtures.recording import UpstreamEndpoint
from ......shared.parity.fixtures.store import fixture_directory, fixture_id
from ......shared.parity.recorded_http import RecordedResponse
from ..test_sdk_parity import CASES, ResponsesCase

FIXTURE_DIR_ENV: Final = "LITELLM_RESPONSES_FIXTURE_DIR"
DEFAULT_FIXTURE_DIRECTORY: Final = Path(__file__).with_name("data")


def input_strategy(model: str) -> SearchStrategy[ResponsesCase]:
    sdk_input: Final = st.one_of(
        st.text(min_size=1, max_size=80),
        st.text(min_size=1, max_size=80).map(lambda text: [{"role": "user", "content": text}]),
    )
    params: Final = st.fixed_dictionaries(
        {"max_output_tokens": st.integers(min_value=1, max_value=64)},
        optional={
            "instructions": st.text(min_size=1, max_size=40),
            "temperature": st.just(1.0),
            "top_p": st.just(1.0),
        },
    )
    return st.builds(
        ResponsesCase,
        name=st.just("generated"),
        model=st.just(model),
        sdk_input=sdk_input,
        params=params,
        provider_responses=st.just(()),
        expected=st.just("success"),
    )


@dataclass(frozen=True, slots=True)
class ResponsesInvocation:
    api_key: str

    def execute(self, provider_url: str, case_input: ResponsesCase) -> None:
        call: Final = cast(Callable[..., object], litellm.responses)
        call(
            model=case_input.model,
            input=case_input.sdk_input,
            api_base=provider_url,
            api_key=self.api_key,
            use_chat_completions_api=True,
            num_retries=0,
            **case_input.params,
        )


def _case_factory(case: ResponsesCase, responses: tuple[RecordedResponse, ...]) -> ResponsesCase:
    return case.model_copy(update={"name": fixture_id(case, case.model), "provider_responses": responses})


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = parse_recording_args()
    api_key: Final = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY to record Responses parity fixtures")
    api_base: Final = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com")
    target: Final = RecordingTarget(
        name="anthropic-responses-via-chat",
        upstream=UpstreamEndpoint(base_url=api_base),
        strategy=input_strategy("anthropic/claude-sonnet-5"),
        invocation=ResponsesInvocation(api_key),
        required_inputs=tuple(case for case in CASES if case.expected == "success"),
    )
    root: Final = fixture_directory(
        args.fixture_dir,
        os.environ.get(FIXTURE_DIR_ENV),
        DEFAULT_FIXTURE_DIRECTORY,
    )
    use_litellm_rust(False)
    summary: Final = record_fixtures((target,), root, args.examples, args.concurrency, ResponsesCase, _case_factory)
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

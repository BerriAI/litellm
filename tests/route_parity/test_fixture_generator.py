from __future__ import annotations

import queue
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from hypothesis import strategies as st

from tests.route_parity.fixture_generator import FixtureSdkCall, FixtureTarget, discover_fixture_targets
from tests.route_parity.fixture_models import SdkInputBase
from tests.route_parity.fixture_recorder import ProviderSpec


class ExampleSdkInput(SdkInputBase):
    model: str


@dataclass(frozen=True, slots=True)
class ExampleProvider:
    name: str
    key_name: str

    def targets(
        self,
        environ: Mapping[str, str],
        sdk_call: FixtureSdkCall,
    ) -> tuple[FixtureTarget[ExampleSdkInput], ...]:
        api_key: Final = environ.get(self.key_name)
        if not api_key:
            return ()

        def invoke(api_base: str, case_input: ExampleSdkInput) -> object:
            return sdk_call(api_base=api_base, api_key=api_key, **case_input.as_sdk_kwargs())

        case_input: Final = ExampleSdkInput(model=f"{self.name}/model")
        return (
            FixtureTarget(
                name=self.name,
                provider_spec=ProviderSpec(upstream_base=f"https://{self.name}.example"),
                strategy=st.just(case_input),
                invoke=invoke,
                required_inputs=(case_input,),
            ),
        )


def test_discover_fixture_targets_flattens_configured_providers_and_injects_sdk_call() -> None:
    calls: Final[queue.SimpleQueue[dict[str, object]]] = queue.SimpleQueue()

    def sdk_call(**kwargs: object) -> object:
        calls.put(kwargs)
        return "response"

    providers: Final = (
        ExampleProvider(name="first", key_name="FIRST_KEY"),
        ExampleProvider(name="skipped", key_name="SKIPPED_KEY"),
        ExampleProvider(name="second", key_name="SECOND_KEY"),
    )
    targets: Final = discover_fixture_targets(
        providers,
        {"FIRST_KEY": "first-secret", "SECOND_KEY": "second-secret"},
        sdk_call,
    )

    assert tuple(target.name for target in targets) == ("first", "second")
    assert targets[1].invoke("http://127.0.0.1:1234", targets[1].required_inputs[0]) == "response"
    assert calls.get_nowait() == {
        "api_base": "http://127.0.0.1:1234",
        "api_key": "second-secret",
        "model": "second/model",
    }

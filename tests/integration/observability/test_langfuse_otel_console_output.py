import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from _langfuse_otel_metadata_support import exported_llm_span, langfuse_rig, scenario_identity, watched_spans
from integration._support.client import Gateway, eventually

CONSOLE_SPAN: Final = '"span_id": "0x'
INHERITED_OTEL_ENV: Final = tuple(name for name in os.environ if name.startswith("OTEL_"))


@pytest.mark.parametrize(
    ("overrides", "printed"),
    [({}, False), ({"OTEL_EXPORTER": "console"}, True)],
    ids=["exporter-unset", "operator-asked-for-console"],
)
def test_langfuse_otel_prints_spans_to_stdout_only_when_the_operator_asks(
    gateway: Gateway, tmp_path: Path, overrides: Mapping[str, str], printed: bool
) -> None:
    marker: Final = "lf-console-" + uuid.uuid4().hex
    with (
        langfuse_rig(
            gateway,
            tmp_path,
            {},
            marker,
            overrides={"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_only", **overrides},
            remove_environment=INHERITED_OTEL_ENV,
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider, team=False)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": identity["model"], "messages": [{"role": "user", "content": marker}]},
            key=key,
        )
        assert response.status_code == 200, response.text

        exported_llm_span(rig.collector, response.json()["id"])

        on_stdout: Final = CONSOLE_SPAN in rig.proxy.log.read_text()
        assert on_stdout is printed, f"span JSON on proxy stdout: {on_stdout}, see {rig.proxy.log}"


@pytest.mark.parametrize("callbacks", [[], ["langfuse_otel"]], ids=["key-only", "callback-listed"])
def test_key_langfuse_without_operator_keys_prints_no_spans(
    gateway: Gateway, tmp_path: Path, callbacks: list[str]
) -> None:
    marker: Final = "lf-console-key-" + uuid.uuid4().hex
    with (
        langfuse_rig(
            gateway,
            tmp_path,
            {"callbacks": callbacks},
            marker,
            overrides={"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_only"},
            remove_environment=INHERITED_OTEL_ENV,
            operator_keys=False,
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(
            key_alias="lf-console-key-" + uuid.uuid4().hex,
            metadata={
                "logging": [
                    {
                        "callback_name": "langfuse_otel",
                        "callback_type": "success",
                        "callback_vars": {
                            "langfuse_public_key": "pk-lf-key",
                            "langfuse_secret_key": "sk-lf-key",
                            "langfuse_host": rig.collector.url,
                        },
                    }
                ]
            },
        )
        model: Final = scenario.model(api_base=rig.provider.url + "/v1")
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}]},
            key=key,
        )
        assert response.status_code == 200, response.text

        eventually(
            watched_spans(rig.collector), lambda spans: any(marker in json.dumps(span) for span in spans), seconds=25
        )

        on_stdout: Final = CONSOLE_SPAN in rig.proxy.log.read_text()
        assert not on_stdout, f"span JSON on proxy stdout, see {rig.proxy.log}"

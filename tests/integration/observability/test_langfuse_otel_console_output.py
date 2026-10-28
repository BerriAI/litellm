import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from _langfuse_otel_metadata_support import exported_llm_span, langfuse_rig, scenario_identity
from integration._support.client import Gateway

CONSOLE_SPAN: Final = '"span_id": "0x'


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
            remove_environment=tuple(name for name in os.environ if name.startswith("OTEL_")),
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

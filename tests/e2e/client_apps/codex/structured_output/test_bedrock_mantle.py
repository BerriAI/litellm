from __future__ import annotations

from typing import Final

import pytest
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta

from client_apps.codex.columns import column_models
from client_apps.codex.features import CodexCell, assert_cell, structured_output
from client_apps.compat_collector import CompatRecorder

MODELS: Final = column_models("bedrock_mantle")


@pytest.mark.e2e
@pytest.mark.covers("llm.responses.bedrock_mantle.structured_output.stream.codex_structured_output")
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.RESPONSES,
        providers=(Provider.BEDROCK_MANTLE,),
        models=MODELS,
        capabilities=(Capability.RESPONSE_SCHEMA,),
        mode=Mode.STREAM,
    )
)
def test_structured_output_bedrock_mantle(compat_result: CompatRecorder, codex_cell: CodexCell) -> None:
    assert_cell(compat_result, codex_cell, MODELS, structured_output)

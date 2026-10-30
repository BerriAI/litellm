from __future__ import annotations

from typing import Final

import pytest
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta

from client_apps.codex.columns import column_models
from client_apps.codex.features import CodexCell, reasoning_effort, assert_cell
from client_apps.compat_collector import CompatRecorder

MODELS: Final = column_models("vertex_ai")


@pytest.mark.e2e
@pytest.mark.covers("llm.responses.vertex.thinking.stream.codex_reasoning_effort")
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.RESPONSES,
        providers=(Provider.VERTEX_AI,),
        models=MODELS,
        capabilities=(Capability.REASONING,),
        mode=Mode.STREAM,
    )
)
def test_reasoning_effort_vertex_ai(compat_result: CompatRecorder, codex_cell: CodexCell) -> None:
    assert_cell(compat_result, codex_cell, MODELS, reasoning_effort)

from __future__ import annotations

from typing import Final

import pytest
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta

from client_apps.codex.columns import column_models
from client_apps.codex.features import CodexCell, multi_turn, assert_cell
from client_apps.compat_collector import CompatRecorder

MODELS: Final = column_models("azure")


@pytest.mark.e2e
@pytest.mark.covers("llm.responses.azure_openai.multi_turn.stream.codex_multi_turn")
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.RESPONSES,
        providers=(Provider.AZURE,),
        models=MODELS,
        mode=Mode.STREAM,
    )
)
def test_multi_turn_azure(compat_result: CompatRecorder, codex_cell: CodexCell) -> None:
    assert_cell(compat_result, codex_cell, MODELS, multi_turn)

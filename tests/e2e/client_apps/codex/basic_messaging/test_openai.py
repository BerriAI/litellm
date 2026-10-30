from __future__ import annotations

from typing import Final

import pytest
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta

from client_apps.codex.columns import column_models
from client_apps.codex.features import CodexCell, basic_messaging, assert_cell
from client_apps.compat_collector import CompatRecorder

MODELS: Final = column_models("openai")


@pytest.mark.e2e
@pytest.mark.covers("llm.responses.openai.basic.stream.codex_basic_messaging")
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.RESPONSES,
        providers=(Provider.OPENAI,),
        models=MODELS,
        mode=Mode.STREAM,
    )
)
def test_basic_messaging_openai(compat_result: CompatRecorder, codex_cell: CodexCell) -> None:
    assert_cell(compat_result, codex_cell, MODELS, basic_messaging)

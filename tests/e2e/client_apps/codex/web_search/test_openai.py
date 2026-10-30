from __future__ import annotations

from typing import Final

import pytest
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta

from client_apps.codex.columns import column_models
from client_apps.codex.features import CodexCell, assert_cell, web_search
from client_apps.compat_collector import CompatRecorder

MODELS: Final = column_models("openai")


@pytest.mark.e2e
@pytest.mark.covers("llm.responses.openai.web_search.stream.codex_web_search")
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.RESPONSES,
        providers=(Provider.OPENAI,),
        models=MODELS,
        capabilities=(Capability.WEB_SEARCH,),
        mode=Mode.STREAM,
    )
)
def test_web_search_openai(compat_result: CompatRecorder, codex_cell: CodexCell) -> None:
    assert_cell(compat_result, codex_cell, MODELS, web_search)

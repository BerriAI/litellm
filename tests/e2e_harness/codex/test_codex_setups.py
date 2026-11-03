from __future__ import annotations

from types import MappingProxyType
from typing import Final

from _pytest.mark.structures import ParameterSet
from codex.setups import SETUPS, Deployment, also_driving, with_expected_failure
from e2e_metadata import Provider, Subject

SECOND_MODEL: Final = "openai/second-model"


def _subject(setup: ParameterSet) -> Subject:
    (meta,) = (mark for mark in setup.marks if mark.name == "meta")
    return _only_subject(meta.args)


def _only_subject(args: tuple[object, ...]) -> Subject:
    (subject,) = args
    assert isinstance(subject, Subject)
    return subject


def _model(setup: ParameterSet) -> str:
    deployment: Final = setup.values[0]
    assert isinstance(deployment, Deployment)
    return deployment.litellm_params.model


def test_also_driving_declares_the_extra_model_on_every_setup_and_keeps_the_xfail() -> None:
    setups: Final = also_driving(
        with_expected_failure("azure_openai", reason="known bug"),
        MappingProxyType(
            {_model(setup): Subject(providers=(Provider.OPENAI,), models=(SECOND_MODEL,)) for setup in SETUPS}
        ),
    )

    assert tuple(setup.id for setup in setups) == tuple(setup.id for setup in SETUPS)
    assert all(_subject(setup).models == tuple(sorted((_model(setup), SECOND_MODEL))) for setup in setups)
    azure: Final = next(setup for setup in setups if setup.id == "azure_openai")
    assert _subject(azure).providers == (Provider.AZURE, Provider.OPENAI)
    assert any(mark.name == "xfail" for mark in azure.marks)

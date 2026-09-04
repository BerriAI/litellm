from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Final, TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from .store import recorded_fixtures

CaseT = TypeVar("CaseT", bound=BaseModel)


def parametrize_recorded_fixtures(
    metafunc: pytest.Metafunc,
    *,
    fixture_name: str,
    case_type: type[CaseT],
    env_var: str,
    default_directory: Path,
    regeneration_command: str,
    id_builder: Callable[[CaseT], str],
    marks_builder: Callable[[CaseT], tuple[pytest.MarkDecorator, ...]] | None = None,
) -> None:
    if fixture_name not in metafunc.fixturenames:
        return
    configured: Final = os.environ.get(env_var)
    if configured == "":
        raise pytest.UsageError(f"{env_var} is set but empty")
    directory: Final = Path(configured).expanduser() if configured is not None else default_directory
    try:
        fixtures: Final = recorded_fixtures(directory, case_type)
    except (ValidationError, ValueError) as error:
        raise pytest.UsageError(
            f"Invalid parity fixture bundle at {directory}. "
            "Each fixture must use the current versioned envelope. "
            f"Record fresh fixtures in an empty directory with: `{regeneration_command}`. "
            f"Validation details: {error}"
        ) from error
    if fixtures:
        metafunc.parametrize(
            fixture_name,
            tuple(
                pytest.param(
                    fixture,
                    id=id_builder(fixture),
                    marks=marks_builder(fixture) if marks_builder is not None else (),
                )
                for fixture in fixtures
            ),
        )
        return
    if configured is not None:
        raise pytest.UsageError(f"no recorded fixtures in {directory}")
    metafunc.parametrize(
        fixture_name,
        (
            pytest.param(
                None,
                marks=pytest.mark.skip(reason=f"no recorded fixtures in {directory}"),
                id="no-recorded-fixtures",
            ),
        ),
    )

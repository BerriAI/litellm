from typing import Final
from unittest.mock import patch

import pytest

from litellm.litellm_core_utils.optional_imports import ensure_optional_import


@pytest.mark.parametrize("module", ["boto3", "botocore", "tokenizers"])
def test_missing_optional_dependency_names_the_installable_package(module: str) -> None:
    with patch.dict("sys.modules", {module: None}):
        with pytest.raises(ModuleNotFoundError) as caught:
            ensure_optional_import(module)
    package: Final = "boto3" if module == "botocore" else module
    assert str(caught.value) == f"Missing optional dependency '{module}'. Run 'pip install {package}'."
    assert caught.value.name == module
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize(
    "failure", [ModuleNotFoundError(name="unrelated_dependency"), ImportError("broken installation")]
)
def test_optional_import_preserves_unrelated_failure(failure: ImportError) -> None:
    with patch("builtins.__import__", side_effect=failure):
        with pytest.raises(ImportError) as caught:
            ensure_optional_import("botocore")
    assert caught.value is failure


def test_available_optional_dependency_does_not_raise() -> None:
    assert ensure_optional_import("json") is None

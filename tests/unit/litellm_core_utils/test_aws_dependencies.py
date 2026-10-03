from typing import Final
from unittest.mock import patch

import pytest

from litellm.litellm_core_utils.aws_dependencies import require_aws_sdk


@pytest.mark.parametrize("missing", ("boto3", "botocore"))
def test_missing_aws_sdk_names_install_extra(missing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ImportError, match=r"litellm\[aws\]") as caught:
        require_aws_sdk()
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert caught.value.__cause__.name == missing


def test_installed_aws_sdk_is_accepted() -> None:
    assert require_aws_sdk() is None


def test_aws_sdk_preserves_unrelated_import_failure() -> None:
    failure: Final = ModuleNotFoundError("broken dependency", name="unrelated_dependency")
    with patch("litellm.litellm_core_utils.aws_dependencies.import_module", side_effect=failure):
        with pytest.raises(ModuleNotFoundError) as caught:
            require_aws_sdk()
    assert caught.value is failure

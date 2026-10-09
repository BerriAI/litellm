import pytest

from litellm.proxy.data_manager.__main__ import main


def test_main_rejects_unexpected_arguments() -> None:
    with pytest.raises(SystemExit, match="usage: python -m litellm.proxy.data_manager"):
        main(["unexpected"])

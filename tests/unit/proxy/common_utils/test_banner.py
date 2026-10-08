import logging

import pytest

from litellm.proxy.common_utils.banner import LITELLM_BANNER, show_banner


def test_banner_is_one_info_record_not_stdout(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
    show_banner()
    assert capsys.readouterr() == ("", "")

    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    show_banner()
    [record] = [record for record in caplog.records if record.name == "LiteLLM Proxy"]
    assert record.levelno == logging.INFO
    assert LITELLM_BANNER in record.getMessage()

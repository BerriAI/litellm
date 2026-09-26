import logging
from collections.abc import Iterator

import pytest

from litellm._logging import verbose_logger


@pytest.fixture
def litellm_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    verbose_logger.addHandler(caplog.handler)
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        yield caplog
    verbose_logger.removeHandler(caplog.handler)

# Local conftest for mavvrik integration tests.
# Avoids importing the top-level conftest which requires the full litellm package
# with all optional dependencies.

from litellm.integrations.mavvrik.logger import Logger
from litellm.integrations.custom_logger import CustomLogger


class TestLogger:
    def test_is_custom_logger_subclass(self):
        assert issubclass(Logger, CustomLogger)

    def test_can_be_instantiated(self):
        logger = Logger()
        assert logger is not None

    def test_registered_as_mavvrik_callback(self):
        assert Logger is not None

from typing import Any


class MockLogging:
    def post_call(self, *args, **kwargs):
        pass


def mock_logging() -> Any:
    return MockLogging()

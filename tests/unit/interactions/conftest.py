import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--interactions-spec-url",
        default=None,
        help="Run the Interactions OpenAPI compliance tests against this live spec instead of the vendored snapshot",
    )

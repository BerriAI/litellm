from importlib.metadata import version

import litellm


def test_dunder_version_matches_installed_distribution() -> None:
    assert litellm.__version__ == version("litellm")

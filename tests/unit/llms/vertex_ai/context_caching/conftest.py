import pytest

from litellm.llms.vertex_ai.context_caching.known_cached_contents import known_cached_contents


@pytest.fixture(autouse=True)
def _forget_known_cached_contents():
    known_cached_contents.clear()
    yield
    known_cached_contents.clear()

import pytest
import litellm


@pytest.fixture(autouse=True)
def cleanup_caches() -> None:
    litellm.in_memory_llm_clients_cache.flush_cache()

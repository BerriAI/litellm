import os

import pytest


def _skip_live_prompt_caching_test():
    if os.environ.get("LITELLM_RUN_LIVE_PROMPT_CACHING_TESTS") != "1":
        pytest.skip("Live prompt-caching E2E tests are opt-in")
    if os.environ.get("CASSETTE_REDIS_URL"):
        pytest.skip("Live prompt-caching E2E tests cannot run under VCR replay")

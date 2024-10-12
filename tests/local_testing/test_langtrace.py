import os
import sys
import time

import pytest

import litellm

sys.path.insert(0, os.path.abspath("../.."))


@pytest.fixture()
def exporter():
    litellm.success_callback = ["langtrace"]
    litellm.set_verbose = True

    return exporter


@pytest.mark.skip(
    reason="langtrace not working correctly. Asked maintainer to fix this - https://github.com/BerriAI/litellm/pull/5341#issuecomment-2408744024"
)
@pytest.mark.parametrize("model", ["claude-2.1"])  # "gpt-3.5-turbo"
def test_langtrace_logging(exporter, model):
    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "This is a test"}],
        max_tokens=1000,
        temperature=0.7,
        timeout=5,
        mock_response="hi",
    )

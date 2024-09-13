import asyncio
import io
import os
import pytest
import sys
import litellm

litellm.num_retries = 3

OPIK_API_KEY = os.environ.get("OPIK_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

def get_test_name():
    return os.environ.get('PYTEST_CURRENT_TEST', "pytest")

@pytest.mark.asyncio
@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OPEN_API_KEY not found in env")
@pytest.mark.skipif(OPIK_API_KEY is None, reason="OPIK_API_KEY not found in env")
async def test_opik_with_router():
    litellm.set_verbose = True
    litellm.success_callback = ["opik"]
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": OPENAI_API_KEY,
            },
        }
    ]
    router = litellm.Router(model_list=model_list)
    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Why is Opik logging and evaluation important?"}
        ],
        metadata = {
            "opik": {
                "metadata": {
                    "test_name": get_test_name(),
                },
                "tags": ["test"],
            },
        },
    )
    assert response.usage.prompt_tokens == 16

@pytest.mark.skipif(ANTHROPIC_API_KEY is None, reason="ANTHROPIC_API_KEY not found in env")
@pytest.mark.skipif(OPIK_API_KEY is None, reason="OPIK_API_KEY not found in env")
def test_opik_completion_with_anthropic():
    litellm.set_verbose = True
    litellm.success_callback = ["opik"]
    response = litellm.completion(
        model="claude-instant-1.2",
        messages=[{"role": "user", "content": "Why is Opik logging and evaluation important?"}],
        max_tokens=10,
        temperature=0.2,
        metadata = {
            "opik": {
                "metadata": {
                    "test_name": get_test_name(),
                },
                "tags": ["test"],
            },
        },
    )
    assert response.usage.prompt_tokens == 18

@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OPEN_API_KEY not found in env")
@pytest.mark.skipif(OPIK_API_KEY is None, reason="OPIK_API_KEY not found in env")
def test_opik_with_openai():
    litellm.set_verbose = True
    litellm.success_callback = ["opik"]
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Why is Opik logging and evaluation important?"}],
        max_tokens=10,
        temperature=0.2,
        metadata = {
            "opik": {
                "metadata": {
                    "test_name": get_test_name(),
                },
                "tags": ["test"],
            },
        },
    )
    assert response.usage.prompt_tokens == 16


@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OPEN_API_KEY not found in env")
@pytest.mark.skipif(OPIK_API_KEY is None, reason="OPIK_API_KEY not found in env")
def test_opik_with_openai_and_track():
    from opik import track, flush_tracker
    from opik.opik_context import get_current_span_data, get_current_trace_data

    litellm.set_verbose = True
    litellm.success_callback = ["opik"]

    @track()
    def complete_function(input):
        assert get_current_span_data() is not None
        assert get_current_trace_data() is not None
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "content": input,
                    "role": "user"
                },
            ],
            metadata={
                "opik": {
                    "current_span_data": get_current_span_data(),
                    "current_trace_data": get_current_trace_data(),
                    "tags": ["test"],
                    "metadata": {
                        "test_name": get_test_name(),
                    },
                },
            },
        )
        return response.to_dict()

    response = complete_function("Why is Opik logging and evaluation important?")
    flush_tracker()
    assert response["usage"]["prompt_tokens"] == 16

@pytest.mark.skipif(OPENAI_API_KEY is None, reason="OPEN_API_KEY not found in env")
@pytest.mark.skipif(OPIK_API_KEY is None, reason="OPIK_API_KEY not found in env")
def test_opik_with_streaming_openai():
    from opik import track, flush_tracker
    from opik.opik_context import get_current_trace_data, get_current_span_data

    litellm.set_verbose = True
    litellm.success_callback = ["opik"]

    @track()
    def streaming_function(input):
        messages = [{"role": "user", "content": input}]
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=messages,
            metadata = {
                "opik": {
                    "current_span_data": get_current_span_data(),
                    "current_trace_data": get_current_trace_data(),
                    "metadata": {
                        "test_name": get_test_name(),
                    },
                    "tags": ["test"],
                },
            },
            stream=True,
        )
        return response

    response = streaming_function("Why is Opik logging and evaluation important?")
    chunks = list(response)
    flush_tracker()
    assert len(chunks) > 10

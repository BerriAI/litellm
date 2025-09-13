import os
import asyncio
import pytest

pytestmark = pytest.mark.skipif(
    not (
        os.getenv("RUN_LIVE")
        and os.getenv("GEMINI_API_KEY")
        and os.getenv("LITELLM_ENABLE_PARALLEL_ACOMPLETIONS") in ("1", "true", "yes", "on")
    ),
    reason=(
        "External live test skipped. Set RUN_LIVE=1, GEMINI_API_KEY, and "
        "LITELLM_ENABLE_PARALLEL_ACOMPLETIONS=1 to run."
    ),
)


@pytest.mark.asyncio
async def test_parallel_acompletions_live_gemini():
    from litellm import Router
    from litellm.router_utils.parallel_acompletion import RouterParallelRequest

    router = Router(
        model_list=[
            {
                "model_name": "gemini-2-flash",
                "litellm_params": {
                    "model": "gemini/gemini-2.0-flash",
                    "api_key": os.environ["GEMINI_API_KEY"],
                },
            }
        ],
        timeout=30.0,
    )

    requests = [
        RouterParallelRequest(
            model="gemini-2-flash",
            messages=[{"role": "user", "content": "Say only: hello"}],
            kwargs={"timeout": 30.0, "num_retries": 0},
        ),
        RouterParallelRequest(
            model="gemini-2-flash",
            messages=[{"role": "user", "content": "Reply only: ok"}],
            kwargs={"timeout": 30.0, "num_retries": 0},
        ),
    ]

    results = await router.parallel_acompletions(
        requests, concurrency=2, preserve_order=True
    )

    assert len(results) == 2
    for i, r in enumerate(results):
        assert r.exception is None
        assert r.index == i
        assert r.request.model == requests[i].model
        assert r.request.messages == requests[i].messages
        content = r.response.choices[0].message.content
        s = (content or "").strip().lower()
        expected = "hello" if i == 0 else "ok"
        # Be lenient to minor trailing punctuation/newlines
        s = s.strip(" .!\n\r\t")
        assert s == expected


"""
Parity of the Rust core with ``router_python_core.py`` on a retry whose answer fails conversion
after a rate limit (fuzz case ff68399265ba9aee, ruled 2026-11-05: a 500 on both cores).

The Python core runs a retry inside the ``except`` of the attempt before it, so the retry's
conversion error is raised while the first attempt's RateLimitError is being handled. The
"Invalid response object" message once printed that whole chain, and exception mapping read the
rate-limit phrase in it and answered 429. The retry here runs the real conversion and the real
mapping, as ``OpenAIChatCompletion.acompletion`` and ``atext_completion`` do, at the moment the core
calls the deployment, so the error being handled then is whatever the core leaves open.
"""

from collections.abc import Awaitable, Callable
from typing import Final, cast

import pytest

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type  # pyright: ignore[reportUnknownVariableType]  # its untyped parameters
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import convert_to_model_response_object  # pyright: ignore[reportUnknownVariableType]  # its untyped start_time, end_time and dict parameters
from litellm.llms.openai.common_utils import OpenAIError
from litellm.router_core import RouterCore
from litellm.types.utils import ModelResponse
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit

MakeCore = Callable[..., RouterCore]

# A text_completion body handed to the chat conversion (gpt-4o is a chat model): no ``message``.
_TEXT_COMPLETION_BODY: Final = {
    "id": "cmpl-b6",
    "object": "text_completion",
    "created": 1,
    "model": "gpt-4o",
    "choices": [{"text": "ok", "index": 0, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _malformed_answer(**_request: object) -> BaseException:  # kwargs-ok: the scripted call's request
    """The error the OpenAI handler and ``atext_completion`` raise for ``_TEXT_COMPLETION_BODY``:
    the conversion fails, the handler wraps it in a status-500 ``OpenAIError``, and ``exception_type``
    maps that. Returned, so the host raises and logs it as any scripted error."""
    try:
        _ = convert_to_model_response_object(
            response_object=dict(_TEXT_COMPLETION_BODY), model_response_object=ModelResponse()
        )
    except Exception as conversion_error:  # noqa: BLE001  # the handler wraps every conversion failure
        try:
            raise exception_type(  # exception_type raises the mapped error itself
                model="gpt-4o",
                original_exception=OpenAIError(status_code=500, message=str(conversion_error)),
                custom_llm_provider="openai",
            )
        except Exception as mapped:  # noqa: BLE001  # the mapped error is the scripted answer
            return mapped
    raise AssertionError("the malformed body converted")


def _atext_completion(core: RouterCore) -> Callable[..., Awaitable[object]]:
    """``atext_completion``, which both cores serve outside the ``RouterCore`` protocol."""
    return cast(
        "Callable[..., Awaitable[object]]", getattr(core, "atext_completion")
    )  # cast-ok: the member is a coroutine function


@pytest.mark.asyncio
async def test_a_retry_whose_answer_fails_conversion_after_a_rate_limit_is_a_500(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "b6-d1", weight=1), deployment("g", "b6-d2", weight=1)],
        num_retries=1,
        retry_after=0,
    )
    host.answer("*", rate_limit(message="rate limit exceeded"), _malformed_answer)

    with pytest.raises(litellm.InternalServerError) as raised:
        _ = await _atext_completion(core)(model="g", prompt="hi")

    error: Final = raised.value
    assert error.status_code == 500
    assert "Invalid response object" in str(error)
    assert "KeyError: 'message'" in str(error)
    assert "rate limit exceeded" not in str(error)
    assert len(host.calls_to("atext_completion")) == 2
    assert type(host.raised[0]) is litellm.RateLimitError

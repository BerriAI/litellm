# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

"""
Module containing "timeout" decorator for sync and async callables.
"""

import asyncio

from concurrent import futures
from inspect import iscoroutinefunction
from functools import wraps
from threading import Thread
from litellm.exceptions import Timeout


def timeout(timeout_duration: float = 0.0, exception_to_raise=Timeout):
    """
    Wraps a function to raise the specified exception if execution time
    is greater than the specified timeout.

    Works with both synchronous and asynchronous callables, but with synchronous ones will introduce
    some overhead due to the backend use of threads and asyncio.

        :param float timeout_duration: Timeout duration in seconds. If none callable won't time out.
        :param OpenAIError exception_to_raise: Exception to raise when the callable times out.
            Defaults to TimeoutError.
        :return: The decorated function.
        :rtype: callable
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            async def async_func():
                return func(*args, **kwargs)

            thread = _LoopWrapper()
            thread.start()
            future = asyncio.run_coroutine_threadsafe(async_func(), thread.loop)
            local_timeout_duration = timeout_duration
            if "force_timeout" in kwargs and kwargs["force_timeout"] is not None:
                local_timeout_duration = kwargs["force_timeout"]
            elif "request_timeout" in kwargs and kwargs["request_timeout"] is not None:
                local_timeout_duration = kwargs["request_timeout"]
            try:
                result = future.result(timeout=local_timeout_duration)
            except futures.TimeoutError:
                thread.stop_loop()
                model = args[0] if len(args) > 0 else kwargs["model"]
                raise exception_to_raise(
                    f"A timeout error occurred. The function call took longer than {local_timeout_duration} second(s).",
                    model=model,  # [TODO]: replace with logic for parsing out llm provider from model name
                    llm_provider="openai",
                )
            thread.stop_loop()
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            local_timeout_duration = timeout_duration
            if "force_timeout" in kwargs:
                local_timeout_duration = kwargs["force_timeout"]
            elif "request_timeout" in kwargs and kwargs["request_timeout"] is not None:
                local_timeout_duration = kwargs["request_timeout"]
            try:
                value = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout_duration
                )
                return value
            except asyncio.TimeoutError:
                model = args[0] if len(args) > 0 else kwargs["model"]
                raise exception_to_raise(
                    f"A timeout error occurred. The function call took longer than {local_timeout_duration} second(s).",
                    model=model,  # [TODO]: replace with logic for parsing out llm provider from model name
                    llm_provider="openai",
                )

        if iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


class _LoopWrapper(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()

    def run(self) -> None:
        try:
            self.loop.run_forever()
            self.loop.call_soon_threadsafe(self.loop.close)
        except Exception as e:
            # Log exception here
            pass
        finally:
            self.loop.close()
            asyncio.set_event_loop(None)

    def stop_loop(self):
        for task in asyncio.all_tasks(self.loop):
            task.cancel()
        self.loop.call_soon_threadsafe(self.loop.stop)

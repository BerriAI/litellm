from collections.abc import Callable


async def invoke_callback(callback: Callable[[], object]) -> object:
    return callback()

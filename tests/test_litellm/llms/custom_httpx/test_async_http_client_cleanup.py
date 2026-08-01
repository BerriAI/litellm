import asyncio


def test_async_http_handler_retains_destructor_cleanup_task():
    from litellm.llms.custom_httpx.http_handler import (
        AsyncHTTPHandler,
        _async_client_cleanup_tasks,
    )

    async def exercise_cleanup():
        _async_client_cleanup_tasks.clear()
        release_close = asyncio.Event()
        close_finished = asyncio.Event()

        async def delayed_close():
            await release_close.wait()
            close_finished.set()

        handler = object.__new__(AsyncHTTPHandler)
        handler.close = delayed_close

        handler.__del__()

        assert len(_async_client_cleanup_tasks) == 1
        cleanup_task = next(iter(_async_client_cleanup_tasks))

        await asyncio.sleep(0)
        assert cleanup_task in _async_client_cleanup_tasks
        assert not close_finished.is_set()

        release_close.set()
        await cleanup_task
        await asyncio.sleep(0)

        assert close_finished.is_set()
        assert not _async_client_cleanup_tasks

    asyncio.run(exercise_cleanup())

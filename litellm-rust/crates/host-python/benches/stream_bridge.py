import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


async def _consume(handed):
    stream = await handed
    count = 0
    async for _ in stream:
        count += 1
    return count


def consume(handed):
    return loop.run_until_complete(_consume(handed))


async def _python_stream(chunks, payload, pending):
    for _ in range(chunks):
        if pending:
            await asyncio.sleep(0)
        yield payload


async def _consume_python(chunks, payload, pending):
    count = 0
    async for _ in _python_stream(chunks, payload, pending):
        count += 1
    return count


def consume_python(chunks, payload, pending):
    return loop.run_until_complete(_consume_python(chunks, payload, pending))


async def _count_steps(handed):
    from litellm.rust_bridge.lifecycle import Await, Yield

    stream = await handed
    execution = stream._execution
    chunks = 0
    awaits = 0
    step = execution.resume_value(None)
    while True:
        while isinstance(step, Await):
            awaits += 1
            step = execution.resume_value(await step.awaitable)
        if not isinstance(step, Yield):
            break
        chunks += 1
        step = execution.resume_value(None)
    execution.close()
    return chunks, awaits


def count_steps(handed):
    return loop.run_until_complete(_count_steps(handed))

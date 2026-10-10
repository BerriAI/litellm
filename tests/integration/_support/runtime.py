import asyncio
from concurrent.futures import ThreadPoolExecutor


def configure_executor() -> None:
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=32, thread_name_prefix="integration-provider")
    )

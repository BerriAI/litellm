import ast
import os
import sys
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
import asyncio
import aiohttp
import base64
import time
from typing import Tuple
import statistics


async def asyncify(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


def get_image_details(image_url) -> Tuple[str, str]:
    try:
        client = HTTPHandler(concurrent_limit=1)
        response = client.get(image_url)
        response.raise_for_status()

        content_type = response.headers.get("content-type")
        if not content_type or "image" not in content_type:
            raise ValueError(
                f"URL does not point to a valid image (content-type: {content_type})"
            )

        base64_bytes = base64.b64encode(response.content).decode("utf-8")
        return base64_bytes, content_type
    except Exception as e:
        raise e


async def get_image_details_async(image_url) -> Tuple[str, str]:
    try:
        client = AsyncHTTPHandler(concurrent_limit=1)
        response = await client.get(image_url)
        response.raise_for_status()

        content_type = response.headers.get("content-type")
        if not content_type or "image" not in content_type:
            raise ValueError(
                f"URL does not point to a valid image (content-type: {content_type})"
            )

        base64_bytes = base64.b64encode(response.content).decode("utf-8")
        return base64_bytes, content_type
    except Exception as e:
        raise e


async def get_image_details_aio(image_url) -> Tuple[str, str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type")
                if not content_type or "image" not in content_type:
                    raise ValueError(
                        f"URL does not point to a valid image (content-type: {content_type})"
                    )
                content = await response.read()
                base64_bytes = base64.b64encode(content).decode("utf-8")
                return base64_bytes, content_type
    except Exception as e:
        raise e


async def test_asyncified(urls: list[str], iterations: int = 3) -> list[float]:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        await asyncio.gather(*[asyncify(get_image_details, url) for url in urls])
        times.append(time.perf_counter() - start)
    return times


async def test_async_httpx(urls: list[str], iterations: int = 3) -> list[float]:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        await asyncio.gather(*[get_image_details_async(url) for url in urls])
        times.append(time.perf_counter() - start)
    return times


async def test_aiohttp(urls: list[str], iterations: int = 3) -> list[float]:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        await asyncio.gather(*[get_image_details_aio(url) for url in urls])
        times.append(time.perf_counter() - start)
    return times


async def run_comparison():
    urls = [
        "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
    ] * 150

    print("Testing asyncified version...")
    asyncified_times = await test_asyncified(urls)

    print("Testing async httpx version...")
    async_httpx_times = await test_async_httpx(urls)

    print("Testing aiohttp version...")
    aiohttp_times = await test_aiohttp(urls)

    print("\nResults:")
    print(
        f"Asyncified version - Mean: {statistics.mean(asyncified_times):.3f}s, Std: {statistics.stdev(asyncified_times):.3f}s"
    )
    print(
        f"Async HTTPX version - Mean: {statistics.mean(async_httpx_times):.3f}s, Std: {statistics.stdev(async_httpx_times):.3f}s"
    )
    print(
        f"Aiohttp version    - Mean: {statistics.mean(aiohttp_times):.3f}s, Std: {statistics.stdev(aiohttp_times):.3f}s"
    )
    print(
        f"Speed improvement over asyncified: {statistics.mean(asyncified_times)/statistics.mean(aiohttp_times):.2f}x"
    )
    print(
        f"Speed improvement over async httpx: {statistics.mean(async_httpx_times)/statistics.mean(aiohttp_times):.2f}x"
    )


if __name__ == "__main__":
    asyncio.run(run_comparison())

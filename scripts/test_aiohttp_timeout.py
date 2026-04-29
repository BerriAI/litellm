import argparse
import asyncio
import time

import aiohttp
import httpx

from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single httpbin delay test for LiteLLM aiohttp transport timeouts."
    )
    parser.add_argument("--delay-seconds", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--pass-timeout", type=bool, default=False)
    args = parser.parse_args()

    url = f"https://httpbin.org/delay/{args.delay_seconds}"
    timeout = httpx.Timeout(args.timeout_seconds)
    transport = LiteLLMAiohttpTransport(
        client=lambda: aiohttp.ClientSession(trust_env=True)
    )

    print(f"url={url}")
    print(f"timeout={args.timeout_seconds}s")

    started_at = time.perf_counter()
    try:
        request = httpx.Request(
            method="GET",
            url=url,
            extensions={"timeout": timeout.as_dict() if args.pass_timeout else {}},
        )
        print(timeout.as_dict() if args.pass_timeout else {})
        print(f"request.extensions['timeout']={request.extensions.get('timeout')}")

        response = await transport.handle_async_request(request)

        elapsed = time.perf_counter() - started_at
        print(f"SUCCESS status_code={response.status_code} elapsed_s={elapsed:.2f}")
        print("If elapsed is much greater than timeout, timeout is not respected.")
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        print(
            f"EXCEPTION type={type(e).__name__} elapsed_s={elapsed:.2f} detail={e}"
        )
    finally:
        await transport.aclose()


if __name__ == "__main__":
    asyncio.run(main())

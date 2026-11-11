import asyncio
import json
import sys
from importlib.metadata import version
from typing import Final

import httpx
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion


def _summary(response: ChatCompletion) -> dict[str, object]:
    choice: Final = response.choices[0]
    calls: Final = choice.message.tool_calls or ()
    return {
        "id": response.id,
        "object": response.object,
        "usage": [response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens]
        if response.usage
        else None,
        "finish_reason": choice.finish_reason,
        "content": choice.message.content,
        "tool_calls": [[call.id, call.function.name, json.loads(call.function.arguments)] for call in calls],
    }


async def _asynchronous(base_url: str, key: str, parameters: dict[str, object]) -> tuple[ChatCompletion, ...]:
    plain: Final = {name: value for name, value in parameters.items() if name != "tools"}
    async with AsyncOpenAI(
        api_key=key, base_url=base_url, max_retries=0, http_client=httpx.AsyncClient(timeout=10, trust_env=False)
    ) as client:
        return (await client.chat.completions.create(**parameters), await client.chat.completions.create(**plain))


def main(base_url: str, key: str, parameters_json: str) -> None:
    parameters: Final = json.loads(parameters_json)
    plain: Final = {name: value for name, value in parameters.items() if name != "tools"}
    with OpenAI(
        api_key=key, base_url=base_url, max_retries=0, http_client=httpx.Client(timeout=10, trust_env=False)
    ) as client:
        sync_tools: Final[ChatCompletion] = client.chat.completions.create(**parameters)
        sync_text: Final[ChatCompletion] = client.chat.completions.create(**plain)
    async_tools, async_text = asyncio.run(_asynchronous(base_url, key, parameters))
    print(
        json.dumps(
            {
                "openai": version("openai"),
                "tools": [_summary(sync_tools), _summary(async_tools)],
                "text": [_summary(sync_text), _summary(async_text)],
            }
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])

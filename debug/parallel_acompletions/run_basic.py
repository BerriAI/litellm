#!/usr/bin/env python3
import asyncio
from litellm import Router
from litellm.router_utils.parallel_acompletion import RouterParallelRequest

async def main():
    router = Router(model_list=[{"model_name": "parallel", "litellm_params": {"model": "ollama/glm4:latest", "custom_llm_provider": "ollama"}}])
    reqs = [
        RouterParallelRequest(model="parallel", messages=[{"role": "user", "content": "Say hi"}]),
        RouterParallelRequest(model="parallel", messages=[{"role": "user", "content": "Say bye"}]),
    ]
    out = await router.parallel_acompletions(reqs)
    for item in out:
        print(item.index, item.content)

if __name__ == '__main__':
    asyncio.run(main())

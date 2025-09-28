#!/usr/bin/env python3
import asyncio
from litellm import Router

async def main():
    router = Router(model_list=[{"model_name": "debug-local", "litellm_params": {"model": "ollama/glm4:latest", "custom_llm_provider": "ollama"}}])
    resp = await router.acompletion(
        model="debug-local",
        messages=[{"role": "user", "content": "List two colors"}],
    )
    print(getattr(resp.choices[0].message, "content", "").strip())

if __name__ == "__main__":
    asyncio.run(main())

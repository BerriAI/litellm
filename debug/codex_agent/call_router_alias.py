#!/usr/bin/env python3
import asyncio
import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from litellm import Router

async def main():
    alias = os.getenv('LITELLM_DEFAULT_CODE_MODEL', 'ollama/qwen2.5-coder:14b')
    provider = 'ollama' if ':' in alias else None
    if alias.startswith('codex-agent/') and os.getenv('LITELLM_ENABLE_CODEX_AGENT') != '1':
        print('codex-agent disabled; set LITELLM_ENABLE_CODEX_AGENT=1 to run this debug script.')
        return
    router = Router(model_list=[{"model_name": "debug-code", "litellm_params": {"model": alias, "custom_llm_provider": provider}}])
    resp = await router.acompletion(
        model="debug-code",
        messages=[{"role": "user", "content": "Provide one advantage of LiteLLM."}],
    )
    print(getattr(resp.choices[0].message, "content", "").strip())

if __name__ == '__main__':
    asyncio.run(main())

#!/usr/bin/env python3
import asyncio
import json
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from litellm.experimental_mcp_client.mini_agent.agent_proxy import AgentRunReq, run

async def main():
    req = AgentRunReq(
        messages=[{"role": "user", "content": "Say hi"}],
        model="debug-mini-agent",
        tool_backend="local",
        max_iterations=2,
        max_total_seconds=20,
    )
    resp = await run(req)
    print(json.dumps(resp, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

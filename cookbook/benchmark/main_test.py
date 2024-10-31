import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from fastapi import FastAPI
from memory_profiler import profile
import uvicorn

app = FastAPI()

from pydantic import BaseModel


class ExampleRequest(BaseModel):
    query: str


@app.post("/debug")
async def debug(body: ExampleRequest) -> str:
    return await main_logic(body.query)


@profile
async def main_logic(query) -> str:
    stream = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        stream=True,
    )
    result = ""
    async for chunk in stream:
        result += chunk.choices[0].delta.content or ""
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

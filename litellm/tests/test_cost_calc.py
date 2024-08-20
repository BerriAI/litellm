import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

import litellm
from litellm import Router, completion_cost, stream_chunk_builder

models = [
    dict(
        model_name="openai/gpt-3.5-turbo",
    ),
    dict(
        model_name="anthropic/claude-3-haiku-20240307",
    ),
    dict(
        model_name="together_ai/meta-llama/Llama-2-7b-chat-hf",
    ),
]

router = Router(
    model_list=[
        {
            "model_name": m["model_name"],
            "litellm_params": {
                "model": m.get("model", m["model_name"]),
            },
        }
        for m in models
    ],
    routing_strategy="simple-shuffle",
    num_retries=3,
    retry_after=1,
    timeout=60.0,
    allowed_fails=2,
    cooldown_time=0,
    debug_level="INFO",
)


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku-20240307",
        "together_ai/meta-llama/Llama-2-7b-chat-hf",
    ],
)
def test_run(model: str):
    """
    Relevant issue - https://github.com/BerriAI/litellm/issues/4965
    """
    prompt = "Hi"
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.001,
        top_p=0.001,
        max_tokens=20,
        input_cost_per_token=2,
        output_cost_per_token=2,
    )

    print(f"--------- {model} ---------")
    print(f"Prompt: {prompt}")

    response = router.completion(**kwargs)  # type: ignore
    non_stream_output = response.choices[0].message.content.replace("\n", "")  # type: ignore
    non_stream_cost_calc = response._hidden_params["response_cost"] * 100

    print(f"Non-stream output: {non_stream_output}")
    print(f"Non-stream usage : {response.usage}")  # type: ignore
    try:
        print(
            f"Non-stream cost  : {response._hidden_params['response_cost'] * 100:.4f}"
        )
    except TypeError:
        print("Non-stream cost  : NONE")
    print(f"Non-stream cost  : {completion_cost(response) * 100:.4f} (response)")

    response = router.completion(**kwargs, stream=True)  # type: ignore
    response = stream_chunk_builder(list(response), messages=kwargs["messages"])  # type: ignore
    output = response.choices[0].message.content.replace("\n", "")  # type: ignore
    streaming_cost_calc = completion_cost(response) * 100
    print(f"Stream output    : {output}")

    if output == non_stream_output:
        # assert cost is the same
        assert streaming_cost_calc == non_stream_cost_calc
    print(f"Stream usage     : {response.usage}")  # type: ignore
    print(f"Stream cost      : {streaming_cost_calc} (response)")
    print("")

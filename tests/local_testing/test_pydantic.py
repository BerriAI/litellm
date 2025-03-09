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
import asyncio
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.types.utils import (
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    Delta,
    ModelResponse,
    StreamingChoices,
    TopLogprob,
)

obj = ModelResponse(
    id="chat-f9bad6ec3c1146e99368682a0e7403fc",
    choices=[
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(content="", role=None, function_call=None, tool_calls=None),
            logprobs=ChoiceLogprobs(
                content=[
                    ChatCompletionTokenLogprob(
                        token="",
                        bytes=[],
                        logprob=-0.00018153927521780133,
                        top_logprobs=[
                            TopLogprob(
                                token="", bytes=[], logprob=-0.00018153927521780133
                            ),
                            TopLogprob(
                                token="\n\n", bytes=[10, 10], logprob=-9.062681198120117
                            ),
                        ],
                    )
                ]
            ),
        )
    ],
    created=1721976759,
    model="Meta-Llama-3-8B-Instruct",
    object="chat.completion.chunk",
    system_fingerprint=None,
)

print(obj.model_dump())

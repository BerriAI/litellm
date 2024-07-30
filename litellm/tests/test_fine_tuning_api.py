import os
import sys
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from openai import APITimeoutError as Timeout

import litellm

litellm.num_retries = 0
import logging

from litellm import create_fine_tuning_job
from litellm._logging import verbose_logger


def test_create_fine_tune_job():
    verbose_logger.setLevel(logging.DEBUG)
    file_name = "openai_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)

    file_obj = litellm.create_file(
        file=open(file_path, "rb"),
        purpose="fine-tune",
        custom_llm_provider="openai",
    )
    print("Response from creating file=", file_obj)

    response = litellm.create_fine_tuning_job(
        model="gpt-3.5-turbo",
        training_file=file_obj.id,
    )

    print("response from litellm.create_fine_tuning_job=", response)

    assert response.id is not None
    assert response.model == "gpt-3.5-turbo"

    # delete file

    # cancel ft job
    pass

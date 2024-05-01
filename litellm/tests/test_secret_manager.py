import sys, os, uuid
import time
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm import get_secret
from litellm.proxy.secret_managers.aws_secret_manager import load_aws_secret_manager


@pytest.mark.skip(reason="AWS Suspended Account")
def test_aws_secret_manager():
    load_aws_secret_manager(use_aws_secret_manager=True)

    secret_val = get_secret("litellm_master_key")

    print(f"secret_val: {secret_val}")

    assert secret_val == "sk-1234"

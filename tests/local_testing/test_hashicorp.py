import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


from litellm.secret_managers.hashicorp_secret_manager import HashicorpSecretManager

hashicorp_secret_manager = HashicorpSecretManager()


@pytest.mark.asyncio
async def test_hashicorp_secret_manager():
    secret = await hashicorp_secret_manager.async_read_secret("sample-secret")

    # assert secret == "new_test"

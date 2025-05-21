

import logging
import os
import sys
import traceback
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_openai_image_edit_litellm_sdk(image_url, sync_mode):
    from litellm import image_edit, aimage_edit

    if sync_mode:
        image_edit()
    else:
        await aimage_edit()
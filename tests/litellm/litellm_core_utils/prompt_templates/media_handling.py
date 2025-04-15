import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


@pytest.mark.parametrize(
    "http_url",
    ["https://i.pinimg.com/736x/b4/b1/be/b4b1becad04d03a9071db2817fc9fe77.jpg",
     "https://videos.pexels.com/video-files/3571264/3571264-sd_426_240_30fps.mp4"],
)
@pytest.mark.asyncio
async def test_async_convert_url_to_base64(http_url):
    from litellm.litellm_core_utils.prompt_templates.media_handling import (
        async_convert_url_to_base64
    )

    # Convert HTTP URL to base64 str.
    base64_str = await async_convert_url_to_base64(http_url)
    assert base64_str.startswith("data:")
    cached_base64_str = await async_convert_url_to_base64(http_url)
    assert base64_str == cached_base64_str

@pytest.mark.parametrize(
    "http_url",
    ["https://i.pinimg.com/736x/b4/b1/be/b4b1becad04d03a9071db2817fc9fe77.jpg",
     "https://videos.pexels.com/video-files/3571264/3571264-sd_426_240_30fps.mp4"],
)
def test_convert_url_to_base64(http_url):
    from litellm.litellm_core_utils.prompt_templates.media_handling import (
        convert_url_to_base64
    )

    # Convert HTTP URL to base64 str.
    base64_str = convert_url_to_base64(http_url)
    assert base64_str.startswith("data:")
    cached_base64_str = convert_url_to_base64(http_url)
    assert base64_str == cached_base64_str


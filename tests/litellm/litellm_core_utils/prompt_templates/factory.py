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
def test_convert_generic_media_chunk_to_openai_media_obj(http_url):
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_generic_media_chunk_to_openai_media_obj,
        convert_to_anthropic_media_obj,
    )

    # Convert HTTP URL to GenericMediaParsingChunk.
    chunk = convert_to_anthropic_media_obj(http_url, format=None)
    # Convert GenericMediaParsingChunk to base64 str.
    base64_str = convert_generic_media_chunk_to_openai_media_obj(chunk)
    # Convert base64 str to GenericMediaParsingChunk.
    new_chunk = convert_to_anthropic_media_obj(base64_str, format=None)
    assert new_chunk == chunk


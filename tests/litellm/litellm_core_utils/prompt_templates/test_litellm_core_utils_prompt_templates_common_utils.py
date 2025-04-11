import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_format_from_file_id,
    update_messages_with_model_file_ids,
)


def test_update_messages_with_model_file_ids():
    unified_file_id = (
        "litellm_proxy:application/pdf;unified_id,cbbe3534-8bf8-4386-af00-f5f6b7e370bf"
    )

    format = get_format_from_file_id(unified_file_id)

    assert format == "application/pdf"

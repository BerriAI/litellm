import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))
import json

import litellm


def test_generic_event():
    from litellm.types.llms.openai import GenericEvent

    event = {"type": "test", "test": "test"}
    event = GenericEvent(**event)
    assert event.type == "test"
    assert event.test == "test"

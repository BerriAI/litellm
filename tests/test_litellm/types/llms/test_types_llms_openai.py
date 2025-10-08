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


def test_output_item_added_event():
    from litellm.types.llms.openai import OutputItemAddedEvent

    event = {
        "type": "response.output_item.added",
        "sequence_number": 4,
        "output_index": 1,
        "item": None,
    }
    event = OutputItemAddedEvent(**event)
    assert event.type == "response.output_item.added"
    assert event.sequence_number == 4
    assert event.output_index == 1
    assert event.item is None

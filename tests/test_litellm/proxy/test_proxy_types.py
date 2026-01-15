import asyncio
import importlib
import json
import os
import socket
import subprocess
import sys
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import click
import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path


def test_audit_log_masking():
    from datetime import datetime

    from litellm.proxy._types import LiteLLM_AuditLogs

    audit_log = LiteLLM_AuditLogs(
        id="123",
        updated_at=datetime.now(),
        changed_by="test",
        changed_by_api_key="test",
        table_name="LiteLLM_VerificationToken",
        object_id="test",
        action="updated",
        updated_values=json.dumps({"key": "sk-1234567890", "token": "1q2132r222"}),
        before_value=json.dumps({"key": "sk-1234567890", "token": "1q2132r222"}),
    )

    print(audit_log.updated_values)
    json_updated_values = json.loads(audit_log.updated_values)
    assert json_updated_values["token"] == "1q2132r222"
    assert json_updated_values["key"] == "sk-1*****7890"
    assert audit_log.before_value
    json_before_value = json.loads(audit_log.before_value)
    assert json_before_value["token"] == "1q2132r222"
    assert json_before_value["key"] == "sk-1*****7890"

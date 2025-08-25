import json
import os
import sys
import httpx
import pytest
import respx

from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm_proxy_extras.utils import ProxyExtrasDBManager

def test_custom_prisma_dir(monkeypatch):
    import tempfile
    # create a temp directory
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("LITELLM_MIGRATION_DIR", temp_dir)

    ## Check if the prisma dir is the temp directory
    assert ProxyExtrasDBManager._get_prisma_dir() == temp_dir

    ## Check if the schema.prisma file is in the temp directory
    schema_path = os.path.join(temp_dir, "schema.prisma")
    assert os.path.exists(schema_path)

    ## Check if the migrations dir is in the temp directory
    migrations_dir = os.path.join(temp_dir, "migrations")
    assert os.path.exists(migrations_dir)


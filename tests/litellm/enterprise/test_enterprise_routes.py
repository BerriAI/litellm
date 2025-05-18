import json
import os
import sys
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from enterprise.litellm_enterprise.proxy.enterprise_routes import router


@pytest.fixture
def client():
    return TestClient(router)


# def test_robots_when_blocked(client):
#     """Test get_robots returns block instructions when _should_block_robots returns True"""
#     with mock.patch(
#         "litellm_enterprise.proxy.enterprise_routes._should_block_robots",
#         return_value=True,
#     ):
#         response = client.get("/robots.txt")
#         print("got response", response)
#         print("got response text", response.text)
#         print("got response headers", response.headers)
#         assert response.status_code == 200
#         assert response.text == "User-agent: *\nDisallow: /"


# def test_robots_when_not_blocked(client):
#     """Test get_robots returns 404 when _should_block_robots returns False"""
#     with mock.patch(
#         "litellm_enterprise.proxy.enterprise_routes._should_block_robots",
#         return_value=False,
#     ):
#         response = client.get("/robots.txt")
#         assert response.status_code == 404

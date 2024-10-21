import asyncio
import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from litellm.proxy.middleware.request_disconnected_middleware import (
    RequestDisconnectMiddleware,
)
import pytest
import asyncio


@pytest.fixture
def app():
    app = Starlette()
    app.add_middleware(RequestDisconnectMiddleware)

    @app.route("/test")
    async def test_route(request):
        await asyncio.sleep(5)  # Simulate a long-running operation
        return JSONResponse({"status": "completed"})

    @app.route("/quick")
    async def quick_route(request):
        return JSONResponse({"status": "quick response"})

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_normal_request(client):
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"status": "completed"}


def test_quick_response(client):
    response = client.get("/quick")
    assert response.status_code == 200
    assert response.json() == {"status": "quick response"}


@pytest.mark.asyncio
async def test_disconnected_request(client):
    async def disconnect_request():
        # Simulate a client disconnection after 1 second
        await asyncio.sleep(1)
        raise Exception("Client disconnected")

    with pytest.raises(Exception, match="Client disconnected"):
        await client.get("/test", background=disconnect_request)

    # Allow some time for the server to process the disconnection
    await asyncio.sleep(0.5)

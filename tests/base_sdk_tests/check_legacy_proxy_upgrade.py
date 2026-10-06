"""Check the same dashboard route before and after an ordinary legacy upgrade."""

import os
from importlib.metadata import version
from typing import Final

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from fastapi.testclient import TestClient

from litellm.proxy.proxy_server import app


def check_legacy_dashboard() -> None:
    client: Final = TestClient(app)
    for route in ("/ui/", "/ui/login/"):
        response: Final = client.get(route)
        assert response.status_code == 200, f"Legacy dashboard {route} returned {response.status_code}"
        assert "<html" in response.text.lower(), f"Legacy dashboard {route} did not return HTML"
    print(f"litellm {version('litellm')}, companion {version('litellm-proxy-extras')}: dashboard routes pass")


if __name__ == "__main__":
    check_legacy_dashboard()

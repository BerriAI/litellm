"""Check custom dashboard startup without the optional dashboard package."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final


def check_custom_ui(case: str) -> None:
    with TemporaryDirectory() as directory:
        root: Final = Path(directory)
        os.environ.pop("LITELLM_NON_ROOT", None)
        if case == "unset":
            os.environ.pop("LITELLM_UI_PATH", None)
        else:
            os.environ["LITELLM_UI_PATH"] = directory
        if case in ("ready", "unstructured"):
            (root / "_next").mkdir()
            (root / "index.html").write_text("custom-dashboard")
            (root / "_next/probe.js").write_text("custom-javascript")
            if case == "ready":
                (root / "login").mkdir()
                (root / "login/index.html").write_text("custom-login")
            else:
                (root / "login.html").write_text("custom-login")
        os.environ["LITELLM_FAVICON_URL"] = "https://example.com/custom.ico"
        sys.modules["litellm_proxy_extras"] = None

        from fastapi.testclient import TestClient

        from litellm.proxy.proxy_server import app

        client: Final = TestClient(app)
        response: Final = client.get("/ui/")
        if case in ("ready", "unstructured"):
            assert response.status_code == 200, response.text
            assert response.text == "custom-dashboard"
            login: Final = client.get("/ui/login/")
            assert login.status_code == 200
            assert login.text == "custom-login"
            for prefix in ("", "/litellm-asset-prefix"):
                asset: Final = client.get(f"{prefix}/_next/probe.js")
                assert asset.status_code == 200
                assert asset.text == "custom-javascript"
        else:
            assert response.status_code == 404
        favicon: Final = client.get("/get_favicon", follow_redirects=False)
        assert favicon.status_code == 307
        assert favicon.headers["location"] == "https://example.com/custom.ico"


if __name__ == "__main__":
    check_custom_ui(sys.argv[1])

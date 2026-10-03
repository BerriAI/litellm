"""Check dashboard packaging and serving from installed wheels."""

import argparse
import importlib.util
import os
import subprocess
import sys
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path
from typing import Final


def check_ui_packaging(profile: str) -> str:
    sdk_files: Final = distribution("litellm").files or ()
    bundled_ui: Final = tuple(
        str(path) for path in sdk_files if str(path).startswith("litellm/proxy/_experimental/out/")
    )
    _require(not bundled_ui, "core SDK still bundles dashboard assets")
    has_proxy_extras: Final = importlib.util.find_spec("litellm_proxy_extras") is not None
    _require(has_proxy_extras == (profile == "proxy"), f"unexpected proxy extras installation for {profile}")
    if profile == "proxy":
        ui: Final = files("litellm_proxy_extras").joinpath("ui")
        _require(ui.joinpath("index.html").is_file(), "proxy wheel is missing the dashboard entrypoint")
        _require(ui.joinpath("_next").is_dir(), "proxy wheel is missing Next.js assets")
    return "dashboard assets are installed only with the proxy extra"


def check_proxy_ui() -> str:
    from fastapi.testclient import TestClient
    from litellm.proxy.proxy_server import app, ui_path

    root: Final = Path(ui_path)
    configured_path: Final = os.getenv("LITELLM_UI_PATH")
    if configured_path is not None:
        _require(root == Path(configured_path), "proxy ignored the configured UI directory")
    client: Final = TestClient(app)
    favicon: Final = client.get("/get_favicon")
    _require(favicon.status_code == 200, "default favicon is unavailable")
    _require(favicon.content == files("litellm_proxy_extras").joinpath("ui/favicon.ico").read_bytes(), "wrong favicon")
    for route in ("", "login/"):
        response: Final = client.get(f"/ui/{route}")
        _require(response.status_code == 200, f"dashboard route {route!r} returned {response.status_code}")
        _require(
            response.content == (root / route / "index.html").read_bytes(), f"wrong dashboard content for {route!r}"
        )
    for extension in ("*.js", "*.css"):
        asset: Final = next(root.joinpath("_next").rglob(extension))
        relative: Final = asset.relative_to(root).as_posix()
        for prefix in ("", "/litellm-asset-prefix"):
            asset_response: Final = client.get(f"{prefix}/{relative}")
            _require(
                asset_response.status_code == 200, f"dashboard asset {relative} returned {asset_response.status_code}"
            )
            _require(asset_response.content == asset.read_bytes(), f"wrong dashboard asset content for {relative}")
    callback: Final = client.get("/ui/mcp/oauth/callback?code=abc&state=xyz", follow_redirects=False)
    _require(callback.status_code == 307, "nested callback route did not redirect")
    _require(callback.headers["location"].endswith("/ui/mcp/oauth/callback/?code=abc&state=xyz"), "callback redirect lost query")
    landed: Final = client.get("/ui/mcp/oauth/callback?code=abc&state=xyz")
    _require(landed.status_code == 200 and "<html" in landed.text.lower(), "nested callback page failed")
    for case in ("ready", "unstructured", "empty", "unset"):
        subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("check_custom_ui.py")), case],
            check=True,
        )
    return "installed proxy serves dashboard, nested routes, JavaScript and CSS"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("core", "proxy"), required=True)
    args: Final = parser.parse_args()
    print(check_ui_packaging(args.profile))
    if args.profile == "proxy":
        print(check_proxy_ui())

"""Live e2e: /model/info surfaces the cost map's capability flags on a deployment.

The cost map (model_prices_and_context_window.json) carries supports_video_input
on the video-capable models, and /model/info must project it onto each
deployment's model_info block. A proxy that drops the flag during resolution
reports supports_video_input as absent for every model even though the cost map
declares it.

The deployment is registered through /model/new (deleted on teardown) on the
gemini/gemini-2.5-flash backend; /model/info resolves capability flags from the
cost map without calling the provider. The expected value is read back from the
proxy's own cost map endpoint rather than hardcoded, so the test stays correct
if the map's flag ever changes.
"""

import pytest

from e2e_config import unique_marker
from models import LiteLLMParamsBody, ModelInfoEntry
from proxy_client import ProxyClient
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

BACKEND_MODEL = "gemini/gemini-2.5-pro"
GEMINI_API_KEY = "os.environ/GEMINI_API_KEY"


def _model_info_entry(entries: list[ModelInfoEntry], model_name: str) -> ModelInfoEntry:
    for entry in entries:
        if entry.model_name == model_name:
            return entry
    pytest.fail(f"{model_name} absent from /model/info; the deployment did not load")


class TestModelInfoCapabilities:
    def test_model_info_surfaces_supports_video_input_from_cost_map(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model_name = f"video-cap-{unique_marker()}"
        model_id = proxy.create_model(
            model_name,
            LiteLLMParamsBody(model=BACKEND_MODEL, api_key=GEMINI_API_KEY),
        )
        resources.defer(lambda: proxy.delete_model(model_id))

        cost_map = proxy.model_cost_map()
        backend = cost_map.get(BACKEND_MODEL)
        assert backend is not None, f"{BACKEND_MODEL} absent from the proxy cost map"
        expected = backend.supports_video_input
        assert expected is not None, (
            f"cost map declares no supports_video_input for {BACKEND_MODEL}; "
            f"pick a video-capable backend for this test"
        )

        entry = _model_info_entry(proxy.model_info(), model_name)
        assert entry.model_info.supports_video_input == expected, (
            f"/model/info supports_video_input "
            f"{entry.model_info.supports_video_input} != cost map value {expected} "
            f"for backend {BACKEND_MODEL}"
        )

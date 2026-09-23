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
    @pytest.mark.covers("mgmt.model.info.capability_flags")
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

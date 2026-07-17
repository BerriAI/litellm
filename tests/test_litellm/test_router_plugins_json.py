import json
from pathlib import Path


ROUTER_PLUGINS_PATH = Path(__file__).parents[2] / "router_plugins.json"


def test_router_plugins_json_is_valid_reference_catalog():
    data = json.loads(ROUTER_PLUGINS_PATH.read_text())

    assert isinstance(data, dict)
    assert "language-detector" in data

    language_detector = data["language-detector"]
    assert isinstance(language_detector, dict)
    assert language_detector["name"] == "litellm-plugin-language-detector"
    assert language_detector["source"] == {
        "source": "github",
        "repo": "jeann2013/language-detector",
    }
    assert language_detector["config"]["proxy"]["router_settings.plugins"] == [
        "litellm_plugin_language_detector.plugin.language_detector_plugin"
    ]
    assert language_detector["config"]["proxy"]["complexity_router_config.plugins"] == [
        "litellm_plugin_language_detector.plugin.language_detector_plugin"
    ]
    assert language_detector["signals"]["language-detector"] == {
        "language": "ISO-639-1 code or unknown",
        "confidence": "float from 0.0 to 1.0",
        "detector": "backend name",
    }

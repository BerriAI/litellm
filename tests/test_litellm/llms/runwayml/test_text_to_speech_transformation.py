"""
Test RunwayML text-to-speech transformation
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.runwayml.text_to_speech.transformation import (
    RunwayMLTextToSpeechConfig,
)


def test_openai_voice_mapping_to_runwayml():
    """
    Test that OpenAI voice names are correctly mapped to RunwayML preset IDs
    """
    config = RunwayMLTextToSpeechConfig()
    
    # Test OpenAI voice mappings
    openai_to_runway = {
        "alloy": "Maya",
        "echo": "James",
        "fable": "Bernard",
        "onyx": "Vincent",
        "nova": "Serene",
        "shimmer": "Ella",
    }
    
    for openai_voice, expected_runway_voice in openai_to_runway.items():
        mapped_voice, mapped_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={},
            voice=openai_voice,
            drop_params=False,
            kwargs={},
        )
        
        assert mapped_voice is None
        assert "runwayml_voice" in mapped_params
        assert mapped_params["runwayml_voice"]["type"] == "runway-preset"
        assert mapped_params["runwayml_voice"]["presetId"] == expected_runway_voice


def test_runwayml_native_voice_passthrough():
    """
    Test that RunwayML native voice names are passed through correctly as-is
    """
    config = RunwayMLTextToSpeechConfig()
    
    # Test various RunwayML native voices
    runway_voices = ["Bernard", "Maya", "Arjun", "Serene", "Chad"]
    
    for runway_voice in runway_voices:
        mapped_voice, mapped_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={},
            voice=runway_voice,
            drop_params=False,
            kwargs={},
        )
        
        assert mapped_voice is None
        assert "runwayml_voice" in mapped_params
        assert mapped_params["runwayml_voice"]["type"] == "runway-preset"
        assert mapped_params["runwayml_voice"]["presetId"] == runway_voice


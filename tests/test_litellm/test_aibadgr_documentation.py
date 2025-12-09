"""
Test to validate AI Badgr documentation examples in README.md
This ensures that the documented examples for using AI Badgr as an OpenAI-compatible backend are correct.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../.."))


def test_aibadgr_readme_section_exists():
    """
    Test that the README.md contains the AI Badgr section with proper examples.
    """
    readme_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "README.md"
    )
    
    # Ensure README exists
    assert os.path.exists(readme_path), "README.md not found"
    
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    
    # Check for AI Badgr section
    assert "AI Badgr" in readme_content, "AI Badgr section not found in README"
    
    # Check for base_url examples
    assert "base_url" in readme_content or "baseURL" in readme_content, \
        "base_url/baseURL parameter not mentioned in README"
    
    # Check for the AI Badgr API endpoint
    assert "https://aibadgr.com/api/v1" in readme_content, \
        "AI Badgr API endpoint not found in README"
    
    # Check for Python example
    assert "from openai import OpenAI" in readme_content, \
        "Python OpenAI SDK import not found in README"
    
    # Check for JavaScript/Node.js example
    assert "import OpenAI from 'openai'" in readme_content or \
           "import OpenAI from \"openai\"" in readme_content, \
        "JavaScript OpenAI SDK import not found in README"
    
    # Check for cURL example
    assert "curl https://aibadgr.com/api/v1/chat/completions" in readme_content, \
        "cURL example not found in README"
    
    # Check for streaming support mention
    assert '"stream": true' in readme_content or "'stream': true" in readme_content, \
        "Streaming support not mentioned in README"
    
    # Check for JSON mode support mention
    assert '"response_format"' in readme_content or "'response_format'" in readme_content, \
        "JSON mode support not mentioned in README"
    assert '"json_object"' in readme_content or "'json_object'" in readme_content, \
        "JSON object response format not mentioned in README"


def test_aibadgr_examples_have_required_parameters():
    """
    Test that the AI Badgr examples include all required parameters.
    """
    readme_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "README.md"
    )
    
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    
    # Extract the AI Badgr section
    assert "AI Badgr" in readme_content, "AI Badgr section not found"
    
    # Check Python example has required components
    assert "api_key=" in readme_content or "apiKey:" in readme_content, \
        "API key parameter not shown in examples"
    
    assert "model=" in readme_content or "model:" in readme_content, \
        "Model parameter not shown in examples"
    
    assert "messages=" in readme_content or "messages:" in readme_content, \
        "Messages parameter not shown in examples"


def test_aibadgr_section_placement():
    """
    Test that the AI Badgr section is properly placed in the README structure.
    """
    readme_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "README.md"
    )
    
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    
    # Find the position of the AI Badgr section
    aibadgr_pos = readme_content.find("AI Badgr")
    assert aibadgr_pos > 0, "AI Badgr section not found in README"
    
    # Ensure it's after the streaming section
    streaming_pos = readme_content.find("## Streaming")
    assert streaming_pos > 0, "Streaming section not found in README"
    assert aibadgr_pos > streaming_pos, \
        "AI Badgr section should be after Streaming section"

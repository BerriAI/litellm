from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail
from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail
import os

# Set a dummy API key for testing
os.environ["AIM_API_KEY"] = "test-key"

def test_aim_base_url_trailing_slash():
    # Test with trailing slash
    guardrail = AimGuardrail(api_base="https://api.aim.security/")
    assert guardrail.api_base == "https://api.aim.security"
    assert guardrail.ws_api_base.startswith("wss://api.aim.security")

    # Test without trailing slash
    guardrail2 = AimGuardrail(api_base="https://api.aim.security")
    assert guardrail2.api_base == "https://api.aim.security"
    assert guardrail2.ws_api_base.startswith("wss://api.aim.security")

    # Test with environment variable
    os.environ["AIM_API_BASE"] = "https://api.aim.security/"
    guardrail3 = AimGuardrail(api_base=None)
    assert guardrail3.api_base == "https://api.aim.security"
    assert guardrail3.ws_api_base.startswith("wss://api.aim.security")
    del os.environ["AIM_API_BASE"]

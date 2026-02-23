"""
End-to-End Tests for Databricks LiteLLM Integration
====================================================

⚠️  WARNING: These tests require REAL Databricks credentials and make ACTUAL API calls.
    They are NOT suitable for automated CI/CD pipelines.

For unit tests that use mocks and don't require credentials, see:
    test_databricks_partner_integration.py

Purpose:
    - Validate actual API connectivity with Databricks
    - Test all authentication methods (OAuth M2M, PAT, SDK)
    - Verify User-Agent strings appear correctly in Databricks audit logs
    - Test chat completions and embeddings with real models
    - Test different SDK integration methods with custom user agents

LiteLLM Integration Tests:
    This test file includes tests for different ways of calling Databricks via LiteLLM:
    
    1. LiteLLM SDK Direct - Using litellm.completion() with user_agent parameter
    2. LangChain + LiteLLM - Using ChatLiteLLM wrapper (requires langchain-community)
    3. LiteLLM Async - Using litellm.acompletion() async API
    4. LiteLLM Streaming - Using litellm.completion() with stream=True
    5. LiteLLM Embedding - Using litellm.embedding() with user_agent parameter
    
    All tests use the CUSTOM_USER_AGENT value from the config file and call
    Databricks endpoints through LiteLLM's unified interface.

Prerequisites:
    - Valid Databricks workspace access
    - Configured credentials (OAuth Service Principal, PAT, or Databricks CLI)
    - Access to serving endpoints (e.g., databricks-gpt-oss-120b)
    
Optional Dependencies (for LiteLLM integration tests):
    - pip install langchain-litellm  # For LangChain tests (recommended)

Setup:
    1. Copy the template to create your config file:
       cp databricks_config.template.txt ~/.databricks_litellm_config.txt

    2. Edit the config file with your Databricks credentials:
       - DATABRICKS_API_BASE (required)
       - DATABRICKS_HOST (required for Databricks SDK tests)
       - DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET (for OAuth)
       - DATABRICKS_API_KEY (for PAT)
       - CUSTOM_USER_AGENT (for partner attribution tests)

    3. Optionally set a custom config path:
       export DATABRICKS_TEST_CONFIG=/path/to/your/config.txt

Run with:
    cd /path/to/litellm
    python tests/test_litellm/llms/databricks/test_databricks_e2e.py

Config Options:
    TEST_AUTH_METHOD=oauth  # Test OAuth M2M authentication
    TEST_AUTH_METHOD=pat    # Test Personal Access Token
    TEST_AUTH_METHOD=sdk    # Test Databricks SDK (~/.databrickscfg)
    TEST_AUTH_METHOD=all    # Test all three methods sequentially
"""

import os
import sys

import pytest

# Skip all tests in this module during unit test runs (make test-unit)
# These are E2E tests that require real Databricks credentials
pytestmark = pytest.mark.skip(
    reason="E2E tests require real Databricks credentials. Run directly with: "
    "python tests/test_litellm/llms/databricks/test_databricks_e2e.py"
)

# Add the litellm package to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

# Config file path - can be overridden with DATABRICKS_TEST_CONFIG env var
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.databricks_litellm_config.txt")
CONFIG_FILE = os.environ.get("DATABRICKS_TEST_CONFIG", DEFAULT_CONFIG_PATH)


def load_config(config_file: str) -> dict:
    """Load configuration from file."""
    config = {}

    template_path = os.path.join(
        os.path.dirname(__file__), "databricks_config.template.txt"
    )

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n\n"
            f"To set up:\n"
            f"  1. Copy the template:\n"
            f"     cp {template_path} {config_file}\n\n"
            f"  2. Edit {config_file} with your Databricks credentials\n\n"
            f"  3. Or set a custom path:\n"
            f"     export DATABRICKS_TEST_CONFIG=/your/path/config.txt"
        )

    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if value:  # Only set if value is not empty
                    config[key] = value

    return config


def setup_environment(config: dict, auth_method: str):
    """Set up environment variables based on auth method."""
    # Clear any existing Databricks env vars (including SDK-specific ones)
    for var in [
        "DATABRICKS_API_KEY",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_API_BASE",
        "DATABRICKS_USER_AGENT",
        "LITELLM_USER_AGENT",
        "DATABRICKS_TOKEN",
        "DATABRICKS_HOST",
    ]:  # Added SDK env vars
        os.environ.pop(var, None)

    # Set auth based on method
    if auth_method == "oauth":
        if (
            "DATABRICKS_CLIENT_ID" not in config
            or "DATABRICKS_CLIENT_SECRET" not in config
        ):
            raise ValueError(
                "OAuth auth requires DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET"
            )
        # For OAuth, set the API base
        if "DATABRICKS_API_BASE" in config:
            os.environ["DATABRICKS_API_BASE"] = config["DATABRICKS_API_BASE"]
        os.environ["DATABRICKS_CLIENT_ID"] = config["DATABRICKS_CLIENT_ID"]
        os.environ["DATABRICKS_CLIENT_SECRET"] = config["DATABRICKS_CLIENT_SECRET"]
        print("  Auth method: OAuth M2M (Service Principal)")

    elif auth_method == "pat":
        if "DATABRICKS_API_KEY" not in config:
            raise ValueError("PAT auth requires DATABRICKS_API_KEY")
        # For PAT, set the API base
        if "DATABRICKS_API_BASE" in config:
            os.environ["DATABRICKS_API_BASE"] = config["DATABRICKS_API_BASE"]
        os.environ["DATABRICKS_API_KEY"] = config["DATABRICKS_API_KEY"]
        print("  Auth method: Personal Access Token (PAT)")

    elif auth_method == "sdk":
        # For SDK mode, don't set any env vars - let SDK use ~/.databrickscfg
        # But we still need to pass api_base to litellm, so set it if provided
        if "DATABRICKS_API_BASE" in config:
            os.environ["DATABRICKS_API_BASE"] = config["DATABRICKS_API_BASE"]
        print("  Auth method: Databricks SDK (automatic from ~/.databrickscfg)")

    else:
        raise ValueError(f"Unknown auth method: {auth_method}")

    # Set custom user agent if provided
    if "CUSTOM_USER_AGENT" in config:
        os.environ["DATABRICKS_USER_AGENT"] = config["CUSTOM_USER_AGENT"]
        print(f"  Custom User-Agent: {config['CUSTOM_USER_AGENT']}")


def test_user_agent_building():
    """Test User-Agent string building."""
    print("\n" + "=" * 60)
    print("TEST: User-Agent Building")
    print("=" * 60)

    from litellm.llms.databricks.common_utils import DatabricksBase

    # Test 1: Default
    ua = DatabricksBase._build_user_agent(None)
    print(f"  Default: {ua}")
    assert ua.startswith("litellm/"), f"Expected litellm/, got {ua}"
    print("  ✓ Default user agent works")

    # Test 2: With partner
    ua = DatabricksBase._build_user_agent("mycompany/1.0.0")
    print(f"  With partner: {ua}")
    assert ua.startswith("mycompany_litellm/"), f"Expected mycompany_litellm/, got {ua}"
    print("  ✓ Partner prefixing works")

    # Test 3: Partner without version
    ua = DatabricksBase._build_user_agent("acme")
    print(f"  Without version: {ua}")
    assert ua.startswith("acme_litellm/"), f"Expected acme_litellm/, got {ua}"
    print("  ✓ Partner without version works")

    print("  ✓ All user agent tests passed!")


def test_token_redaction():
    """Test sensitive data redaction."""
    print("\n" + "=" * 60)
    print("TEST: Token Redaction")
    print("=" * 60)

    from litellm.llms.databricks.common_utils import DatabricksBase

    # Test header redaction
    headers = {
        "Authorization": "Bearer dapi123456789abcdef",
        "Content-Type": "application/json",
    }
    redacted = DatabricksBase.redact_headers_for_logging(headers)
    print(f"  Original: Authorization: Bearer dapi123456789abcdef")
    print(f"  Redacted: Authorization: {redacted['Authorization']}")
    assert "[REDACTED]" in redacted["Authorization"]
    assert redacted["Content-Type"] == "application/json"
    print("  ✓ Header redaction works")

    # Test dict redaction
    data = {"api_key": "secret123", "model": "dbrx"}
    redacted = DatabricksBase.redact_sensitive_data(data)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["model"] == "dbrx"
    print("  ✓ Dict redaction works")

    # Test PAT redaction
    text = "Token: dapi_fake_test_token_for_testing"
    redacted = DatabricksBase.redact_sensitive_data(text)
    assert "dapi_fake_test" not in redacted
    print("  ✓ PAT string redaction works")

    print("  ✓ All redaction tests passed!")


def test_chat_completion(config: dict):
    """Test chat completion with Databricks."""
    print("\n" + "=" * 60)
    print("TEST: Chat Completion")
    print("=" * 60)

    import litellm

    model = config.get("TEST_CHAT_MODEL", "databricks-gpt-oss-120b")
    full_model = f"databricks/{model}"

    print(f"  Model: {full_model}")
    print(f"  API Base: {os.environ.get('DATABRICKS_API_BASE', 'Not set')}")

    try:
        response = litellm.completion(
            model=full_model,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello, LiteLLM test!' in exactly those words.",
                }
            ],
            max_tokens=50,
            temperature=0.1,
        )

        content = response.choices[0].message.content
        print(f"  Response: {content[:100]}...")
        print(f"  Model returned: {response.model}")
        print(f"  Usage: {response.usage}")
        print("  ✓ Chat completion test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Chat completion failed: {e}")
        return False


def test_chat_completion_default_user_agent(config: dict):
    """Test chat completion with default user agent (no custom agent)."""
    print("\n" + "=" * 60)
    print("TEST: Chat Completion with DEFAULT User-Agent")
    print("=" * 60)

    import litellm

    # Clear any custom user agent from environment
    saved_user_agent = os.environ.pop("DATABRICKS_USER_AGENT", None)
    saved_litellm_ua = os.environ.pop("LITELLM_USER_AGENT", None)

    try:
        from litellm._version import version
    except Exception:
        version = "unknown"

    model = config.get("TEST_CHAT_MODEL", "databricks-gpt-oss-120b")
    full_model = f"databricks/{model}"

    print(f"  Model: {full_model}")
    print(f"  Expected User-Agent: litellm/{version}")
    print(f"  (No custom user agent set)")

    try:
        response = litellm.completion(
            model=full_model,
            messages=[{"role": "user", "content": "Say 'default' only."}],
            max_tokens=10,
            # Note: NOT passing user_agent parameter
        )

        print(f"  Response: {response.choices[0].message.content}")
        print("  ✓ Default user-agent test passed!")
        print(
            f"  Note: Check Databricks Query History to verify User-Agent is 'litellm/{version}'"
        )
        return True

    except Exception as e:
        print(f"  ✗ Default user-agent test failed: {e}")
        return False

    finally:
        # Restore environment variables
        if saved_user_agent:
            os.environ["DATABRICKS_USER_AGENT"] = saved_user_agent
        if saved_litellm_ua:
            os.environ["LITELLM_USER_AGENT"] = saved_litellm_ua


def test_chat_completion_with_custom_user_agent(config: dict):
    """Test chat completion with custom user agent passed as parameter."""
    print("\n" + "=" * 60)
    print("TEST: Chat Completion with Custom User-Agent (parameter)")
    print("=" * 60)

    import litellm

    # Clear any env user agent to ensure parameter takes precedence
    saved_user_agent = os.environ.pop("DATABRICKS_USER_AGENT", None)
    saved_litellm_ua = os.environ.pop("LITELLM_USER_AGENT", None)

    try:
        from litellm._version import version
    except Exception:
        version = "unknown"

    model = config.get("TEST_CHAT_MODEL", "databricks-gpt-oss-120b")
    full_model = f"databricks/{model}"

    print(f"  Model: {full_model}")
    print(f"  Custom User-Agent param: testpartner/2.0.0")
    print(f"  Expected User-Agent: testpartner_litellm/{version}")

    try:
        response = litellm.completion(
            model=full_model,
            messages=[{"role": "user", "content": "Say 'test' only."}],
            max_tokens=10,
            user_agent="testpartner/2.0.0",  # This should result in testpartner_litellm/{version}
        )

        print(f"  Response: {response.choices[0].message.content}")
        print("  ✓ Custom user-agent test passed!")
        print(
            f"  Note: Check Databricks Query History to verify User-Agent is 'testpartner_litellm/{version}'"
        )
        return True

    except Exception as e:
        print(f"  ✗ Custom user-agent test failed: {e}")
        return False

    finally:
        # Restore environment variables
        if saved_user_agent:
            os.environ["DATABRICKS_USER_AGENT"] = saved_user_agent
        if saved_litellm_ua:
            os.environ["LITELLM_USER_AGENT"] = saved_litellm_ua


def test_chat_completion_with_env_user_agent(config: dict):
    """Test chat completion with user agent set via environment variable."""
    print("\n" + "=" * 60)
    print("TEST: Chat Completion with User-Agent from ENV VAR")
    print("=" * 60)

    import litellm

    # Set a specific user agent via environment
    test_partner = "envpartner"
    os.environ["DATABRICKS_USER_AGENT"] = test_partner

    try:
        from litellm._version import version
    except Exception:
        version = "unknown"

    model = config.get("TEST_CHAT_MODEL", "databricks-gpt-oss-120b")
    full_model = f"databricks/{model}"

    print(f"  Model: {full_model}")
    print(f"  DATABRICKS_USER_AGENT env var: {test_partner}")
    print(f"  Expected User-Agent: {test_partner}_litellm/{version}")

    try:
        response = litellm.completion(
            model=full_model,
            messages=[{"role": "user", "content": "Say 'env' only."}],
            max_tokens=10,
            # Note: NOT passing user_agent parameter - should use env var
        )

        print(f"  Response: {response.choices[0].message.content}")
        print("  ✓ Env var user-agent test passed!")
        print(
            f"  Note: Check Databricks Query History to verify User-Agent is '{test_partner}_litellm/{version}'"
        )
        return True

    except Exception as e:
        print(f"  ✗ Env var user-agent test failed: {e}")
        return False

    finally:
        # Clean up
        os.environ.pop("DATABRICKS_USER_AGENT", None)


def test_embedding(config: dict):
    """Test embeddings with Databricks."""
    print("\n" + "=" * 60)
    print("TEST: Embeddings")
    print("=" * 60)

    import litellm

    model = config.get("TEST_EMBEDDING_MODEL", "databricks-bge-large-en")
    full_model = f"databricks/{model}"

    print(f"  Model: {full_model}")

    try:
        response = litellm.embedding(
            model=full_model,
            input=["Hello, world!"],
        )

        # Handle both object and dict response formats
        if hasattr(response, "data"):
            data = response.data
        else:
            data = response.get("data", [])

        if data:
            first_item = data[0]
            if hasattr(first_item, "embedding"):
                embedding = first_item.embedding
            else:
                embedding = first_item.get("embedding", [])

            print(f"  Embedding dimensions: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            print("  ✓ Embedding test passed!")
            return True
        else:
            print("  ✗ Embedding test failed: No data in response")
            return False

    except Exception as e:
        print(f"  ✗ Embedding test failed: {e}")
        print("  (This is expected if embedding model is not available)")
        return False


def test_oauth_token_retrieval(config: dict):
    """Test OAuth M2M token retrieval."""
    print("\n" + "=" * 60)
    print("TEST: OAuth M2M Token Retrieval")
    print("=" * 60)

    if "DATABRICKS_CLIENT_ID" not in config or "DATABRICKS_CLIENT_SECRET" not in config:
        print("  Skipped: OAuth credentials not configured")
        return None

    from litellm.llms.databricks.common_utils import DatabricksBase

    try:
        db = DatabricksBase()
        token = db._get_oauth_m2m_token(
            api_base=config["DATABRICKS_API_BASE"],
            client_id=config["DATABRICKS_CLIENT_ID"],
            client_secret=config["DATABRICKS_CLIENT_SECRET"],
        )

        # Redact token for display
        redacted_token = (
            f"{token[:10]}...[REDACTED]" if len(token) > 10 else "[REDACTED]"
        )
        print(f"  Token obtained: {redacted_token}")
        print("  ✓ OAuth M2M token retrieval passed!")
        return True

    except Exception as e:
        print(f"  ✗ OAuth token retrieval failed: {e}")
        return False


# ==============================================================================
# SDK INTEGRATION TESTS - Different ways of calling Databricks via LiteLLM
# ==============================================================================


def test_litellm_sdk_with_config_user_agent(config: dict):
    """
    Test 1: LiteLLM SDK with custom user agent from config file.

    This test uses the LiteLLM SDK directly with the CUSTOM_USER_AGENT
    specified in the databricks config file.
    """
    print("\n" + "=" * 60)
    print("TEST: LiteLLM SDK with Config User-Agent")
    print("=" * 60)

    import litellm
    from litellm.llms.databricks.common_utils import DatabricksBase

    custom_ua = config.get("CUSTOM_USER_AGENT")
    if not custom_ua:
        print("  Skipped: CUSTOM_USER_AGENT not set in config")
        return None

    try:
        from litellm._version import version
    except Exception:
        version = "unknown"

    model = config.get("TEST_CHAT_MODEL", "databricks-gpt-oss-120b")
    full_model = f"databricks/{model}"

    # Build and display the final User-Agent that will be sent
    final_user_agent = DatabricksBase._build_user_agent(custom_ua)

    print(f"  Model: {full_model}")
    print(f"  Custom User-Agent from config: {custom_ua}")
    print(f"  >>> Final User-Agent sent: {final_user_agent}")

    try:
        response = litellm.completion(
            model=full_model,
            messages=[{"role": "user", "content": "Say 'LiteLLM SDK test' only."}],
            max_tokens=20,
            temperature=0.1,
            user_agent=custom_ua,  # Use config user agent
        )

        content = response.choices[0].message.content
        print(f"  Response: {content}")
        print("  ✓ LiteLLM SDK with config user-agent test passed!")
        return True

    except Exception as e:
        print(f"  ✗ LiteLLM SDK test failed: {e}")
        return False


def test_langchain_litellm_with_user_agent(config: dict):
    """
    Test 2: LangChain with LiteLLM integration.

    This test uses LangChain's ChatLiteLLM wrapper to call Databricks
    with custom user agent from config.

    Requires: pip install langchain-litellm (recommended)
              or: pip install langchain langchain-community (deprecated)
    """
    print("\n" + "=" * 60)
    print("TEST: LangChain + LiteLLM with Config User-Agent")
    print("=" * 60)

    from litellm.llms.databricks.common_utils import DatabricksBase

    custom_ua = config.get("CUSTOM_USER_AGENT")
    if not custom_ua:
        print("  Skipped: CUSTOM_USER_AGENT not set in config")
        return None

    # Try the new langchain-litellm package first, fall back to deprecated import
    ChatLiteLLM = None
    HumanMessage = None

    try:
        from langchain_litellm import ChatLiteLLM
        from langchain_core.messages import HumanMessage

        print("  Using: langchain-litellm package (recommended)")
    except ImportError:
        try:
            # Fall back to deprecated import
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                from langchain_community.chat_models import ChatLiteLLM
                from langchain_core.messages import HumanMessage
            print(
                "  Using: langchain-community (deprecated, consider: pip install langchain-litellm)"
            )
        except ImportError:
            print("  Skipped: langchain-litellm not installed")
            print("  Install with: pip install langchain-litellm")
            return None

    model = config.get("TEST_CHAT_MODEL", "databricks-gpt-oss-120b")
    full_model = f"databricks/{model}"

    # Build and display the final User-Agent that will be sent
    final_user_agent = DatabricksBase._build_user_agent(custom_ua)

    print(f"  Model: {full_model}")
    print(f"  Custom User-Agent from config: {custom_ua}")
    print(f"  >>> Final User-Agent sent: {final_user_agent}")

    try:
        # Set user agent via environment for LangChain integration
        os.environ["DATABRICKS_USER_AGENT"] = custom_ua

        chat = ChatLiteLLM(
            model=full_model,
            max_tokens=20,
            temperature=0.1,
        )

        messages = [HumanMessage(content="Say 'LangChain test' only.")]
        response = chat.invoke(messages)

        content = response.content
        print(f"  Response: {content}")
        print("  ✓ LangChain + LiteLLM with config user-agent test passed!")
        return True

    except Exception as e:
        print(f"  ✗ LangChain + LiteLLM test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up env var
        os.environ.pop("DATABRICKS_USER_AGENT", None)


def test_litellm_async_completion(config: dict):
    """
    Test 3: LiteLLM Async Completion API with custom User-Agent.

    This test uses LiteLLM's async completion API (acompletion) to call
    Databricks with custom user agent from config.
    """
    print("\n" + "=" * 60)
    print("TEST: LiteLLM Async Completion with Config User-Agent")
    print("=" * 60)

    import asyncio
    import litellm
    from litellm.llms.databricks.common_utils import DatabricksBase

    custom_ua = config.get("CUSTOM_USER_AGENT")
    if not custom_ua:
        print("  Skipped: CUSTOM_USER_AGENT not set in config")
        return None

    model = config.get("TEST_CHAT_MODEL", "databricks-gpt-oss-120b")
    full_model = f"databricks/{model}"

    # Build and display the final User-Agent that will be sent
    final_user_agent = DatabricksBase._build_user_agent(custom_ua)

    print(f"  Model: {full_model}")
    print(f"  Custom User-Agent from config: {custom_ua}")
    print(f"  >>> Final User-Agent sent: {final_user_agent}")

    async def run_async_completion():
        response = await litellm.acompletion(
            model=full_model,
            messages=[{"role": "user", "content": "Say 'LiteLLM async test' only."}],
            max_tokens=20,
            temperature=0.1,
            user_agent=custom_ua,
        )
        return response

    try:
        response = asyncio.run(run_async_completion())

        content = response.choices[0].message.content
        print(f"  Response: {content}")
        print("  ✓ LiteLLM async completion with config user-agent test passed!")
        return True

    except Exception as e:
        print(f"  ✗ LiteLLM async completion test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_litellm_streaming_completion(config: dict):
    """
    Test 4: LiteLLM Streaming Completion with custom User-Agent.

    This test uses LiteLLM's streaming completion API to call
    Databricks with custom user agent from config.
    """
    print("\n" + "=" * 60)
    print("TEST: LiteLLM Streaming Completion with Config User-Agent")
    print("=" * 60)

    import litellm
    from litellm.llms.databricks.common_utils import DatabricksBase

    custom_ua = config.get("CUSTOM_USER_AGENT")
    if not custom_ua:
        print("  Skipped: CUSTOM_USER_AGENT not set in config")
        return None

    model = config.get("TEST_CHAT_MODEL", "databricks-gpt-oss-120b")
    full_model = f"databricks/{model}"

    # Build and display the final User-Agent that will be sent
    final_user_agent = DatabricksBase._build_user_agent(custom_ua)

    print(f"  Model: {full_model}")
    print(f"  Custom User-Agent from config: {custom_ua}")
    print(f"  >>> Final User-Agent sent: {final_user_agent}")

    try:
        # Use streaming completion
        response = litellm.completion(
            model=full_model,
            messages=[
                {"role": "user", "content": "Say 'LiteLLM streaming test' only."}
            ],
            max_tokens=20,
            temperature=0.1,
            user_agent=custom_ua,
            stream=True,
        )

        # Collect streamed content
        collected_content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                collected_content += chunk.choices[0].delta.content

        print(f"  Response (streamed): {collected_content}")
        print("  ✓ LiteLLM streaming completion with config user-agent test passed!")
        return True

    except Exception as e:
        print(f"  ✗ LiteLLM streaming completion test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_litellm_embedding_with_user_agent(config: dict):
    """
    Test 5: LiteLLM Embedding API with custom User-Agent.

    This test uses LiteLLM's embedding API to call Databricks
    with custom user agent from config.
    """
    print("\n" + "=" * 60)
    print("TEST: LiteLLM Embedding with Config User-Agent")
    print("=" * 60)

    import litellm
    from litellm.llms.databricks.common_utils import DatabricksBase

    custom_ua = config.get("CUSTOM_USER_AGENT")
    if not custom_ua:
        print("  Skipped: CUSTOM_USER_AGENT not set in config")
        return None

    model = config.get("TEST_EMBEDDING_MODEL", "databricks-bge-large-en")
    full_model = f"databricks/{model}"

    # Build and display the final User-Agent that will be sent
    final_user_agent = DatabricksBase._build_user_agent(custom_ua)

    print(f"  Model: {full_model}")
    print(f"  Custom User-Agent from config: {custom_ua}")
    print(f"  >>> Final User-Agent sent: {final_user_agent}")

    try:
        response = litellm.embedding(
            model=full_model,
            input=["Hello, this is a LiteLLM embedding test with custom user agent!"],
            user_agent=custom_ua,
        )

        # Handle both object and dict response formats
        if hasattr(response, "data"):
            data = response.data
        else:
            data = response.get("data", [])

        if data:
            first_item = data[0]
            if hasattr(first_item, "embedding"):
                embedding = first_item.embedding
            else:
                embedding = first_item.get("embedding", [])

            print(f"  Embedding dimensions: {len(embedding)}")
            print(f"  First 3 values: {embedding[:3]}")
            print("  ✓ LiteLLM embedding with config user-agent test passed!")
            return True
        else:
            print("  ✗ LiteLLM embedding test failed: No data in response")
            return False

    except Exception as e:
        print(f"  ✗ LiteLLM embedding test failed: {e}")
        print("  (This may fail if embedding model is not available)")
        import traceback

        traceback.print_exc()
        return False


def run_integration_tests_for_auth_method(config: dict, auth_method: str) -> list:
    """Run integration tests for a specific auth method. Returns list of (name, result) tuples."""
    results = []

    print("\n" + "=" * 60)
    print(f"INTEGRATION TESTS - {auth_method.upper()} Authentication")
    print("=" * 60)

    # Setup environment for this auth method
    try:
        setup_environment(config, auth_method)
    except ValueError as e:
        print(f"  ✗ Setup failed: {e}")
        return [(f"[{auth_method.upper()}] Setup", False)]

    # Test OAuth token retrieval (only for oauth method)
    if auth_method == "oauth":
        results.append(
            (
                f"[{auth_method.upper()}] OAuth Token Retrieval",
                test_oauth_token_retrieval(config),
            )
        )

    # Test chat completion
    results.append(
        (f"[{auth_method.upper()}] Chat Completion", test_chat_completion(config))
    )

    # Test embeddings
    results.append((f"[{auth_method.upper()}] Embeddings", test_embedding(config)))

    return results


def main():
    print("=" * 60)
    print("DATABRICKS LITELLM INTEGRATION TESTS")
    print("=" * 60)

    # Load config
    print(f"\nLoading config from: {CONFIG_FILE}")
    try:
        config = load_config(CONFIG_FILE)
        print(f"  Loaded {len(config)} configuration values")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    # Validate required config
    if "DATABRICKS_API_BASE" not in config:
        print("\nERROR: DATABRICKS_API_BASE is required in config file")
        return 1

    auth_method = config.get("TEST_AUTH_METHOD", "pat").lower()
    print(f"\nTest Configuration:")
    print(f"  API Base: {config['DATABRICKS_API_BASE']}")
    print(f"  Auth Method: {auth_method}")

    # Run unit tests (no credentials needed)
    print("\n" + "=" * 60)
    print("UNIT TESTS (No credentials needed)")
    print("=" * 60)

    test_user_agent_building()
    test_token_redaction()

    all_results = []

    # Determine which auth methods to test
    if auth_method == "all":
        auth_methods_to_test = ["oauth", "pat", "sdk"]
        print("\n" + "#" * 60)
        print("# TESTING ALL AUTHENTICATION METHODS")
        print("#" * 60)
    else:
        auth_methods_to_test = [auth_method]

    # Run integration tests for each auth method
    for method in auth_methods_to_test:
        results = run_integration_tests_for_auth_method(config, method)
        all_results.extend(results)

    # Run User-Agent tests (only once, using the last auth method or 'pat' for 'all')
    print("\n" + "-" * 60)
    print("USER-AGENT INTEGRATION TESTS")
    print("-" * 60)

    # Setup environment for user-agent tests (use 'pat' as it's simplest)
    if auth_method == "all":
        setup_environment(config, "pat")

    # Test 1: Default user agent (no custom agent set)
    all_results.append(
        (
            "Chat with DEFAULT User-Agent",
            test_chat_completion_default_user_agent(config),
        )
    )

    # Test 2: Custom user agent passed as parameter
    all_results.append(
        (
            "Chat with Custom User-Agent (param)",
            test_chat_completion_with_custom_user_agent(config),
        )
    )

    # Test 3: User agent from environment variable
    all_results.append(
        (
            "Chat with User-Agent from ENV",
            test_chat_completion_with_env_user_agent(config),
        )
    )

    # Run SDK Integration Tests with different calling methods
    print("\n" + "#" * 60)
    print("# SDK INTEGRATION TESTS - DIFFERENT CALLING METHODS")
    print("# Using CUSTOM_USER_AGENT from config file")
    print("#" * 60)

    # Setup environment for SDK tests (use 'pat' as it's most compatible)
    setup_environment(config, "pat")

    # Test 1: LiteLLM SDK with config user agent
    all_results.append(
        (
            "LiteLLM SDK with Config User-Agent",
            test_litellm_sdk_with_config_user_agent(config),
        )
    )

    # Test 2: LangChain + LiteLLM with config user agent
    all_results.append(
        (
            "LangChain + LiteLLM with Config User-Agent",
            test_langchain_litellm_with_user_agent(config),
        )
    )

    # Test 3: LiteLLM Async Completion with config user agent
    all_results.append(
        (
            "LiteLLM Async Completion with Config User-Agent",
            test_litellm_async_completion(config),
        )
    )

    # Test 4: LiteLLM Streaming Completion with config user agent
    all_results.append(
        (
            "LiteLLM Streaming Completion with Config User-Agent",
            test_litellm_streaming_completion(config),
        )
    )

    # Test 5: LiteLLM Embedding with config user agent
    all_results.append(
        (
            "LiteLLM Embedding with Config User-Agent",
            test_litellm_embedding_with_user_agent(config),
        )
    )

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in all_results if r is True)
    failed = sum(1 for _, r in all_results if r is False)
    skipped = sum(1 for _, r in all_results if r is None)

    for name, result in all_results:
        status = (
            "✓ PASSED"
            if result is True
            else ("✗ FAILED" if result is False else "○ SKIPPED")
        )
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")

    if auth_method == "all":
        print(f"\n  Auth methods tested: {', '.join(auth_methods_to_test)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

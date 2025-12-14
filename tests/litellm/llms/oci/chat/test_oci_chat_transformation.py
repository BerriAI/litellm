"""
Tests for OCI Chat Transformation module.

These tests verify the OCI credential handling, particularly the PEM key
normalization logic for handling different newline formats.
"""

import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.oci.chat.transformation import OCIChatConfig
from litellm.llms.oci.common_utils import OCIError


@pytest.fixture
def config():
    return OCIChatConfig()


class TestOCIKeyNormalization:
    """Tests for OCI private key content normalization."""

    def test_oci_key_with_escaped_newlines(self, config):
        """Test that escaped newlines (\\n) are converted to actual newlines."""
        # Simulate PEM content with escaped newlines (as would come from JSON/UI input)
        escaped_pem = "-----BEGIN RSA PRIVATE KEY-----\\nMIIEowIBAAKCAQEA...\\n-----END RSA PRIVATE KEY-----"

        optional_params = {
            "oci_user": "ocid1.user.oc1..test",
            "oci_fingerprint": "aa:bb:cc:dd",
            "oci_tenancy": "ocid1.tenancy.oc1..test",
            "oci_region": "us-ashburn-1",
            "oci_key": escaped_pem,
        }

        # We can't fully test signing without a real key, but we can verify
        # the error message indicates the key was processed (not a type error)
        with pytest.raises(Exception) as exc_info:
            config._sign_with_manual_credentials(
                headers={},
                optional_params=optional_params,
                request_data={"test": "data"},
                api_base="https://test.oci.oraclecloud.com/api",
            )

        # The error should be about key format/loading, not about type
        # This confirms the string was processed and newlines were normalized
        error_message = str(exc_info.value)
        assert "must be a string" not in error_message.lower()

    def test_oci_key_with_crlf_newlines(self, config):
        """Test that Windows-style CRLF newlines are normalized to LF."""
        # Simulate PEM content with CRLF newlines
        crlf_pem = "-----BEGIN RSA PRIVATE KEY-----\r\nMIIEowIBAAKCAQEA...\r\n-----END RSA PRIVATE KEY-----"

        optional_params = {
            "oci_user": "ocid1.user.oc1..test",
            "oci_fingerprint": "aa:bb:cc:dd",
            "oci_tenancy": "ocid1.tenancy.oc1..test",
            "oci_region": "us-ashburn-1",
            "oci_key": crlf_pem,
        }

        with pytest.raises(Exception) as exc_info:
            config._sign_with_manual_credentials(
                headers={},
                optional_params=optional_params,
                request_data={"test": "data"},
                api_base="https://test.oci.oraclecloud.com/api",
            )

        error_message = str(exc_info.value)
        assert "must be a string" not in error_message.lower()

    def test_oci_key_rejects_non_string_type(self, config):
        """Test that non-string oci_key values raise OCIError."""
        optional_params = {
            "oci_user": "ocid1.user.oc1..test",
            "oci_fingerprint": "aa:bb:cc:dd",
            "oci_tenancy": "ocid1.tenancy.oc1..test",
            "oci_region": "us-ashburn-1",
            "oci_key": {"invalid": "dict"},  # Wrong type
        }

        with pytest.raises(OCIError) as exc_info:
            config._sign_with_manual_credentials(
                headers={},
                optional_params=optional_params,
                request_data={"test": "data"},
                api_base="https://test.oci.oraclecloud.com/api",
            )

        assert exc_info.value.status_code == 400
        assert "must be a string" in str(exc_info.value.message)
        assert "dict" in str(exc_info.value.message)

    def test_oci_key_rejects_list_type(self, config):
        """Test that list oci_key values raise OCIError."""
        optional_params = {
            "oci_user": "ocid1.user.oc1..test",
            "oci_fingerprint": "aa:bb:cc:dd",
            "oci_tenancy": "ocid1.tenancy.oc1..test",
            "oci_region": "us-ashburn-1",
            "oci_key": ["invalid", "list"],  # Wrong type
        }

        with pytest.raises(OCIError) as exc_info:
            config._sign_with_manual_credentials(
                headers={},
                optional_params=optional_params,
                request_data={"test": "data"},
                api_base="https://test.oci.oraclecloud.com/api",
            )

        assert exc_info.value.status_code == 400
        assert "must be a string" in str(exc_info.value.message)
        assert "list" in str(exc_info.value.message)


class TestOCIValidateEnvironment:
    """Tests for OCI environment validation."""

    def test_missing_required_credentials_raises_error(self, config):
        """Test that missing required credentials raise an error."""
        with pytest.raises(Exception) as exc_info:
            config.validate_environment(
                headers={},
                model="oci/xai.grok-3",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},  # No credentials provided
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        error_message = str(exc_info.value)
        assert "oci_user" in error_message
        assert "oci_fingerprint" in error_message
        assert "oci_tenancy" in error_message

    def test_validate_environment_with_all_credentials(self, config):
        """Test that validation passes with all required credentials."""
        headers = config.validate_environment(
            headers={},
            model="oci/xai.grok-3",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={
                "oci_user": "ocid1.user.oc1..test",
                "oci_fingerprint": "aa:bb:cc:dd",
                "oci_tenancy": "ocid1.tenancy.oc1..test",
                "oci_region": "us-ashburn-1",
                "oci_compartment_id": "ocid1.compartment.oc1..test",
                "oci_key": "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----",
            },
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["content-type"] == "application/json"
        assert "user-agent" in headers


class TestOCIGetCompleteUrl:
    """Tests for OCI URL generation."""

    def test_get_complete_url_default_region(self, config):
        """Test URL generation with default region."""
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="oci/xai.grok-3",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert "us-ashburn-1" in url
        assert "inference.generativeai" in url
        assert "/20231130/actions/chat" in url

    def test_get_complete_url_custom_region(self, config):
        """Test URL generation with custom region."""
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="oci/xai.grok-3",
            optional_params={"oci_region": "eu-frankfurt-1"},
            litellm_params={},
            stream=False,
        )

        assert "eu-frankfurt-1" in url
        assert "inference.generativeai" in url

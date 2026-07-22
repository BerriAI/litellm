import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Adds the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.gdc.chat.transformation import GDCGeminiConfig

TEST_API_KEY = '{"type": "gdch_service_account", "project_id": "test-project"}'
TEST_MODEL = "gdc/gemini-2.5-flash"
TEST_API_BASE = "https://gdc-endpoint.com"
TEST_PROJECT = "test-project"
TEST_LOCATION = "test-location"


class TestGDCGeminiConfig:
    def test_get_complete_url(self):
        config = GDCGeminiConfig()
        url = config.get_complete_url(
            api_base=TEST_API_BASE,
            api_key=None,
            model=TEST_MODEL,
            optional_params={
                "vertex_project": TEST_PROJECT,
                "vertex_location": TEST_LOCATION,
            },
            litellm_params={},
        )
        assert (
            url
            == f"{TEST_API_BASE}/v1/projects/{TEST_PROJECT}/locations/{TEST_LOCATION}/chat/completions"
        )

    def test_get_complete_url_adds_https_scheme(self):
        config = GDCGeminiConfig()
        url = config.get_complete_url(
            api_base="gdc-endpoint.com",
            api_key=None,
            model=TEST_MODEL,
            optional_params={},
            litellm_params={
                "vertex_project": TEST_PROJECT,
                "vertex_location": TEST_LOCATION,
            },
        )
        assert url.startswith("https://gdc-endpoint.com/v1/projects/")

    def test_get_complete_url_preformed_base_returned_as_is(self):
        config = GDCGeminiConfig()
        preformed = f"{TEST_API_BASE}/v1/projects/{TEST_PROJECT}/locations/{TEST_LOCATION}/chat/completions"
        url = config.get_complete_url(
            api_base=preformed,
            api_key=None,
            model=TEST_MODEL,
            optional_params={"vertex_project": TEST_PROJECT},
            litellm_params={},
        )
        assert url == preformed

    def test_get_complete_url_missing_api_base(self):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="api_base/host is required for GDC Gemini"):
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model=TEST_MODEL,
                optional_params={
                    "vertex_project": TEST_PROJECT,
                    "vertex_location": TEST_LOCATION,
                },
                litellm_params={},
            )

    def test_get_complete_url_missing_project(self):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="project is required for GDC Gemini"):
            config.get_complete_url(
                api_base=TEST_API_BASE,
                api_key=None,
                model=TEST_MODEL,
                optional_params={},
                litellm_params={},
            )

    def test_get_complete_url_missing_location(self):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="location is required for GDC Gemini"):
            config.get_complete_url(
                api_base=TEST_API_BASE,
                api_key=None,
                model=TEST_MODEL,
                optional_params={"vertex_project": TEST_PROJECT},
                litellm_params={},
            )

    def test_get_complete_url_accepts_vertex_ai_aliases(self):
        config = GDCGeminiConfig()
        url = config.get_complete_url(
            api_base=TEST_API_BASE,
            api_key=None,
            model=TEST_MODEL,
            optional_params={},
            litellm_params={
                "vertex_ai_project": TEST_PROJECT,
                "vertex_ai_location": TEST_LOCATION,
            },
        )
        assert (
            url
            == f"{TEST_API_BASE}/v1/projects/{TEST_PROJECT}/locations/{TEST_LOCATION}/chat/completions"
        )

    def test_get_complete_url_preformed_base_is_authoritative_over_litellm_params(self):
        config = GDCGeminiConfig()
        preformed = f"{TEST_API_BASE}/v1/projects/pinned-project/locations/pinned-loc/chat/completions"
        url = config.get_complete_url(
            api_base=preformed,
            api_key=None,
            model=TEST_MODEL,
            optional_params={"vertex_project": "attacker-optional", "vertex_location": "attacker-loc"},
            litellm_params={
                "vertex_project": "attacker-project",
                "vertex_location": "attacker-loc",
            },
        )
        assert url == preformed

    def test_get_complete_url_preformed_base_needs_no_project_param(self):
        config = GDCGeminiConfig()
        preformed = f"{TEST_API_BASE}/v1/projects/pinned-project/locations/pinned-loc/chat/completions"
        url = config.get_complete_url(
            api_base=preformed,
            api_key=None,
            model=TEST_MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == preformed

    def test_deployment_project_takes_precedence_over_request(self):
        config = GDCGeminiConfig()
        url = config.get_complete_url(
            api_base=TEST_API_BASE,
            api_key=None,
            model=TEST_MODEL,
            optional_params={
                "vertex_project": "caller-project",
                "vertex_location": "caller-location",
            },
            litellm_params={
                "vertex_project": "deployment-project",
                "vertex_location": "deployment-location",
            },
        )
        assert url == (
            f"{TEST_API_BASE}/v1/projects/deployment-project"
            "/locations/deployment-location/chat/completions"
        )

    @patch("google.auth.load_credentials_from_dict")
    @patch("requests.Session")
    def test_validate_environment(self, mock_session, mock_load_creds):
        mock_creds = MagicMock()
        mock_creds.token = "mock-token"
        mock_creds.with_gdch_audience.return_value = mock_creds
        mock_load_creds.return_value = (mock_creds, None)

        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        config = GDCGeminiConfig()
        result = config.validate_environment(
            headers={},
            model=TEST_MODEL,
            messages=[],
            optional_params={
                "vertex_project": TEST_PROJECT,
                "vertex_location": TEST_LOCATION,
            },
            litellm_params={},
            api_key=TEST_API_KEY,
            api_base=TEST_API_BASE,
        )

        assert result["Authorization"] == "Bearer mock-token"
        assert result["Content-Type"] == "application/json"
        assert result["x-goog-user-project"] == f"projects/{TEST_PROJECT}"

        mock_creds.with_gdch_audience.assert_called_once_with(TEST_API_BASE)
        mock_creds.refresh.assert_called_once()
        assert mock_session_instance.verify is True

    def test_validate_environment_strips_audience_trailing_slash(self):
        config = GDCGeminiConfig()
        mock_creds = MagicMock()
        mock_creds.token = "mock-token"
        mock_creds.with_gdch_audience.return_value = mock_creds

        with patch(
            "google.auth.load_credentials_from_dict", return_value=(mock_creds, None)
        ), patch("requests.Session"):
            config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={},
                litellm_params={"vertex_project": TEST_PROJECT},
                api_key=TEST_API_KEY,
                api_base="https://gdc-endpoint.com/",
            )

        mock_creds.with_gdch_audience.assert_called_once_with("https://gdc-endpoint.com")

    def test_validate_environment_audience_is_host_for_preformed_base(self):
        config = GDCGeminiConfig()
        mock_creds = MagicMock()
        mock_creds.token = "mock-token"
        mock_creds.with_gdch_audience.return_value = mock_creds

        with patch(
            "google.auth.load_credentials_from_dict", return_value=(mock_creds, None)
        ), patch("requests.Session"):
            config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={},
                litellm_params={
                    "vertex_project": "deployment-project",
                    "vertex_location": "deployment-loc",
                },
                api_key=TEST_API_KEY,
                api_base=f"{TEST_API_BASE}/v1/projects/embedded/locations/embedded/chat/completions",
            )

        mock_creds.with_gdch_audience.assert_called_once_with(TEST_API_BASE)

    def test_validate_environment_missing_api_base(self, monkeypatch):
        monkeypatch.setattr(litellm, "api_base", None, raising=False)
        monkeypatch.setattr(litellm, "gdc_api_base", None, raising=False)
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="api_base/host is required for GDC Gemini"):
            config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={},
                litellm_params={"vertex_project": TEST_PROJECT},
                api_key=TEST_API_KEY,
                api_base=None,
            )

    def test_validate_environment_missing_api_key(self):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="api_key is required for GDC Gemini"):
            config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={},
                litellm_params={"vertex_project": TEST_PROJECT},
                api_key=None,
                api_base=TEST_API_BASE,
            )

    def test_validate_environment_missing_project(self):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="project is required for GDC Gemini"):
            config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=TEST_API_KEY,
                api_base=TEST_API_BASE,
            )

    def test_validate_environment_raw_token_used_as_bearer(self):
        config = GDCGeminiConfig()
        headers = config.validate_environment(
            headers={},
            model=TEST_MODEL,
            messages=[],
            optional_params={},
            litellm_params={"vertex_project": TEST_PROJECT},
            api_key="ya29.raw-access-token",
            api_base=TEST_API_BASE,
        )
        assert headers["Authorization"] == "Bearer ya29.raw-access-token"
        assert headers["x-goog-user-project"] == f"projects/{TEST_PROJECT}"

    def test_validate_environment_bad_credentials_raise_auth_error(self):
        config = GDCGeminiConfig()
        with patch(
            "google.auth.load_credentials_from_dict",
            side_effect=ValueError("bad creds"),
        ):
            with pytest.raises(
                Exception, match="Failed to load service account credentials"
            ):
                config.validate_environment(
                    headers={},
                    model=TEST_MODEL,
                    messages=[],
                    optional_params={},
                    litellm_params={"vertex_project": TEST_PROJECT},
                    api_key=TEST_API_KEY,
                    api_base=TEST_API_BASE,
                )

    def test_validate_environment_string_false_disables_token_caching(self):
        config = GDCGeminiConfig()
        mock_creds = MagicMock()
        mock_creds.token = "mock-token"
        mock_creds.with_gdch_audience.return_value = mock_creds

        with patch(
            "google.auth.load_credentials_from_dict", return_value=(mock_creds, None)
        ), patch("requests.Session"), patch.object(
            config, "_cached_fetch_token"
        ) as mock_cached:
            config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={},
                litellm_params={
                    "vertex_project": TEST_PROJECT,
                    "gdc_token_caching": "false",
                },
                api_key=TEST_API_KEY,
                api_base=TEST_API_BASE,
            )

        mock_cached.assert_not_called()

    def test_validate_environment_token_caching_path(self):
        config = GDCGeminiConfig()
        mock_creds = MagicMock()
        mock_creds.token = "cached-token"
        mock_creds.valid = True
        mock_creds.with_gdch_audience.return_value = mock_creds

        with patch(
            "google.auth.load_credentials_from_dict", return_value=(mock_creds, None)
        ):
            headers = config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={},
                litellm_params={
                    "vertex_project": TEST_PROJECT,
                    "gdc_token_caching": True,
                },
                api_key=TEST_API_KEY,
                api_base=TEST_API_BASE,
            )

        assert headers["Authorization"] == "Bearer cached-token"
        mock_creds.refresh.assert_not_called()

    def test_validate_environment_preserves_content_type_but_rebinds_quota_project(self):
        config = GDCGeminiConfig()
        headers = config.validate_environment(
            headers={
                "Content-Type": "text/plain",
                "x-goog-user-project": "projects/attacker",
            },
            model=TEST_MODEL,
            messages=[],
            optional_params={},
            litellm_params={"vertex_project": TEST_PROJECT},
            api_key="raw-token",
            api_base=TEST_API_BASE,
        )
        assert headers["Content-Type"] == "text/plain"
        assert headers["x-goog-user-project"] == f"projects/{TEST_PROJECT}"

    @pytest.mark.parametrize(
        "header_name", ["x-goog-user-project", "X-Goog-User-Project", "X-GOOG-USER-PROJECT"]
    )
    def test_validate_environment_strips_caller_forwarded_quota_header(self, header_name):
        config = GDCGeminiConfig()
        mock_creds = MagicMock()
        mock_creds.token = "tok"
        mock_creds.with_gdch_audience.return_value = mock_creds
        preformed = f"{TEST_API_BASE}/v1/projects/deployment-proj/locations/us-central1/chat/completions"
        with patch(
            "google.auth.load_credentials_from_dict", return_value=(mock_creds, None)
        ):
            headers = config.validate_environment(
                headers={header_name: "projects/attacker"},
                model=TEST_MODEL,
                messages=[],
                optional_params={"vertex_project": "attacker-proj"},
                litellm_params={},
                api_key=TEST_API_KEY,
                api_base=preformed,
            )
        quota_values = [v for k, v in headers.items() if k.lower() == "x-goog-user-project"]
        assert quota_values == ["projects/deployment-proj"]

    @pytest.mark.parametrize(
        "bad", ["p/locations/l/chat/completions?", "a/b", "a?b", "a#b", "..", "a b", "a:b", "a%2Fb"]
    )
    def test_get_complete_url_rejects_project_path_injection(self, bad):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="vertex_project must be a plain identifier"):
            config.get_complete_url(
                api_base=TEST_API_BASE,
                api_key=None,
                model=TEST_MODEL,
                optional_params={"vertex_project": bad, "vertex_location": TEST_LOCATION},
                litellm_params={},
            )

    @pytest.mark.parametrize("bad", ["../../evil", "l/chat/completions", "l?x", ".."])
    def test_get_complete_url_rejects_location_path_injection(self, bad):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="vertex_location must be a plain identifier"):
            config.get_complete_url(
                api_base=TEST_API_BASE,
                api_key=None,
                model=TEST_MODEL,
                optional_params={"vertex_project": TEST_PROJECT, "vertex_location": bad},
                litellm_params={},
            )

    @pytest.mark.parametrize("good", ["test-project", "us-central1", "123456", "proj_1", "MyProj-2"])
    def test_get_complete_url_accepts_valid_ids(self, good):
        config = GDCGeminiConfig()
        url = config.get_complete_url(
            api_base=TEST_API_BASE,
            api_key=None,
            model=TEST_MODEL,
            optional_params={"vertex_project": good, "vertex_location": good},
            litellm_params={},
        )
        assert url == f"{TEST_API_BASE}/v1/projects/{good}/locations/{good}/chat/completions"

    def test_validate_environment_rejects_project_path_injection(self):
        config = GDCGeminiConfig()
        with pytest.raises(Exception, match="vertex_project must be a plain identifier"):
            config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={"vertex_project": "p/../admin"},
                litellm_params={},
                api_key="raw-token",
                api_base=TEST_API_BASE,
            )

    def test_validate_environment_quota_header_bound_to_deployment_url(self):
        config = GDCGeminiConfig()
        mock_creds = MagicMock()
        mock_creds.token = "tok"
        mock_creds.with_gdch_audience.return_value = mock_creds
        preformed = f"{TEST_API_BASE}/v1/projects/deployment-proj/locations/us-central1/chat/completions"
        with patch(
            "google.auth.load_credentials_from_dict", return_value=(mock_creds, None)
        ):
            headers = config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={"vertex_project": "attacker-proj"},
                litellm_params={},
                api_key=TEST_API_KEY,
                api_base=preformed,
            )
        assert headers["x-goog-user-project"] == "projects/deployment-proj"

    def test_validate_environment_quota_header_pinned_to_preformed_url(self):
        config = GDCGeminiConfig()
        mock_creds = MagicMock()
        mock_creds.token = "tok"
        mock_creds.with_gdch_audience.return_value = mock_creds
        preformed = f"{TEST_API_BASE}/v1/projects/url-proj/locations/us-central1/chat/completions"
        with patch(
            "google.auth.load_credentials_from_dict", return_value=(mock_creds, None)
        ):
            headers = config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params={"vertex_project": "attacker-proj"},
                litellm_params={"vertex_project": "override-proj"},
                api_key=TEST_API_KEY,
                api_base=preformed,
            )
        assert headers["x-goog-user-project"] == "projects/url-proj"

    def test_transform_request(self):
        config = GDCGeminiConfig()
        data = config.transform_request(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={
                "vertex_project": TEST_PROJECT,
                "vertex_location": TEST_LOCATION,
            },
            litellm_params={"ssl_verify": True},
            headers={},
        )
        assert data["model"] == "gemini-2.5-flash"
        assert "vertex_project" not in data
        assert "vertex_location" not in data
        assert "ssl_verify" not in data

    def test_load_creds_from_key_ignores_file_paths(self, tmp_path):
        config = GDCGeminiConfig()
        creds_file = tmp_path / "service_account.json"
        creds_file.write_text(
            '{"type": "gdch_service_account", "project_id": "host-only-project"}'
        )

        creds, is_service_account = config._load_creds_from_key(str(creds_file))

        assert creds is None
        assert is_service_account is False

    def test_load_creds_from_key_rejects_non_gdch_credential_types(self):
        config = GDCGeminiConfig()
        external_account = (
            '{"type": "external_account", '
            '"token_url": "http://169.254.169.254/latest/api/token", '
            '"credential_source": {"url": "http://169.254.169.254/"}}'
        )
        with patch(
            "google.auth.load_credentials_from_dict",
            return_value=(MagicMock(), None),
        ) as mock_load:
            with pytest.raises(ValueError, match="GDCH service account"):
                config._load_creds_from_key(external_account)
        mock_load.assert_not_called()

    def test_validate_environment_rejects_non_gdch_credential_without_refresh(self):
        config = GDCGeminiConfig()
        mock_creds = MagicMock()
        mock_creds.token = "leaked-token"
        mock_creds.with_gdch_audience.return_value = mock_creds
        malicious = (
            '{"type": "external_account", '
            '"token_url": "http://169.254.169.254/latest/api/token"}'
        )

        with patch(
            "google.auth.load_credentials_from_dict",
            return_value=(mock_creds, None),
        ) as mock_load, patch("requests.Session") as mock_session:
            with pytest.raises(
                Exception, match="Failed to load service account credentials"
            ):
                config.validate_environment(
                    headers={},
                    model=TEST_MODEL,
                    messages=[],
                    optional_params={},
                    litellm_params={"vertex_project": TEST_PROJECT},
                    api_key=malicious,
                    api_base=TEST_API_BASE,
                )

        mock_load.assert_not_called()
        mock_session.assert_not_called()
        mock_creds.refresh.assert_not_called()

    def test_validate_environment_does_not_read_api_key_file_path(self, tmp_path):
        config = GDCGeminiConfig()
        creds_file = tmp_path / "service_account.json"
        creds_file.write_text(
            '{"type": "service_account", "project_id": "host-only-project"}'
        )

        headers = config.validate_environment(
            headers={},
            model=TEST_MODEL,
            messages=[],
            optional_params={},
            litellm_params={
                "vertex_project": TEST_PROJECT,
                "vertex_location": TEST_LOCATION,
            },
            api_key=str(creds_file),
            api_base=TEST_API_BASE,
        )

        assert headers["Authorization"] == f"Bearer {creds_file}"
        assert headers["x-goog-user-project"] == f"projects/{TEST_PROJECT}"

    @pytest.mark.parametrize(
        "val, env_value, default, expected",
        [
            (True, None, True, True),
            (False, "true", True, False),
            ("literal", None, True, "literal"),
            (None, None, True, True),
            (None, None, False, False),
            (None, "true", False, True),
            (None, "1", False, True),
            (None, "on", False, True),
            (None, "false", True, False),
            (None, "0", True, False),
            (None, "off", True, False),
            (None, "verbose", True, "verbose"),
        ],
    )
    def test_read_env_bool(self, monkeypatch, val, env_value, default, expected):
        config = GDCGeminiConfig()
        env_var = "GDC_TEST_FLAG"
        if env_value is None:
            monkeypatch.delenv(env_var, raising=False)
        else:
            monkeypatch.setenv(env_var, env_value)
        assert config._read_env_bool(val, env_var, default=default) == expected

    def test_cached_fetch_token_keys_by_credential(self):
        config = GDCGeminiConfig()

        def make_creds(token):
            creds = MagicMock()
            creds.with_gdch_audience.return_value = creds
            creds.valid = True
            creds.token = token
            return creds

        creds_a = make_creds("token-a")
        creds_b = make_creds("token-b")

        assert (
            config._cached_fetch_token(creds_a, TEST_API_BASE, True, api_key="key-a")
            == "token-a"
        )
        assert (
            config._cached_fetch_token(creds_b, TEST_API_BASE, True, api_key="key-b")
            == "token-b"
        )
        # same credential identity reuses the cached entry
        config._cached_fetch_token(creds_a, TEST_API_BASE, True, api_key="key-a")
        creds_a.with_gdch_audience.assert_called_once()

    def test_cached_fetch_token_refreshes_when_invalid(self):
        config = GDCGeminiConfig()
        creds = MagicMock()
        creds.with_gdch_audience.return_value = creds
        creds.valid = False
        creds.token = "refreshed"

        with patch.object(config, "_fetch_auth") as mock_fetch:
            token = config._cached_fetch_token(
                creds, TEST_API_BASE, True, api_key="key"
            )

        assert token == "refreshed"
        mock_fetch.assert_called_once()

    def test_init_sets_up_lock_and_cache(self):
        config = GDCGeminiConfig()
        assert config._gdch_creds_cache == {}
        assert config._creds_lock is not None


class TestCompleteGDC:
    @patch("litellm.main.base_llm_http_handler.completion")
    def test_complete_gdc_resolves_key_and_base(self, mock_completion, monkeypatch):
        from litellm.main import gdc_transformation

        mock_completion.return_value = MagicMock()
        monkeypatch.setattr(litellm, "gdc_key", "resolved-key", raising=False)
        monkeypatch.setattr(
            litellm, "gdc_api_base", "https://resolved-base.com", raising=False
        )
        monkeypatch.setattr(litellm, "api_base", None, raising=False)

        litellm.completion(
            model="gdc/gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            vertex_project=TEST_PROJECT,
            vertex_location=TEST_LOCATION,
        )

        assert mock_completion.called
        _, kwargs = mock_completion.call_args
        assert kwargs["custom_llm_provider"] == "gdc"
        assert kwargs["api_key"] == "resolved-key"
        assert kwargs["api_base"] == "https://resolved-base.com"
        assert kwargs["provider_config"] is gdc_transformation

    @patch("litellm.main.base_llm_http_handler.completion")
    def test_complete_gdc_prefers_gdc_api_base_over_global(
        self, mock_completion, monkeypatch
    ):
        mock_completion.return_value = MagicMock()
        monkeypatch.setattr(litellm, "gdc_key", "resolved-key", raising=False)
        monkeypatch.setattr(
            litellm, "gdc_api_base", "https://gdc-specific.com", raising=False
        )
        monkeypatch.setattr(
            litellm, "api_base", "https://other-provider.com", raising=False
        )

        litellm.completion(
            model="gdc/gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            vertex_project=TEST_PROJECT,
            vertex_location=TEST_LOCATION,
        )

        _, kwargs = mock_completion.call_args
        assert kwargs["api_base"] == "https://gdc-specific.com"

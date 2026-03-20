"""Base class for LiteLLM integration tests.

Supports both local (mock) and remote testing modes via environment variables:
- USE_LOCAL_LITELLM: When "true", uses local LiteLLM at localhost:4000 (default: false)
- USE_MOCK_MODELS: When "true", uses mock model names (default: false)
- LITELLM_API_KEY: API key for remote LiteLLM (required when USE_LOCAL_LITELLM=false)
- LITELLM_BASE_URL: Base URL for remote LiteLLM (required when USE_LOCAL_LITELLM=false)
"""

import enum
import os
import time
import uuid
from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Union

import httpx
import openai
import pytest
import requests
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

LOCAL_LITELLM_BASE_URL = "http://localhost:4000"
LOCAL_MOCK_SERVER_URL = "http://localhost:8090"

if "USE_LOCAL_LITELLM" not in os.environ:
    os.environ["USE_LOCAL_LITELLM"] = "true"
if "USE_MOCK_MODELS" not in os.environ:
    os.environ["USE_MOCK_MODELS"] = "true"
if "USE_STATE_TRACKER" not in os.environ:
    os.environ["USE_STATE_TRACKER"] = "true"
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://llmproxy:dbpassword9090@localhost:5432/litellm"


def use_local_litellm() -> bool:
    return os.environ.get("USE_LOCAL_LITELLM", "false").lower() == "true"


def use_remote_litellm() -> bool:
    return not use_local_litellm()


def use_mock_models() -> bool:
    return os.environ.get("USE_MOCK_MODELS", "false").lower() == "true"


def get_local_litellm_base_url() -> str:
    return LOCAL_LITELLM_BASE_URL


def get_remote_litellm_base_url() -> str:
    return os.environ.get("LITELLM_BASE_URL", "").rstrip("/")


def get_litellm_base_url() -> str:
    if use_local_litellm():
        return get_local_litellm_base_url()
    return get_remote_litellm_base_url()


def get_litellm_api_key() -> str:
    if use_local_litellm():
        return "sk-1234"
    return os.environ.get("LITELLM_API_KEY", "")


def get_mock_server_base_url() -> str:
    return LOCAL_MOCK_SERVER_URL


def get_responses_model_name() -> str:
    if use_mock_models():
        return "openai-fake-gpt-4o"
    return "gpt-4o-mini-2024-07-18"


def model_id(param) -> str:
    """Generate a test ID from a model name or tuple containing model name.

    Handles both:
    - String: "gpt-4o-mini" -> "gpt_4o_mini"
    - Tuple: ("gpt-4o", "openai/gpt-4o") -> "gpt_4o"
    """
    if isinstance(param, tuple):
        name = param[0]
    else:
        name = param
    return name.replace("-", "_").replace(".", "_")


def generate_test_id(
    params: Tuple[str, ...],
    test_name: str = "test",
) -> str:
    """Generate test ID from model parameters tuple.

    Handles two tuple formats:
    - 6 elements: (provider, deployment, model_name, api_version, action, reason)
    - 7 elements: (provider, deployment, model_name, api_version, model_id, action, reason)

    Uses model_id (position 4) if 7 elements, otherwise model_name (position 2).
    """
    provider = params[0]
    deployment = params[1]
    api_version = params[3]

    if len(params) == 7:
        identifier = params[4]  # model_id
    else:
        identifier = params[2]  # model_name

    test_id = "/".join([provider, deployment, api_version, identifier, test_name])
    return test_id.replace("-", "_").replace(".", "_")


class ModelTestAction(enum.Enum):
    NOT_APPLICABLE = 1
    SKIP = 2
    RUN = 3
    WARN_ON_FAIL = 4

    def applicable(self) -> bool:
        return self.value != ModelTestAction.NOT_APPLICABLE.value


class BaseLiteLLMIntegrationTest(ABC):
    """Base class for all LiteLLM integration tests.

    Supports both local/mock and remote testing based on environment variables.
    """

    @staticmethod
    def get_api_key() -> str:
        return get_litellm_api_key()

    @staticmethod
    def get_base_url() -> str:
        return get_litellm_base_url()

    @staticmethod
    def get_ca_bundle_path() -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # change if needed

    @classmethod
    def _get_ssl_verify_setting(cls) -> Union[bool, str]:
        """Get the appropriate SSL verification setting based on mode.

        Returns path string (not SSLContext) for compatibility with both
        requests and httpx libraries.
        """
        if use_local_litellm():
            return False
        ca_bundle_path = cls.get_ca_bundle_path()
        if os.path.exists(ca_bundle_path):
            return ca_bundle_path
        return True

    @classmethod
    def setup_class(cls):
        cls.api_key = cls.get_api_key()
        cls.base_url = cls.get_base_url()

        if not cls.api_key:
            pytest.fail(
                "API key is not available. Set LITELLM_API_KEY or USE_LOCAL_LITELLM=true",
            )
        if not cls.base_url:
            pytest.fail(
                "Base URL is not available. Set LITELLM_BASE_URL or USE_LOCAL_LITELLM=true",
            )

        verify_setting = cls._get_ssl_verify_setting()

        if use_remote_litellm() and isinstance(verify_setting, str):
            os.environ["REQUESTS_CA_BUNDLE"] = verify_setting
            os.environ["CURL_CA_BUNDLE"] = verify_setting
            print(f"Using CA bundle: {verify_setting}")

        cls.openai_client = openai.OpenAI(
            base_url=cls.base_url,
            api_key=cls.api_key,
            http_client=httpx.Client(verify=verify_setting),
        )

    @classmethod
    def make_request(
        cls,
        method: str,
        endpoint: str,
        timeout_secs: int,
        **kwargs,
    ) -> requests.Response:
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {cls.api_key}"
        kwargs["headers"] = headers
        kwargs.setdefault("timeout", timeout_secs)
        kwargs.setdefault("verify", cls._get_ssl_verify_setting())

        url = f"{cls.base_url}{endpoint}"
        return requests.request(method, url, **kwargs)

    @staticmethod
    def generate_request_id() -> str:
        return f"req-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def get_timeout_secs(model_name: str) -> int:
        model_lower = model_name.lower()
        slow_models = ["gpt-5", "gpt_5", "o1", "claude-opus", "claude_opus", "o3", "o4"]

        if any(slow_model in model_lower for slow_model in slow_models):
            return 300
        return 60

    @staticmethod
    def generate_unique_filename(extension: str = "txt") -> str:
        return f"test_{time.time()}.{extension}"

    @staticmethod
    def extract_model_params(model_data: Dict[str, Any]) -> Tuple[str, str, str, str]:
        """Extract standardized parameters from model data."""
        model_name = model_data.get("model_name", "")
        model_info = model_data.get("model_info", {})
        provider = model_info.get("litellm_provider", "unknown")
        litellm_params = model_data.get("litellm_params", {})

        if provider == "azure":
            api_base = litellm_params.get("api_base", "unknown")
            if api_base != "unknown" and "//" in api_base:
                domain_name = api_base.split("//")[1]
                deployment = domain_name.split(".")[0]
            else:
                deployment = "unknown"
            api_version = litellm_params.get("api_version", "unknown")
        elif provider in ["bedrock", "bedrock_converse"]:
            deployment = litellm_params.get("aws_region_name", "unknown")
            api_version = "unknown"
        else:
            deployment = "unknown"
            api_version = "unknown"

        return provider, deployment, model_name, api_version

    @classmethod
    def _fetch_all_models_from_litellm(cls) -> List[Dict[str, Any]]:
        base_url = cls.get_base_url()
        api_key = cls.get_api_key()

        if not api_key or not base_url:
            return []

        verify_setting = cls._get_ssl_verify_setting()

        response = requests.get(
            f"{base_url}/model/info",
            headers={"Authorization": f"Bearer {api_key}"},
            verify=verify_setting,
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch all models from {base_url}. Response code: {response.status_code}",
            )

        data = response.json()
        return data.get("data", [])

    @classmethod
    def _fetch_all_approved_models(cls) -> List[Dict[str, Any]]:
        return cls._fetch_all_models_from_litellm()

    @classmethod
    def build_model_test_params(
        cls,
        should_skip_model: Callable[
            [str, str, str, str, Dict[str, Any]],
            Tuple["ModelTestAction", str],
        ],
        include_model_id: bool = False,
        include_load_balanced: bool = False,
    ) -> List[Tuple[str, ...]]:
        """Build test parameters from all approved models.

        Args:
            should_skip_model: Callback that determines if a model should be skipped.
                Signature: (provider, deployment, model_name, api_version, model_info) -> (action, reason)
            include_model_id: If True, includes model_id in tuple (7 elements), else 6 elements.
            include_load_balanced: If True, adds extra tests for load-balanced model groups.

        Returns:
            List of tuples with model test parameters.
            - 6-element: (provider, deployment, model_name, api_version, action, reason)
            - 7-element: (provider, deployment, model_name, api_version, model_id, action, reason)
        """
        models = cls._fetch_all_approved_models()
        test_params: List[Tuple[str, ...]] = []
        models_by_model_name: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)

        for model_data in models:
            model_info = model_data.get("model_info", {}) or {}

            provider, deployment, model_name, api_version = cls.extract_model_params(
                model_data,
            )

            model_test_action, model_test_action_reason = should_skip_model(
                provider,
                deployment,
                model_name,
                api_version,
                model_info,
            )

            if model_test_action.applicable():
                if include_model_id:
                    model_id = str(model_info.get("id"))
                    params_tuple: Tuple[str, ...] = (
                        provider,
                        deployment,
                        model_name,
                        api_version,
                        model_id,
                        model_test_action,
                        model_test_action_reason,
                    )
                else:
                    params_tuple = (
                        provider,
                        deployment,
                        model_name,
                        api_version,
                        model_test_action,
                        model_test_action_reason,
                    )

                test_params.append(params_tuple)

                if include_load_balanced:
                    models_by_model_name[model_name].append(params_tuple)

        if include_load_balanced and include_model_id:
            for load_balanced_model_name, deployments in models_by_model_name.items():
                if len(deployments) <= 1:
                    continue

                first_deployment = deployments[0]
                test_params.append(
                    (
                        first_deployment[0],  # provider
                        "load_balanced",
                        load_balanced_model_name,
                        "load_balanced",
                        load_balanced_model_name,  # model_id = model_name for LB
                        first_deployment[5],  # model_test_action
                        first_deployment[6],  # model_test_action_reason
                    ),
                )

        return test_params


class UserKeyTestMixin:
    """Mixin for tests that need to create users and API keys."""

    allowed_routes: list[str] = []

    _base_url: str = None
    _master_api_key: str = None
    admin_client: httpx.Client = None

    @classmethod
    def setup_admin_client(cls):
        cls._base_url = get_litellm_base_url()
        cls._master_api_key = get_litellm_api_key()
        verify_setting = (
            False
            if use_local_litellm()
            else BaseLiteLLMIntegrationTest._get_ssl_verify_setting()
        )
        cls.admin_client = httpx.Client(base_url=cls._base_url, verify=verify_setting)

    @classmethod
    def teardown_admin_client(cls):
        if cls.admin_client:
            cls.admin_client.close()

    @staticmethod
    def unique_suffix() -> str:
        return f"{time.strftime('%Y%m%d%H%M%S')}{int(time.time() * 1000) % 1000:03d}"

    @classmethod
    def create_user_and_key(cls, user_suffix: str) -> tuple[str, str, str]:
        user_email = f"test-user-{user_suffix}-{cls.unique_suffix()}@test.com"
        user_response = cls.admin_client.post(
            "/user/new",
            json={
                "user_email": user_email,
                "user_alias": user_email,
                "user_role": "internal_user",
                "auto_create_key": "false",
            },
            headers={
                "Authorization": f"Bearer {cls._master_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        assert user_response.status_code == 200, (
            f"Failed to create user: {user_response.status_code} - {user_response.text}"
        )
        user_id = user_response.json().get("user_id")

        key_alias = user_email.replace("@", "-at-").replace(".", "-")
        key_response = cls.admin_client.post(
            "/key/generate",
            json={
                "user_id": user_id,
                "key_alias": key_alias,
                "allowed_routes": cls.allowed_routes,
            },
            headers={
                "Authorization": f"Bearer {cls._master_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        assert key_response.status_code == 200, (
            f"Failed to create key: {key_response.status_code} - {key_response.text}"
        )
        api_key = key_response.json().get("key")

        print(f"Created user {user_email}")
        return user_id, api_key, user_email

    @classmethod
    def create_user_key_and_client(
        cls,
        user_suffix: str,
    ) -> tuple[str, str, str, openai.OpenAI]:
        user_id, api_key, user_email = cls.create_user_and_key(user_suffix)
        verify_setting = (
            False
            if use_local_litellm()
            else BaseLiteLLMIntegrationTest._get_ssl_verify_setting()
        )
        client = openai.OpenAI(
            base_url=cls._base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=verify_setting),
        )
        return user_id, api_key, user_email, client

    @classmethod
    def create_key_and_client(
        cls,
        user_id: str,
        key_suffix: str,
    ) -> tuple[str, openai.OpenAI]:
        key_alias = f"additional-key-{key_suffix}-{cls.unique_suffix()}"
        key_response = cls.admin_client.post(
            "/key/generate",
            json={
                "user_id": user_id,
                "key_alias": key_alias,
                "allowed_routes": cls.allowed_routes,
            },
            headers={
                "Authorization": f"Bearer {cls._master_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        assert key_response.status_code == 200, (
            f"Failed to create additional key: {key_response.status_code} - {key_response.text}"
        )
        api_key = key_response.json().get("key")
        verify_setting = (
            False
            if use_local_litellm()
            else BaseLiteLLMIntegrationTest._get_ssl_verify_setting()
        )
        client = openai.OpenAI(
            base_url=cls._base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=verify_setting),
        )
        print(f"Created additional key for user {user_id}")
        return api_key, client
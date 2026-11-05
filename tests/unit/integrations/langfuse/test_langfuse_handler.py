from collections.abc import Iterator
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

from litellm.integrations.langfuse.langfuse import LangFuseLogger
from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler
from litellm.litellm_core_utils.specialty_caches.dynamic_logging_cache import DynamicLoggingCache
from litellm.types.utils import StandardCallbackDynamicParams


@pytest.fixture(autouse=True)
def langfuse_api_unreachable() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=httpx.ConnectError("langfuse is unreachable"))
        yield router


@pytest.fixture
def dynamic_logging_cache():
    return DynamicLoggingCache()


@pytest.fixture
def globalLangfuseLogger(request: pytest.FixtureRequest) -> LangFuseLogger | None:
    if request.param is None:
        return None
    return LangFuseLogger(
        langfuse_public_key="global_public_key",
        langfuse_secret="global_secret",
        langfuse_host="https://global.langfuse.com",
    )


standard_params_1 = StandardCallbackDynamicParams(
    langfuse_public_key="test_public_key",
    langfuse_secret="test_secret",
    langfuse_host="https://test.langfuse.com",
)

standard_params_2 = StandardCallbackDynamicParams(
    langfuse_public_key="test_public_key",
    langfuse_secret_key="test_secret",
    langfuse_host="https://test.langfuse.com",
)


@pytest.mark.parametrize("globalLangfuseLogger", [None, "global"], indirect=True)
@pytest.mark.parametrize("standard_params", [standard_params_1, standard_params_2])
def test_get_langfuse_logger_for_request_with_dynamic_params(
    dynamic_logging_cache, globalLangfuseLogger, standard_params
):
    result = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=standard_params,
        in_memory_dynamic_logger_cache=dynamic_logging_cache,
        globalLangfuseLogger=globalLangfuseLogger,
    )

    assert isinstance(result, LangFuseLogger)
    assert result.public_key == "test_public_key"
    assert result.secret_key == "test_secret"
    assert result.langfuse_host == "https://test.langfuse.com"

    logger_for_identical_repeat_request = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=standard_params,
        in_memory_dynamic_logger_cache=dynamic_logging_cache,
        globalLangfuseLogger=globalLangfuseLogger,
    )
    assert logger_for_identical_repeat_request is result


@pytest.mark.parametrize("globalLangfuseLogger", [None, "global"], indirect=True)
def test_get_langfuse_logger_for_request_with_no_dynamic_params(dynamic_logging_cache, globalLangfuseLogger):
    result = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=StandardCallbackDynamicParams(),
        in_memory_dynamic_logger_cache=dynamic_logging_cache,
        globalLangfuseLogger=globalLangfuseLogger,
    )

    assert result is not None
    assert isinstance(result, LangFuseLogger)

    if globalLangfuseLogger is not None:
        assert result.public_key == "global_public_key"
        assert result.secret_key == "global_secret"
        assert result.langfuse_host == "https://global.langfuse.com"


def test_dynamic_langfuse_credentials_are_passed():
    params_with_credentials = StandardCallbackDynamicParams(
        langfuse_public_key="test_key",
        langfuse_secret="test_secret",
        langfuse_host="https://test.langfuse.com",
    )
    assert LangFuseHandler._dynamic_langfuse_credentials_are_passed(params_with_credentials) is True

    params_without_credentials = StandardCallbackDynamicParams()
    assert LangFuseHandler._dynamic_langfuse_credentials_are_passed(params_without_credentials) is False

    params_partial_credentials = StandardCallbackDynamicParams(langfuse_public_key="test_key")
    assert LangFuseHandler._dynamic_langfuse_credentials_are_passed(params_partial_credentials) is True


def test_get_dynamic_langfuse_logging_config():
    dynamic_params = StandardCallbackDynamicParams(
        langfuse_public_key="dynamic_key",
        langfuse_secret="dynamic_secret",
        langfuse_host="https://dynamic.langfuse.com",
    )
    config = LangFuseHandler.get_dynamic_langfuse_logging_config(dynamic_params)
    assert config["langfuse_public_key"] == "dynamic_key"
    assert config["langfuse_secret"] == "dynamic_secret"
    assert config["langfuse_host"] == "https://dynamic.langfuse.com"

    empty_params = StandardCallbackDynamicParams()
    config = LangFuseHandler.get_dynamic_langfuse_logging_config(empty_params)
    assert config["langfuse_public_key"] is None
    assert config["langfuse_secret"] is None
    assert config["langfuse_host"] is None


def test_return_global_langfuse_logger():
    mock_cache = Mock()
    global_logger = LangFuseLogger(langfuse_public_key="global_key", langfuse_secret="global_secret")

    result = LangFuseHandler._return_global_langfuse_logger(global_logger, mock_cache)
    assert result == global_logger

    mock_cache.get_cache.return_value = global_logger
    result = LangFuseHandler._return_global_langfuse_logger(None, mock_cache)
    assert result == global_logger

    mock_cache.get_cache.return_value = None
    with patch.object(
        LangFuseHandler,
        "_create_langfuse_logger_from_credentials",
        return_value=global_logger,
    ):
        result = LangFuseHandler._return_global_langfuse_logger(None, mock_cache)
        assert result == global_logger


def test_get_langfuse_logger_for_request_with_cached_logger():
    mock_cache = Mock()
    cached_logger = LangFuseLogger(langfuse_public_key="cached_key", langfuse_secret="cached_secret")
    mock_cache.get_cache.return_value = cached_logger

    dynamic_params = StandardCallbackDynamicParams(
        langfuse_public_key="test_key",
        langfuse_secret="test_secret",
        langfuse_host="https://test.langfuse.com",
    )

    result = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=dynamic_params,
        in_memory_dynamic_logger_cache=mock_cache,
        globalLangfuseLogger=None,
    )

    assert result == cached_logger
    mock_cache.get_cache.assert_called_once()

import os


def pytest_sessionstart(session):
    _add_default_redis_env_variables()
    _add_default_azure_env_variables()


def _add_default_redis_env_variables():
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6379")
    os.environ.setdefault("REDIS_PASSWORD", "")

def _add_default_azure_env_variables():
    os.environ.setdefault("AZURE_OPENAI_API_KEY", "empty_key")
    os.environ.setdefault("AZURE_API_BASE", "https://dummy.openai.azure.com/")
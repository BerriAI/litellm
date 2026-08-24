import httpx

from litellm.llms.base import BaseLLM


def test_exit_accepts_the_context_manager_arguments_and_closes_the_session():
    """
    `__exit__` is always called with (exc_type, exc_val, exc_tb). A `self`-only signature
    raises TypeError before the session is ever closed, leaking the connection pool.
    """
    llm = BaseLLM()
    llm._client_session = httpx.Client()

    llm.__exit__(None, None, None)

    assert llm._client_session.is_closed

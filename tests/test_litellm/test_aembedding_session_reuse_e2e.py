"""
Regression test for commit 819a6b5f18

Ensures shared_session is in all_litellm_params to prevent
"Object of type ClientSession is not JSON serializable" errors.
"""
import os
import sys
import inspect

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.types.utils import all_litellm_params


def test_shared_session_in_all_litellm_params():
    """
    CRITICAL: shared_session must be in all_litellm_params.
    
    If missing, it gets passed to provider APIs causing JSON serialization errors.
    Regression test for commit 819a6b5f18.
    """
    assert "shared_session" in all_litellm_params


def test_openai_embedding_passes_shared_session():
    """
    Verify shared_session flows through the complete call chain.
    
    Full chain: litellm.embedding() -> OpenAI.embedding() -> _get_openai_client() 
                -> AsyncHTTPHandler -> _create_async_transport() -> _create_aiohttp_transport()
    """
    import litellm
    from litellm.llms.openai.openai import OpenAIChatCompletion
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    
    # Step 1: litellm.embedding() extracts and passes shared_session
    main_source = inspect.getsource(litellm.embedding)
    assert 'shared_session' in main_source
    
    # Step 2: OpenAI handlers pass it forward
    aembedding_source = inspect.getsource(OpenAIChatCompletion.aembedding)
    embedding_source = inspect.getsource(OpenAIChatCompletion.embedding)
    assert 'shared_session=shared_session' in aembedding_source
    assert 'shared_session=shared_session' in embedding_source
    
    # Step 3: _get_openai_client passes it to AsyncHTTPHandler
    client_source = inspect.getsource(OpenAIChatCompletion._get_openai_client)
    assert 'shared_session' in client_source
    
    # Step 4: AsyncHTTPHandler.create_client passes it to _create_async_transport
    create_client_source = inspect.getsource(AsyncHTTPHandler.create_client)
    assert 'shared_session=shared_session' in create_client_source
    
    # Step 5: _create_async_transport passes it to _create_aiohttp_transport
    async_transport_source = inspect.getsource(AsyncHTTPHandler._create_async_transport)
    assert 'shared_session=shared_session' in async_transport_source
    
    # Step 6: _create_aiohttp_transport uses it
    aiohttp_transport_source = inspect.getsource(AsyncHTTPHandler._create_aiohttp_transport)
    assert 'shared_session' in aiohttp_transport_source

import importlib
from unittest.mock import patch, MagicMock

import pytest

# Target functions to exercise header-building code paths without network


def _reset_litellm():
    import litellm as _ll
    importlib.reload(_ll)
    return _ll


@pytest.mark.parametrize(
    "api_base,expected",
    [
        ("https://ollama.com/api/chat", {"Authorization": "sk-XYZ"}),
        ("https://myhost.local:11434/api/chat", {"Authorization": "Bearer sk-XYZ"}),
    ],
)
def test_ollama_chat_headers_for_api_key(api_base, expected):
    ll = _reset_litellm()
    with patch("litellm.llms.ollama_chat.HTTPHandler", autospec=True):
        # Patch module_level_client to avoid real HTTP
        with patch.object(ll, "module_level_client") as client:
            # Simulate simple OK response with .json() access where needed
            client.post.return_value = MagicMock(status_code=200, json=lambda: {"message": {"content": "ok"}})
            from litellm.llms.ollama_chat import get_ollama_response

            get_ollama_response(
                model="ollama/llama2",
                messages=[{"role": "user", "content": "hi"}],
                api_base=api_base,
                api_key="sk-XYZ",
                streaming=False,
            )
            # Assert headers passed to client.post
            assert client.post.called, "client.post not called"
            kwargs = client.post.call_args.kwargs
            assert kwargs.get("headers") == expected


@pytest.mark.parametrize(
    "api_base,expected",
    [
        ("https://ollama.com/api/embed", {"Authorization": "sk-KEY"}),
        ("http://localhost:11434/api/embed", {"Authorization": "Bearer sk-KEY"}),
    ],
)
def test_ollama_embeddings_headers(api_base, expected):
    ll = _reset_litellm()
    # Patch async and sync clients used by embeddings handlers
    with patch.object(ll, "module_level_client") as client:
        client.post.return_value = MagicMock(status_code=200, json=lambda: {"embeddings": [[0.0, 1.0]]})
        from litellm.llms.ollama.completion.handler import ollama_embeddings

        ollama_embeddings(
            model="llama2",
            prompts=["hello"],
            api_base=api_base,
            api_key="sk-KEY",
            optional_params={},
            model_response=ll.utils.EmbeddingResponse(),
            logging_obj=None,
            encoding=None,
        )
        assert client.post.called, "client.post not called"
        kwargs = client.post.call_args.kwargs
        assert kwargs.get("headers") == expected


@pytest.mark.parametrize(
    "api_base,expected",
    [
        ("https://ollama.com/api/chat", {"Authorization": "sk-KEY"}),
        ("http://127.0.0.1:11434/api/chat", {"Authorization": "Bearer sk-KEY"}),
    ],
)
@pytest.mark.asyncio
async def test_ollama_async_streaming_headers(api_base, expected):
    ll = _reset_litellm()
    # Patch httpx.AsyncClient.stream context manager path inside ollama_async_streaming
    with patch("httpx.AsyncClient", autospec=True) as AsyncClient:
        cm = AsyncClient.return_value.__aenter__.return_value
        # mock .stream().__aenter__ response
        # But the code uses: async with client.stream(**_request) as response:
        # So mock client.stream to return an async context manager with status_code 200 and iterate lines
        class DummyStream:
            async def __aenter__(self):
                self.status_code = 200
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                for _ in range(0):
                    yield ""

        cm.stream.return_value = DummyStream()

        from litellm.llms.ollama_chat import ollama_async_streaming

        async for _ in ollama_async_streaming(
            url=api_base,
            api_key="sk-KEY",
            data={},
            logging_obj=None,
            client=cm,
        ):
            pass
        # Verify headers passed into client.stream
        assert cm.stream.called
        kwargs = cm.stream.call_args.kwargs
        assert kwargs.get("headers") == expected

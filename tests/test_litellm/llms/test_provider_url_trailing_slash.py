from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig
from litellm.llms.heroku.chat.transformation import HerokuChatConfig
from litellm.llms.jina_ai.embedding.transformation import JinaAIEmbeddingConfig
from litellm.llms.modelscope.chat.transformation import ModelScopeChatConfig
from litellm.llms.moonshot.chat.transformation import MoonshotChatConfig
from litellm.llms.ollama.chat.transformation import OllamaChatConfig
from litellm.llms.ollama.completion.transformation import OllamaConfig
from litellm.llms.perplexity.embedding.transformation import PerplexityEmbeddingConfig
from litellm.llms.topaz.image_variations.transformation import TopazImageVariationConfig
from litellm.llms.voyage.embedding.transformation import VoyageEmbeddingConfig
from litellm.llms.voyage.embedding.transformation_contextual import VoyageContextualEmbeddingConfig
from litellm.llms.voyage.embedding.transformation_multimodal import VoyageMultimodalEmbeddingConfig


def test_deepseek_get_complete_url_trailing_slash():
    config = DeepSeekChatConfig()
    url_without_slash = config.get_complete_url(
        api_base="https://custom.deepseek.com",
        api_key=None,
        model="deepseek-chat",
        optional_params={},
        litellm_params={},
    )
    url_with_slash = config.get_complete_url(
        api_base="https://custom.deepseek.com/",
        api_key=None,
        model="deepseek-chat",
        optional_params={},
        litellm_params={},
    )
    url_with_endpoint = config.get_complete_url(
        api_base="https://custom.deepseek.com/chat/completions/",
        api_key=None,
        model="deepseek-chat",
        optional_params={},
        litellm_params={},
    )
    url_default = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="deepseek-chat",
        optional_params={},
        litellm_params={},
    )
    assert url_without_slash == "https://custom.deepseek.com/chat/completions"
    assert url_with_slash == "https://custom.deepseek.com/chat/completions"
    assert url_with_endpoint == "https://custom.deepseek.com/chat/completions"
    assert url_default == "https://api.deepseek.com/beta/chat/completions"
    assert "//chat/completions" not in url_with_slash


def test_ollama_get_complete_url_trailing_slash():
    chat_config = OllamaChatConfig()
    url_without_slash = chat_config.get_complete_url(
        api_base="http://localhost:11434",
        api_key=None,
        model="llama3",
        optional_params={},
        litellm_params={},
    )
    url_with_slash = chat_config.get_complete_url(
        api_base="http://localhost:11434/",
        api_key=None,
        model="llama3",
        optional_params={},
        litellm_params={},
    )
    url_with_endpoint = chat_config.get_complete_url(
        api_base="http://localhost:11434/api/chat/",
        api_key=None,
        model="llama3",
        optional_params={},
        litellm_params={},
    )
    url_default = chat_config.get_complete_url(
        api_base=None,
        api_key=None,
        model="llama3",
        optional_params={},
        litellm_params={},
    )
    assert url_without_slash == "http://localhost:11434/api/chat"
    assert url_with_slash == "http://localhost:11434/api/chat"
    assert url_with_endpoint == "http://localhost:11434/api/chat"
    assert url_default == "http://localhost:11434/api/chat"
    assert "//api/chat" not in url_with_slash

    completion_config = OllamaConfig()
    comp_url_without_slash = completion_config.get_complete_url(
        api_base="http://localhost:11434",
        api_key=None,
        model="llama3",
        optional_params={},
        litellm_params={},
    )
    comp_url_with_slash = completion_config.get_complete_url(
        api_base="http://localhost:11434/",
        api_key=None,
        model="llama3",
        optional_params={},
        litellm_params={},
    )
    comp_url_with_endpoint = completion_config.get_complete_url(
        api_base="http://localhost:11434/api/generate/",
        api_key=None,
        model="llama3",
        optional_params={},
        litellm_params={},
    )
    comp_url_default = completion_config.get_complete_url(
        api_base=None,
        api_key=None,
        model="llama3",
        optional_params={},
        litellm_params={},
    )
    assert comp_url_without_slash == "http://localhost:11434/api/generate"
    assert comp_url_with_slash == "http://localhost:11434/api/generate"
    assert comp_url_with_endpoint == "http://localhost:11434/api/generate"
    assert comp_url_default == "http://localhost:11434/api/generate"
    assert "//api/generate" not in comp_url_with_slash


def test_moonshot_get_complete_url_trailing_slash():
    config = MoonshotChatConfig()
    url_without_slash = config.get_complete_url(
        api_base="https://api.moonshot.ai/v1",
        api_key=None,
        model="moonshot-v1-8k",
        optional_params={},
        litellm_params={},
    )
    url_with_slash = config.get_complete_url(
        api_base="https://api.moonshot.ai/v1/",
        api_key=None,
        model="moonshot-v1-8k",
        optional_params={},
        litellm_params={},
    )
    url_with_endpoint = config.get_complete_url(
        api_base="https://api.moonshot.ai/v1/chat/completions/",
        api_key=None,
        model="moonshot-v1-8k",
        optional_params={},
        litellm_params={},
    )
    url_default = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="moonshot-v1-8k",
        optional_params={},
        litellm_params={},
    )
    assert url_without_slash == "https://api.moonshot.ai/v1/chat/completions"
    assert url_with_slash == "https://api.moonshot.ai/v1/chat/completions"
    assert url_with_endpoint == "https://api.moonshot.ai/v1/chat/completions"
    assert url_default == "https://api.moonshot.ai/v1/chat/completions"
    assert "//chat/completions" not in url_with_slash


def test_modelscope_get_complete_url_trailing_slash():
    config = ModelScopeChatConfig()
    url_without_slash = config.get_complete_url(
        api_base="https://api-inference.modelscope.cn/v1",
        api_key=None,
        model="qwen",
        optional_params={},
        litellm_params={},
    )
    url_with_slash = config.get_complete_url(
        api_base="https://api-inference.modelscope.cn/v1/",
        api_key=None,
        model="qwen",
        optional_params={},
        litellm_params={},
    )
    url_with_endpoint = config.get_complete_url(
        api_base="https://api-inference.modelscope.cn/v1/chat/completions/",
        api_key=None,
        model="qwen",
        optional_params={},
        litellm_params={},
    )
    url_default = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="qwen",
        optional_params={},
        litellm_params={},
    )
    assert url_without_slash == "https://api-inference.modelscope.cn/v1/chat/completions"
    assert url_with_slash == "https://api-inference.modelscope.cn/v1/chat/completions"
    assert url_with_endpoint == "https://api-inference.modelscope.cn/v1/chat/completions"
    assert url_default == "https://api-inference.modelscope.cn/v1/chat/completions"
    assert "//chat/completions" not in url_with_slash


def test_jina_ai_get_complete_url_trailing_slash():
    config = JinaAIEmbeddingConfig()
    url_without_slash = config.get_complete_url(
        api_base="https://api.jina.ai/v1",
        api_key=None,
        model="jina-embeddings-v2",
        optional_params={},
        litellm_params={},
    )
    url_with_slash = config.get_complete_url(
        api_base="https://api.jina.ai/v1/",
        api_key=None,
        model="jina-embeddings-v2",
        optional_params={},
        litellm_params={},
    )
    url_with_endpoint = config.get_complete_url(
        api_base="https://api.jina.ai/v1/embeddings/",
        api_key=None,
        model="jina-embeddings-v2",
        optional_params={},
        litellm_params={},
    )
    url_default = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="jina-embeddings-v2",
        optional_params={},
        litellm_params={},
    )
    assert url_without_slash == "https://api.jina.ai/v1/embeddings"
    assert url_with_slash == "https://api.jina.ai/v1/embeddings"
    assert url_with_endpoint == "https://api.jina.ai/v1/embeddings"
    assert url_default == "https://api.jina.ai/v1/embeddings"
    assert "//embeddings" not in url_with_slash


def test_voyage_get_complete_url_trailing_slash():
    config = VoyageEmbeddingConfig()
    url_without_slash = config.get_complete_url(
        api_base="https://api.voyageai.com/v1",
        api_key=None,
        model="voyage-large-2",
        optional_params={},
        litellm_params={},
    )
    url_with_slash = config.get_complete_url(
        api_base="https://api.voyageai.com/v1/",
        api_key=None,
        model="voyage-large-2",
        optional_params={},
        litellm_params={},
    )
    url_with_endpoint = config.get_complete_url(
        api_base="https://api.voyageai.com/v1/embeddings/",
        api_key=None,
        model="voyage-large-2",
        optional_params={},
        litellm_params={},
    )
    url_default = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="voyage-large-2",
        optional_params={},
        litellm_params={},
    )
    assert url_without_slash == "https://api.voyageai.com/v1/embeddings"
    assert url_with_slash == "https://api.voyageai.com/v1/embeddings"
    assert url_with_endpoint == "https://api.voyageai.com/v1/embeddings"
    assert url_default == "https://api.voyageai.com/v1/embeddings"
    assert "//embeddings" not in url_with_slash

    contextual_config = VoyageContextualEmbeddingConfig()
    c_url = contextual_config.get_complete_url(
        api_base="https://api.voyageai.com/v1/",
        api_key=None,
        model="voyage-context-2",
        optional_params={},
        litellm_params={},
    )
    c_url_endpoint = contextual_config.get_complete_url(
        api_base="https://api.voyageai.com/v1/contextualizedembeddings/",
        api_key=None,
        model="voyage-context-2",
        optional_params={},
        litellm_params={},
    )
    c_url_default = contextual_config.get_complete_url(
        api_base=None,
        api_key=None,
        model="voyage-context-2",
        optional_params={},
        litellm_params={},
    )
    assert c_url == "https://api.voyageai.com/v1/contextualizedembeddings"
    assert c_url_endpoint == "https://api.voyageai.com/v1/contextualizedembeddings"
    assert c_url_default == "https://api.voyageai.com/v1/contextualizedembeddings"
    assert "//contextualizedembeddings" not in c_url

    multimodal_config = VoyageMultimodalEmbeddingConfig()
    m_url = multimodal_config.get_complete_url(
        api_base="https://api.voyageai.com/v1/",
        api_key=None,
        model="voyage-multimodal-3",
        optional_params={},
        litellm_params={},
    )
    m_url_endpoint = multimodal_config.get_complete_url(
        api_base="https://api.voyageai.com/v1/multimodalembeddings/",
        api_key=None,
        model="voyage-multimodal-3",
        optional_params={},
        litellm_params={},
    )
    m_url_default = multimodal_config.get_complete_url(
        api_base=None,
        api_key=None,
        model="voyage-multimodal-3",
        optional_params={},
        litellm_params={},
    )
    assert m_url == "https://api.voyageai.com/v1/multimodalembeddings"
    assert m_url_endpoint == "https://api.voyageai.com/v1/multimodalembeddings"
    assert m_url_default == "https://api.voyageai.com/v1/multimodalembeddings"
    assert "//multimodalembeddings" not in m_url


def test_perplexity_get_complete_url_trailing_slash():
    config = PerplexityEmbeddingConfig()
    url_without_slash = config.get_complete_url(
        api_base="https://api.perplexity.ai",
        api_key=None,
        model="sonar-medium",
        optional_params={},
        litellm_params={},
    )
    url_with_slash = config.get_complete_url(
        api_base="https://api.perplexity.ai/",
        api_key=None,
        model="sonar-medium",
        optional_params={},
        litellm_params={},
    )
    url_with_endpoint = config.get_complete_url(
        api_base="https://api.perplexity.ai/v1/embeddings/",
        api_key=None,
        model="sonar-medium",
        optional_params={},
        litellm_params={},
    )
    url_default = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="sonar-medium",
        optional_params={},
        litellm_params={},
    )
    assert url_without_slash == "https://api.perplexity.ai/v1/embeddings"
    assert url_with_slash == "https://api.perplexity.ai/v1/embeddings"
    assert url_with_endpoint == "https://api.perplexity.ai/v1/embeddings"
    assert url_default == "https://api.perplexity.ai/v1/embeddings"
    assert "//v1/embeddings" not in url_with_slash


def test_heroku_get_complete_url_trailing_slash():
    config = HerokuChatConfig()
    url_without_slash = config.get_complete_url(
        api_base="https://my-app.herokuapp.com",
        api_key=None,
        model="claude-3-5-sonnet",
        optional_params={},
        litellm_params={},
    )
    url_with_slash = config.get_complete_url(
        api_base="https://my-app.herokuapp.com/",
        api_key=None,
        model="claude-3-5-sonnet",
        optional_params={},
        litellm_params={},
    )
    url_with_endpoint = config.get_complete_url(
        api_base="https://my-app.herokuapp.com/v1/chat/completions/",
        api_key=None,
        model="claude-3-5-sonnet",
        optional_params={},
        litellm_params={},
    )
    assert url_without_slash == "https://my-app.herokuapp.com/v1/chat/completions"
    assert url_with_slash == "https://my-app.herokuapp.com/v1/chat/completions"
    assert url_with_endpoint == "https://my-app.herokuapp.com/v1/chat/completions"
    assert "//v1/chat/completions" not in url_with_slash


def test_topaz_get_complete_url_trailing_slash():
    config = TopazImageVariationConfig()
    url_without_slash = config.get_complete_url(
        api_base="https://api.topazlabs.com",
        api_key=None,
        model="topaz-v1",
        optional_params={},
        litellm_params={},
    )
    url_with_slash = config.get_complete_url(
        api_base="https://api.topazlabs.com/",
        api_key=None,
        model="topaz-v1",
        optional_params={},
        litellm_params={},
    )
    url_default = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="topaz-v1",
        optional_params={},
        litellm_params={},
    )
    assert url_without_slash == "https://api.topazlabs.com/image/v1/enhance"
    assert url_with_slash == "https://api.topazlabs.com/image/v1/enhance"
    assert url_default == "https://api.topazlabs.com/image/v1/enhance"
    assert "//image/v1/enhance" not in url_with_slash

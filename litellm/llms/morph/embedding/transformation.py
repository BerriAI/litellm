"""
Translate from OpenAI's `/v1/embeddings` to Morph's `/v1/embeddings`
"""

from ...openai_like.embedding.handler import OpenAILikeEmbeddingHandler


class MorphEmbeddingConfig(OpenAILikeEmbeddingHandler):
    pass

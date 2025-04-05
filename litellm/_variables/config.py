from litellm.llms.nvidia_nim.chat import NvidiaNimConfig
from litellm.llms.nvidia_nim.embed import NvidiaNimEmbeddingConfig
from litellm.llms.openai.chat.gpt_audio_transformation import OpenAIGPTAudioConfig
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig
from litellm.llms.vertex_ai.vertex_embeddings.transformation import (
    VertexAITextEmbeddingConfig,
)

vertexAITextEmbeddingConfig = VertexAITextEmbeddingConfig()
openaiOSeriesConfig = OpenAIOSeriesConfig()
openAIGPTConfig = OpenAIGPTConfig()
openAIGPTAudioConfig = OpenAIGPTAudioConfig()
nvidiaNimConfig = NvidiaNimConfig()
nvidiaNimEmbeddingConfig = NvidiaNimEmbeddingConfig()

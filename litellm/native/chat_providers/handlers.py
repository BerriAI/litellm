from typing import Final

from litellm.llms.anthropic.chat import AnthropicChatCompletion
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.chat.o_series_handler import AzureOpenAIO1ChatCompletion
from litellm.llms.azure.completion.handler import AzureTextCompletion
from litellm.llms.azure_ai.anthropic.handler import AzureAnthropicChatCompletion
from litellm.llms.bedrock.chat import BedrockConverseLLM
from litellm.llms.bytez.chat.transformation import BytezChatConfig
from litellm.llms.codestral.completion.handler import CodestralTextCompletion
from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.gdc.chat.transformation import GDCGeminiConfig
from litellm.llms.lemonade.chat.transformation import LemonadeChatConfig
from litellm.llms.openai.completion.handler import OpenAITextCompletion
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
from litellm.llms.ovhcloud.chat.transformation import OVHCloudChatConfig
from litellm.llms.predibase.chat.handler import PredibaseChatCompletion
from litellm.llms.sagemaker.completion.handler import SagemakerLLM
from litellm.llms.sap.chat.handler import GenAIHubOrchestration
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.llms.vertex_ai.vertex_ai_partner_models.main import VertexAIPartnerModels
from litellm.llms.vertex_ai.vertex_gemma_models.main import VertexAIGemmaModels
from litellm.llms.vertex_ai.vertex_model_garden.main import VertexAIModelGardenModels
from litellm.llms.watsonx.chat.handler import WatsonXChatHandler

openai_chat_completions = OpenAIChatCompletion()
openai_text_completions: Final = OpenAITextCompletion()
sap_gen_ai_hub_chat_completions: Final = GenAIHubOrchestration()
anthropic_chat_completions: Final = AnthropicChatCompletion()
azure_anthropic_chat_completions: Final = AzureAnthropicChatCompletion()
azure_chat_completions = AzureChatCompletion()
azure_o1_chat_completions: Final = AzureOpenAIO1ChatCompletion()
azure_text_completions: Final = AzureTextCompletion()
predibase_chat_completions: Final = PredibaseChatCompletion()
codestral_text_completions: Final = CodestralTextCompletion()
bedrock_converse_chat_completion: Final = BedrockConverseLLM()
vertex_chat_completion: Final = VertexLLM()
vertex_partner_models_chat_completion: Final = VertexAIPartnerModels()
vertex_gemma_chat_completion: Final = VertexAIGemmaModels()
vertex_model_garden_chat_completion: Final = VertexAIModelGardenModels()
gdc_transformation: Final = GDCGeminiConfig()
sagemaker_llm: Final = SagemakerLLM()
watsonx_chat_completion: Final = WatsonXChatHandler()
openai_like_chat_completion: Final = OpenAILikeChatHandler()
base_llm_http_handler = BaseLLMHTTPHandler()
base_llm_aiohttp_handler = BaseLLMAIOHTTPHandler()
bytez_transformation: Final = BytezChatConfig()
ovhcloud_transformation: Final = OVHCloudChatConfig()
lemonade_transformation: Final = LemonadeChatConfig()

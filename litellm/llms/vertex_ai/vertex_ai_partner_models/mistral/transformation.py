from litellm.llms.mistral.chat.transformation import MistralConfig


class VertexAIMistralConfig(MistralConfig):
    @property
    def custom_llm_provider(self) -> str:
        return "vertex_ai"

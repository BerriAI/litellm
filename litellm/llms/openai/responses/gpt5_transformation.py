from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig


class OpenAIGPT5ResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def get_supported_openai_params(self, model: str) -> list:
        base_params = super().get_supported_openai_params(model)
        gpt5_params = ["reasoning"]
        return base_params + gpt5_params
###### LiteLLM Integration with GPT Cache #########
import gptcache

# openai.ChatCompletion._llm_handler = litellm.completion
from gptcache.adapter import openai
import litellm


class LiteLLMChatCompletion(gptcache.adapter.openai.ChatCompletion):
    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        return litellm.completion(*llm_args, **llm_kwargs)


completion = LiteLLMChatCompletion.create
###### End of LiteLLM Integration with GPT Cache #########


# ####### Example usage ###############
# from gptcache import cache
# completion = LiteLLMChatCompletion.create
# # set API keys in .env / os.environ
# cache.init()
# cache.set_openai_key()
# result = completion(model="claude-2", messages=[{"role": "user", "content": "cto of litellm"}])
# print(result)

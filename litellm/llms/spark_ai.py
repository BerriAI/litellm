import time
import queue
from queue import Queue
import threading
from typing import Any, Sequence
from sparkai.llm.llm import ChatSparkLLM
from sparkai.core.messages import (
    ChatMessage,
    ChatMessageChunk
)
from sparkai.core.callbacks import BaseCallbackHandler
from sparkai.models.chat import ChatResponse
import litellm
from litellm.utils import ModelResponse
from litellm.utils import CustomStreamWrapper
from sparkai.core.outputs import LLMResult
class StreamerProcess(BaseCallbackHandler):
    """StreamerProcess Handler that prints to std out."""

    def __init__(self, result_q: Queue) -> None:
        """Initialize callback handler."""
        self.result_q = result_q

    def on_llm_new_token(self, token: str,
                         *,
                         chunk: None,
                         **kwargs: Any, ):
        self.result_q.put(token)
"""""
class SparkaiConfig:
    #初始化参数部分
    def __init__(self):
        self.client=ChatSparkLLM()

    def get_config(cls):

    def get_supported_openai_params(self,):

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
"""
def sparkai_config(api_key,streaming):
    split_result=api_key.split("&")
    return ChatSparkLLM(
        spark_app_id=split_result[0],
        spark_api_key=split_result[1],
        spark_api_secret=split_result[2],
        spark_api_url=split_result[4],
        spark_llm_domain=split_result[3],
        streaming=streaming,
    )

def get_sparkai_response(
        model=None,
        messages=None,
        api_key=None,
        model_response=None,
        streaming=False,
):
    user=sparkai_config(api_key,streaming)
    model_response = ModelResponse()
    convert_messages=[ChatMessage(role=messages[0]['role'],
                                  content=messages[0]['content'],
                                  )]
    if streaming == False:
        # 请求得到chat_result
        chat_result = user.generate(
            messages=[convert_messages],
            streaming=False
        )
        # 非流式
        response = chat_result.generations
        llm_out = chat_result.llm_output
        uu_id = chat_result.run[0].run_id
        model_response["choices"][0]["finish_reason"] = "stop"
        model_response["choices"][0]["message"]["role"] = response[0][0].message.type
        model_response["choices"][0]["message"]["content"] = response[0][0].message.content

        model_response["created"] = int(time.time())

        model_response["usage"]['prompt_tokens'] = llm_out['token_usage']["prompt_tokens"]
        model_response["usage"]['completion_tokens'] = llm_out['token_usage']["completion_tokens"]
        model_response["usage"]['total_tokens'] = llm_out['token_usage']["total_tokens"]

        model_response["model"] = model
        return model_response

def get_sparkai_stream(
        model=None,
        messages=None,
        api_key=None,
        model_response=None,
        streaming=True,
):
    user=sparkai_config(api_key,streaming)
    model_response = ModelResponse()
    convert_messages=[ChatMessage(role=messages[0]['role'],
                                  content=messages[0]['content'],
                                  )]
    result_q = Queue()
    handler = StreamerProcess(result_q)
    thr = threading.Thread(target=user.generate, args=([convert_messages], None, [handler]))
    thr.start()
    while thr.is_alive():
        try:
            delta = result_q.get(timeout=2)
            # tr=str+delta
        except queue.Empty:
            continue
        yield delta
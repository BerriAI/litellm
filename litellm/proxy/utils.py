import litellm
from litellm import ModelResponse
from proxy_server import llm_model_list
from typing import Optional

def track_cost_callback(
    kwargs,                                       # kwargs to completion
    completion_response: ModelResponse,           # response from completion
    start_time = None,
    end_time = None,                              # start/end time for completion
):
    try:
        # init logging config
        print("in custom callback tracking cost", llm_model_list)
        if "azure" in kwargs["model"]:
            # for azure cost tracking, we check the provided model list in the config.yaml
            # we need to map azure/chatgpt-deployment to -> azure/gpt-3.5-turbo
            pass
        # check if it has collected an entire stream response
        if "complete_streaming_response" in kwargs:
            # for tracking streaming cost we pass the "messages" and the output_text to litellm.completion_cost 
            completion_response=kwargs["complete_streaming_response"]
            input_text = kwargs["messages"]
            output_text = completion_response["choices"][0]["message"]["content"]
            response_cost = litellm.completion_cost(
                model = kwargs["model"],
                messages = input_text,
                completion=output_text
            )
            print("streaming response_cost", response_cost)
        # for non streaming responses
        else:
            # we pass the completion_response obj
            if kwargs["stream"] != True:
                response_cost = litellm.completion_cost(completion_response=completion_response)
                print("regular response_cost", response_cost)
    except:
        pass
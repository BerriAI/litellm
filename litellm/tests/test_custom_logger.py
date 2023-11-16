# ### What this tests ####
# import sys
# import os

# sys.path.insert(0, os.path.abspath('../..'))

# from litellm import completion, embedding
# import litellm
# from litellm.integrations.custom_logger import CustomLogger

# class MyCustomHandler(CustomLogger):
#     def log_pre_api_call(self, model, messages, kwargs): 
#         print(f"Pre-API Call")
    
#     def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
#         print(f"Post-API Call")
    
#     def log_stream_event(self, kwargs, response_obj, start_time, end_time):
#         print(f"On Stream")
        
#     def log_success_event(self, kwargs, response_obj, start_time, end_time): 
#         print(f"On Success")

#     def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
#         print(f"On Failure")

# customHandler = MyCustomHandler()

# def test_chat_openai():
#     try:
#         litellm.callbacks = [customHandler]
#         response = completion(model="gpt-3.5-turbo",
#                               messages=[{
#                                   "role": "user",
#                                   "content": "Hi 👋 - i'm openai"
#                               }],
#                               stream=True)
#         for chunk in response:
#             # print(chunk)
#             continue
#         # print(response)

#     except Exception as e:
#         print(e)
#         pass


# test_chat_openai()





# # def custom_callback(
# #         kwargs,
# #         completion_response,
# #         start_time,
# #         end_time,
# # ):
# #     print(
# #         "in custom callback func"
# #     )
# #     print("kwargs", kwargs)
# #     print(completion_response)
# #     print(start_time)
# #     print(end_time)
# #     if "complete_streaming_response" in kwargs:
# #         print("\n\n complete response\n\n")
# #         complete_streaming_response = kwargs["complete_streaming_response"]
# #         print(kwargs["complete_streaming_response"])
# #         usage = complete_streaming_response["usage"]
# #         print("usage", usage)
# # def send_slack_alert(
# #         kwargs,
# #         completion_response,
# #         start_time,
# #         end_time,
# # ):
# #     print(
# #         "in custom slack callback func"
# #     )
# #     import requests
# #     import json

# #     # Define the Slack webhook URL
# #     slack_webhook_url = os.environ['SLACK_WEBHOOK_URL']   # "https://hooks.slack.com/services/<>/<>/<>"

# #     # Define the text payload, send data available in litellm custom_callbacks
# #     text_payload = f"""LiteLLM Logging: kwargs: {str(kwargs)}\n\n, response: {str(completion_response)}\n\n, start time{str(start_time)} end time: {str(end_time)}
# #     """
# #     payload = {
# #         "text": text_payload
# #     }

# #     # Set the headers
# #     headers = {
# #         "Content-type": "application/json"
# #     }

# #     # Make the POST request
# #     response = requests.post(slack_webhook_url, json=payload, headers=headers)

# #     # Check the response status
# #     if response.status_code == 200:
# #         print("Message sent successfully to Slack!")
# #     else:
# #         print(f"Failed to send message to Slack. Status code: {response.status_code}")
# #         print(response.json())

# # def get_transformed_inputs(
# #     kwargs,
# # ):
# #     params_to_model = kwargs["additional_args"]["complete_input_dict"]
# #     print("params to model", params_to_model)

# # litellm.success_callback = [custom_callback, send_slack_alert]
# # litellm.failure_callback = [send_slack_alert]


# # litellm.set_verbose = False

# # # litellm.input_callback = [get_transformed_inputs]

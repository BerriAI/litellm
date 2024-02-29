# @pytest.mark.skip(reason="AWS Suspended Account")
# import sys, os
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os, io

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest
# import litellm
# from litellm import embedding, completion, completion_cost, Timeout
# from litellm import RateLimitError

# # litellm.num_retries = 3
# litellm.cache = None
# litellm.success_callback = []
# user_message = "Write a short poem about the sky"
# messages = [{"content": user_message, "role": "user"}]


# @pytest.fixture(autouse=True)
# def reset_callbacks():
#     print("\npytest fixture - resetting callbacks")
#     litellm.success_callback = []
#     litellm._async_success_callback = []
#     litellm.failure_callback = []
#     litellm.callbacks = []


# def test_completion_bedrock_claude_completion_auth():
#     print("calling bedrock claude completion params auth")
#     import os

#     aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
#     aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
#     aws_region_name = os.environ["AWS_REGION_NAME"]

#     os.environ.pop("AWS_ACCESS_KEY_ID", None)
#     os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
#     os.environ.pop("AWS_REGION_NAME", None)

#     try:
#         response = completion(
#             model="bedrock/anthropic.claude-instant-v1",
#             messages=messages,
#             max_tokens=10,
#             temperature=0.1,
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             aws_region_name=aws_region_name,
#         )
#         # Add any assertions here to check the response
#         print(response)

#         os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
#         os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
#         os.environ["AWS_REGION_NAME"] = aws_region_name
#     except RateLimitError:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# # test_completion_bedrock_claude_completion_auth()


# def test_completion_bedrock_claude_2_1_completion_auth():
#     print("calling bedrock claude 2.1 completion params auth")
#     import os

#     aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
#     aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
#     aws_region_name = os.environ["AWS_REGION_NAME"]

#     os.environ.pop("AWS_ACCESS_KEY_ID", None)
#     os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
#     os.environ.pop("AWS_REGION_NAME", None)
#     try:
#         response = completion(
#             model="bedrock/anthropic.claude-v2:1",
#             messages=messages,
#             max_tokens=10,
#             temperature=0.1,
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             aws_region_name=aws_region_name,
#         )
#         # Add any assertions here to check the response
#         print(response)

#         os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
#         os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
#         os.environ["AWS_REGION_NAME"] = aws_region_name
#     except RateLimitError:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# # test_completion_bedrock_claude_2_1_completion_auth()


# def test_completion_bedrock_claude_external_client_auth():
#     print("\ncalling bedrock claude external client auth")
#     import os

#     aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
#     aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
#     aws_region_name = os.environ["AWS_REGION_NAME"]

#     os.environ.pop("AWS_ACCESS_KEY_ID", None)
#     os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
#     os.environ.pop("AWS_REGION_NAME", None)

#     try:
#         import boto3

#         litellm.set_verbose = True

#         bedrock = boto3.client(
#             service_name="bedrock-runtime",
#             region_name=aws_region_name,
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             endpoint_url=f"https://bedrock-runtime.{aws_region_name}.amazonaws.com",
#         )

#         response = completion(
#             model="bedrock/anthropic.claude-instant-v1",
#             messages=messages,
#             max_tokens=10,
#             temperature=0.1,
#             aws_bedrock_client=bedrock,
#         )
#         # Add any assertions here to check the response
#         print(response)

#         os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
#         os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
#         os.environ["AWS_REGION_NAME"] = aws_region_name
#     except RateLimitError:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# # test_completion_bedrock_claude_external_client_auth()


# @pytest.mark.skip(reason="Expired token, need to renew")
# def test_completion_bedrock_claude_sts_client_auth():
#     print("\ncalling bedrock claude external client auth")
#     import os

#     aws_access_key_id = os.environ["AWS_TEMP_ACCESS_KEY_ID"]
#     aws_secret_access_key = os.environ["AWS_TEMP_SECRET_ACCESS_KEY"]
#     aws_region_name = os.environ["AWS_REGION_NAME"]
#     aws_role_name = os.environ["AWS_TEMP_ROLE_NAME"]

#     try:
#         import boto3

#         litellm.set_verbose = True

#         response = completion(
#             model="bedrock/anthropic.claude-instant-v1",
#             messages=messages,
#             max_tokens=10,
#             temperature=0.1,
#             aws_region_name=aws_region_name,
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             aws_role_name=aws_role_name,
#             aws_session_name="my-test-session",
#         )

#         response = embedding(
#             model="cohere.embed-multilingual-v3",
#             input=["hello world"],
#             aws_region_name="us-east-1",
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             aws_role_name=aws_role_name,
#             aws_session_name="my-test-session",
#         )

#         response = completion(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             aws_region_name="us-east-1",
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             aws_role_name=aws_role_name,
#             aws_session_name="my-test-session",
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except RateLimitError:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# # test_completion_bedrock_claude_sts_client_auth()


# def test_provisioned_throughput():
#     try:
#         litellm.set_verbose = True
#         import botocore, json, io
#         import botocore.session
#         from botocore.stub import Stubber

#         bedrock_client = botocore.session.get_session().create_client(
#             "bedrock-runtime", region_name="us-east-1"
#         )

#         expected_params = {
#             "accept": "application/json",
#             "body": '{"prompt": "\\n\\nHuman: Hello, how are you?\\n\\nAssistant: ", '
#             '"max_tokens_to_sample": 256}',
#             "contentType": "application/json",
#             "modelId": "provisioned-model-arn",
#         }
#         response_from_bedrock = {
#             "body": io.StringIO(
#                 json.dumps(
#                     {
#                         "completion": " Here is a short poem about the sky:",
#                         "stop_reason": "max_tokens",
#                         "stop": None,
#                     }
#                 )
#             ),
#             "contentType": "contentType",
#             "ResponseMetadata": {"HTTPStatusCode": 200},
#         }

#         with Stubber(bedrock_client) as stubber:
#             stubber.add_response(
#                 "invoke_model",
#                 service_response=response_from_bedrock,
#                 expected_params=expected_params,
#             )
#             response = litellm.completion(
#                 model="bedrock/anthropic.claude-instant-v1",
#                 model_id="provisioned-model-arn",
#                 messages=[{"content": "Hello, how are you?", "role": "user"}],
#                 aws_bedrock_client=bedrock_client,
#             )
#             print("response stubbed", response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# # test_provisioned_throughput()

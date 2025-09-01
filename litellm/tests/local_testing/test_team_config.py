# #### What this tests ####
# #    This tests if setting team_config actually works
# import sys, os
# import traceback
# import pytest

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm
# from litellm.proxy.proxy_server import ProxyConfig


# @pytest.mark.asyncio
# async def test_team_config():
#     litellm.default_team_settings = [
#         {
#             "team_id": "my-special-team",
#             "success_callback": ["langfuse"],
#             "langfuse_public_key": "os.environ/LANGFUSE_PUB_KEY_2",
#             "langfuse_secret": "os.environ/LANGFUSE_PRIVATE_KEY_2",
#         }
#     ]
#     proxyconfig = ProxyConfig()

#     team_config = await proxyconfig.load_team_config(team_id="my-special-team")
#     assert len(team_config) > 0

#     data = {
#         "model": "gpt-3.5-turbo",
#         "messages": [{"role": "user", "content": "Hey, how's it going?"}],
#     }
#     team_config.pop("team_id")
#     response = litellm.completion(**{**data, **team_config})

#     print(f"response: {response}")

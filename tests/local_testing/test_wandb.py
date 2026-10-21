import asyncio
import os

import litellm

# import logging
# logging.basicConfig(level=logging.DEBUG)

litellm.num_retries = 3
litellm.success_callback = ["wandb"]


def test_wandb_logging_async():
    try:
        litellm.set_verbose = False

        async def _test_langfuse():
            from litellm import Router

            model_list = [
                {  # list of model deployments
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {  # params for litellm completion/embedding call
                        "model": "gpt-3.5-turbo",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                }
            ]

            router = Router(model_list=model_list)

            # openai.ChatCompletion.create replacement
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "this is a test with litellm router ?"}
                ],
            )
            print(response)

        response = asyncio.run(_test_langfuse())
        print(f"response: {response}")
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pass




# test_wandb_logging()

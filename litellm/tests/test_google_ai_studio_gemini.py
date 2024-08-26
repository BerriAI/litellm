import os
import sys
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from dotenv import load_dotenv

import litellm


def generate_text():
    try:
        litellm.set_verbose = True
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://avatars.githubusercontent.com/u/17561003?v=4"
                        },
                    },
                ],
            }
        ]
        response = litellm.completion(
            model="gemini/gemini-pro-vision",
            messages=messages,
            stop="Hello world",
            num_retries=3,
        )
        print(response)
        assert isinstance(response.choices[0].message.content, str) == True
    except Exception as exception:
        raise Exception("An error occurred during text generation:", exception)


# generate_text()


# def test_fine_tuned_model():
#     litellm.set_verbose = True
#     response = litellm.completion(
#         model="gemini/tunedModels/raynarev1short-gefpulfdjpqj",
#         messages=[{"content": "Hello, how are you?", "role": "user"}],
#     )
#     print(response)
#     pass

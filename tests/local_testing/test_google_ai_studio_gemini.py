import os, sys, traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from dotenv import load_dotenv


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

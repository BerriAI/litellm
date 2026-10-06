import os
import openai
import tempfile


client = openai.OpenAI(base_url="http://0.0.0.0:4000/openai", api_key=os.environ["LITELLM_MASTER_KEY"])


def test_pass_through_file_operations():
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".txt", delete=False
    ) as temp_file:
        temp_file.write("This is a test file for the OpenAI Assistants API.")
        temp_file.flush()

        file = client.files.create(
            file=open(temp_file.name, "rb"),
            purpose="assistants",
        )
        print("file created", file)

        delete_file = client.files.delete(file.id)
        print("file deleted", delete_file)

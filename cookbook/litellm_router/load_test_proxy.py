import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm import Router
import litellm

litellm.set_verbose = False
os.environ.pop("AZURE_AD_TOKEN")

model_list = [
    {  # list of model deployments
        "model_name": "gpt-3.5-turbo",  # model alias
        "litellm_params": {  # params for litellm completion/embedding call
            "model": "azure/chatgpt-v-2",  # actual model name
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_version": os.getenv("AZURE_API_VERSION"),
            "api_base": os.getenv("AZURE_API_BASE"),
        },
    },
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {  # params for litellm completion/embedding call
            "model": "azure/chatgpt-functioncalling",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_version": os.getenv("AZURE_API_VERSION"),
            "api_base": os.getenv("AZURE_API_BASE"),
        },
    },
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {  # params for litellm completion/embedding call
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
]
router = Router(model_list=model_list)


file_paths = [
    "test_questions/question1.txt",
    "test_questions/question2.txt",
    "test_questions/question3.txt",
]
questions = []

for file_path in file_paths:
    try:
        print(file_path)
        with open(file_path, "r") as file:
            content = file.read()
            questions.append(content)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# for q in questions:
#     print(q)


# make X concurrent calls to litellm.completion(model=gpt-35-turbo, messages=[]), pick a random question in questions array.
#  Allow me to tune X concurrent calls.. Log question, output/exception, response time somewhere
# show me a summary of requests made, success full calls, failed calls. For failed calls show me the exceptions

import concurrent.futures
import random
import time


# Function to make concurrent calls to OpenAI API
def make_openai_completion(question):
    try:
        start_time = time.time()
        import openai

        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], base_url="http://0.0.0.0:8000"
        )  # base_url="http://0.0.0.0:8000",
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Answer this question{question}",
                }
            ],
        )
        print(response)
        end_time = time.time()

        # Log the request details
        with open("request_log.txt", "a") as log_file:
            log_file.write(
                f"Question: {question[:100]}\nResponse ID:{response.id} Content:{response.choices[0].message.content[:10]}\nTime: {end_time - start_time:.2f} seconds\n\n"
            )

        return response
    except Exception as e:
        # Log exceptions for failed calls
        with open("error_log.txt", "a") as error_log_file:
            error_log_file.write(f"Question: {question[:100]}\nException: {str(e)}\n\n")
        return None


# Number of concurrent calls (you can adjust this)
concurrent_calls = 100

# List to store the futures of concurrent calls
futures = []

# Make concurrent calls
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_calls) as executor:
    for _ in range(concurrent_calls):
        random_question = random.choice(questions)
        futures.append(executor.submit(make_openai_completion, random_question))

# Wait for all futures to complete
concurrent.futures.wait(futures)

# Summarize the results
successful_calls = 0
failed_calls = 0

for future in futures:
    if future.result() is not None:
        successful_calls += 1
    else:
        failed_calls += 1

print("Load test Summary:")
print(f"Total Requests: {concurrent_calls}")
print(f"Successful Calls: {successful_calls}")
print(f"Failed Calls: {failed_calls}")

# Display content of the logs
with open("request_log.txt", "r") as log_file:
    print("\nRequest Log:\n", log_file.read())

with open("error_log.txt", "r") as error_log_file:
    print("\nError Log:\n", error_log_file.read())

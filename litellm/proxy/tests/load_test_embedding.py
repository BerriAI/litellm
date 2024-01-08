# test time it takes to make 100 concurrent embedding requests to OpenaI

import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest


import litellm

litellm.set_verbose = False


question = "embed this very long text" * 100


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
            api_key=os.environ["OPENAI_API_KEY"]
        )  # base_url="http://0.0.0.0:8000",
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[question],
        )
        print(response)
        end_time = time.time()

        # Log the request details
        # with open("request_log.txt", "a") as log_file:
        #     log_file.write(
        #         f"Question: {question[:100]}\nResponse ID:{response.id} Content:{response.choices[0].message.content[:10]}\nTime: {end_time - start_time:.2f} seconds\n\n"
        #     )

        return response
    except Exception as e:
        # Log exceptions for failed calls
        # with open("error_log.txt", "a") as error_log_file:
        #     error_log_file.write(
        #         f"\nException: {str(e)}\n\n"
        #     )
        return None


start_time = time.time()
# Number of concurrent calls (you can adjust this)
concurrent_calls = 500

# List to store the futures of concurrent calls
futures = []

# Make concurrent calls
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_calls) as executor:
    for _ in range(concurrent_calls):
        futures.append(executor.submit(make_openai_completion, question))

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

end_time = time.time()
# Calculate the duration
duration = end_time - start_time

print(f"Load test Summary:")
print(f"Total Requests: {concurrent_calls}")
print(f"Successful Calls: {successful_calls}")
print(f"Failed Calls: {failed_calls}")
print(f"Total Time: {duration:.2f} seconds")

# Display content of the logs
with open("request_log.txt", "r") as log_file:
    print("\nRequest Log:\n", log_file.read())

with open("error_log.txt", "r") as error_log_file:
    print("\nError Log:\n", error_log_file.read())

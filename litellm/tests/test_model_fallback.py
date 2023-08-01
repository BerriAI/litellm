import sys, os
import traceback
sys.path.append('..')  # Adds the parent directory to the system path
import main
from main import embedding, completion
main.success_callback = ["posthog"]
main.failure_callback = ["slack", "sentry", "posthog"]

main.set_verbose = True

model_fallback_list = ["replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1", "claude-instant-1", "gpt-3.5-turbo"]

user_message = "Hello, how are you?"
messages = [{ "content": user_message,"role": "user"}]

# for _ in range(10):
for model in model_fallback_list:
    try:
        response = completion(model=model, messages=messages)
        print(response)
        if response != None:
            break
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}") 
        raise e

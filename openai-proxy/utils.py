import os, litellm
import dotenv
dotenv.load_dotenv() # load env variables

def set_callbacks():
    if ("LANGFUSE_PUBLIC_KEY" in os.environ and "LANGFUSE_SECRET_KEY" in os.environ) or "LANGFUSE_HOST" in os.environ: 
        litellm.success_callback = ["langfuse"] 

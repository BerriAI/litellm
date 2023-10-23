import os, litellm
import dotenv
dotenv.load_dotenv() # load env variables

def set_callbacks():
    if ("LANGUFSE_PUBLIC_KEY" in os.environ and "LANGUFSE_SECRET_KEY" in os.environ) or "LANGFUSE_HOST" in os.environ: 
        litellm.success_callback = ["langfuse"] 

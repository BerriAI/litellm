import os, litellm
import dotenv
dotenv.load_dotenv() # load env variables

def set_callbacks():
    ## LOGGING
    if len(os.getenv("SET_VERBOSE", "")) > 0: 
        if os.getenv("SET_VERBOSE") == "True": 
            litellm.set_verbose = True
        else: 
            litellm.set_verbose = False

    ### LANGFUSE
    if (len(os.getenv("LANGFUSE_PUBLIC_KEY", "")) > 0 and len(os.getenv("LANGFUSE_SECRET_KEY", ""))) > 0 or len(os.getenv("LANGFUSE_HOST", "")) > 0:
        litellm.success_callback = ["langfuse"] 
    
    ## CACHING 
    ### REDIS
    if len(os.getenv("REDIS_HOST", "")) >  0 and len(os.getenv("REDIS_PORT", "")) > 0 and len(os.getenv("REDIS_PASSWORD", "")) > 0: 
        from litellm.caching import Cache
        litellm.cache = Cache(type="redis", host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), password=os.getenv("REDIS_PASSWORD"))



    

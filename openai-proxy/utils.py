import os, litellm
import dotenv
dotenv.load_dotenv() # load env variables

def set_callbacks():
    ## LOGGING
    ### LANGFUSE
    if (len(os.getenv("LANGFUSE_PUBLIC_KEY", "")) > 0 and len(os.getenv("LANGFUSE_SECRET_KEY", ""))) > 0 or len(os.getenv("LANGFUSE_HOST", "")) > 0:
        print(f"sets langfuse integration")
        litellm.success_callback = ["langfuse"] 
    
    ## CACHING 
    ### REDIS
    print(f"redis host: {len(os.getenv('REDIS_HOST', ''))}; redis port: {len(os.getenv('REDIS_PORT', ''))}; redis password: {len(os.getenv('REDIS_PASSWORD'))}")
    if len(os.getenv("REDIS_HOST", "")) >  0 and len(os.getenv("REDIS_PORT", "")) > 0 and len(os.getenv("REDIS_PASSWORD", "")) > 0: 
        print(f"sets caching integration")
        from litellm.caching import Cache
        litellm.cache = Cache(type="redis", host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), password=os.getenv("REDIS_PASSWORD"))



    

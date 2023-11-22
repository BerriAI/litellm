from dotenv import load_dotenv
load_dotenv() 
import json
import redis
from celery import Celery
import time
import sys, os
sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path - for litellm local dev
import litellm

# Redis connection setup
pool = redis.ConnectionPool(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), password=os.getenv("REDIS_PASSWORD"), db=0, max_connections=10)
redis_client = redis.Redis(connection_pool=pool)

# Celery setup
celery_app = Celery('tasks', broker=f"redis://default:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}", backend=f"redis://default:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}")
celery_app.conf.update(
    broker_pool_limit = None,
    broker_transport_options = {'connection_pool': pool},
    result_backend_transport_options = {'connection_pool': pool},
)


# Celery task
@celery_app.task(name='process_job')
def process_job(*args, **kwargs):
    try: 
        llm_router: litellm.Router = litellm.Router(model_list=kwargs.pop("llm_model_list"))
        response = llm_router.completion(*args, **kwargs)
        if isinstance(response, litellm.ModelResponse): 
            response = response.model_dump_json()
            return json.loads(response)
        return str(response)
    except Exception as e: 
        print(e)
        raise e
    
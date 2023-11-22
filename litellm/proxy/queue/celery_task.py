from dotenv import load_dotenv
load_dotenv() 

import sys, os
sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path - for litellm local dev
import litellm
from litellm.proxy.queue.celery_app import celery_app

# Celery task
@celery_app.task(name='process_job')
def process_job(*args, **kwargs):
    llm_router: litellm.Router = litellm.Router(model_list=kwargs.pop("llm_model_list"))
    return llm_router.completion(*args, **kwargs)
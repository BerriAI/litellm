# from dotenv import load_dotenv

# load_dotenv()
# import json, subprocess
# import psutil  # Import the psutil library
# import atexit

# try:
#     ### OPTIONAL DEPENDENCIES ###  - pip install redis and celery only when a user opts into using the async endpoints which require both
#     from celery import Celery
#     import redis
# except:
#     import sys

#     subprocess.check_call([sys.executable, "-m", "pip", "install", "redis", "celery"])

# import time
# import sys, os

# sys.path.insert(
#     0, os.path.abspath("../../..")
# )  # Adds the parent directory to the system path - for litellm local dev
# import litellm

# # Redis connection setup
# pool = redis.ConnectionPool(
#     host=os.getenv("REDIS_HOST"),
#     port=os.getenv("REDIS_PORT"),
#     password=os.getenv("REDIS_PASSWORD"),
#     db=0,
#     max_connections=5,
# )
# redis_client = redis.Redis(connection_pool=pool)

# # Celery setup
# celery_app = Celery(
#     "tasks",
#     broker=f"redis://default:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}",
#     backend=f"redis://default:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}",
# )
# celery_app.conf.update(
#     broker_pool_limit=None,
#     broker_transport_options={"connection_pool": pool},
#     result_backend_transport_options={"connection_pool": pool},
# )


# # Celery task
# @celery_app.task(name="process_job", max_retries=3)
# def process_job(*args, **kwargs):
#     try:
#         llm_router: litellm.Router = litellm.Router(model_list=kwargs.pop("llm_model_list"))  # type: ignore
#         response = llm_router.completion(*args, **kwargs)  # type: ignore
#         if isinstance(response, litellm.ModelResponse):
#             response = response.model_dump_json()
#             return json.loads(response)
#         return str(response)
#     except Exception as e:
#         raise e


# # Ensure Celery workers are terminated when the script exits
# def cleanup():
#     try:
#         # Get a list of all running processes
#         for process in psutil.process_iter(attrs=["pid", "name"]):
#             # Check if the process is a Celery worker process
#             if process.info["name"] == "celery":
#                 print(f"Terminating Celery worker with PID {process.info['pid']}")
#                 # Terminate the Celery worker process
#                 psutil.Process(process.info["pid"]).terminate()
#     except Exception as e:
#         print(f"Error during cleanup: {e}")


# # Register the cleanup function to run when the script exits
# atexit.register(cleanup)

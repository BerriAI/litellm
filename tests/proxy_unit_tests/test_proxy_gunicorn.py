# #### What this tests ####
# #  Allow the user to easily run the local proxy server with Gunicorn
# # LOCAL TESTING ONLY
# import sys, os, subprocess
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os, io

# # this file is to test litellm/proxy

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest
# import litellm

# ### LOCAL Proxy Server INIT ###
# from litellm.proxy.proxy_server import save_worker_config  # Replace with the actual module where your FastAPI router is defined
# filepath = os.path.dirname(os.path.abspath(__file__))
# config_fp = f"{filepath}/test_configs/test_config_custom_auth.yaml"
# def get_openai_info():
#     return {
#         "api_key": os.getenv("AZURE_API_KEY"),
#         "api_base": os.getenv("AZURE_API_BASE"),
#     }

# def run_server(host="0.0.0.0",port=8008,num_workers=None):
#     if num_workers is None:
#         # Set it to min(8,cpu_count())
#         import multiprocessing
#         num_workers = min(4,multiprocessing.cpu_count())

#     ### LOAD KEYS ###

#     # Load the Azure keys. For now get them from openai-usage
#     azure_info = get_openai_info()
#     print(f"Azure info:{azure_info}")
#     os.environ["AZURE_API_KEY"] = azure_info['api_key']
#     os.environ["AZURE_API_BASE"] = azure_info['api_base']
#     os.environ["AZURE_API_VERSION"] = "2023-09-01-preview"

#     ### SAVE CONFIG ###

#     os.environ["WORKER_CONFIG"] = config_fp

#     # In order for the app to behave well with signals, run it with gunicorn
#     # The first argument must be the "name of the command run"
#     cmd = f"gunicorn litellm.proxy.proxy_server:app --workers {num_workers} --worker-class uvicorn.workers.UvicornWorker --bind {host}:{port}"
#     cmd = cmd.split()
#     print(f"Running command: {cmd}")
#     import sys
#     sys.stdout.flush()
#     sys.stderr.flush()

#     # Make sure to propage env variables
#     subprocess.run(cmd)  # This line actually starts Gunicorn

# if __name__ == "__main__":
#     run_server()

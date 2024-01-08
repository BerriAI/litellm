# import os, litellm
# import pkg_resources
# import dotenv
# dotenv.load_dotenv() # load env variables

# def print_verbose(print_statement):
#     pass

# def get_package_version(package_name):
#     try:
#         package = pkg_resources.get_distribution(package_name)
#         return package.version
#     except pkg_resources.DistributionNotFound:
#         return None

# # Usage example
# package_name = "litellm"
# version = get_package_version(package_name)
# if version:
#     print_verbose(f"The version of {package_name} is {version}")
# else:
#     print_verbose(f"{package_name} is not installed")
# import yaml
# import dotenv
# from typing import Optional
# dotenv.load_dotenv() # load env variables

# def set_callbacks():
#     ## LOGGING
#     if len(os.getenv("SET_VERBOSE", "")) > 0:
#         if os.getenv("SET_VERBOSE") == "True":
#             litellm.set_verbose = True
#             print_verbose("\033[92mLiteLLM: Switched on verbose logging\033[0m")
#         else:
#             litellm.set_verbose = False

#     ### LANGFUSE
#     if (len(os.getenv("LANGFUSE_PUBLIC_KEY", "")) > 0 and len(os.getenv("LANGFUSE_SECRET_KEY", ""))) > 0 or len(os.getenv("LANGFUSE_HOST", "")) > 0:
#         litellm.success_callback = ["langfuse"]
#         print_verbose("\033[92mLiteLLM: Switched on Langfuse feature\033[0m")

#     ## CACHING
#     ### REDIS
#     # if len(os.getenv("REDIS_HOST", "")) >  0 and len(os.getenv("REDIS_PORT", "")) > 0 and len(os.getenv("REDIS_PASSWORD", "")) > 0:
#     #     print(f"redis host: {os.getenv('REDIS_HOST')}; redis port: {os.getenv('REDIS_PORT')}; password: {os.getenv('REDIS_PASSWORD')}")
#     #     from litellm.caching import Cache
#     #     litellm.cache = Cache(type="redis", host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), password=os.getenv("REDIS_PASSWORD"))
#     #     print("\033[92mLiteLLM: Switched on Redis caching\033[0m")


# def load_router_config(router: Optional[litellm.Router], config_file_path: Optional[str]='/app/config.yaml'):
#     config = {}
#     server_settings  = {}
#     try:
#         if os.path.exists(config_file_path): # type: ignore
#             with open(config_file_path, 'r') as file: # type: ignore
#                 config = yaml.safe_load(file)
#         else:
#             pass
#     except:
#         pass

#     ## SERVER SETTINGS (e.g. default completion model = 'ollama/mistral')
#     server_settings = config.get("server_settings", None)
#     if server_settings:
#         server_settings = server_settings

#     ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
#     litellm_settings = config.get('litellm_settings', None)
#     if litellm_settings:
#         for key, value in litellm_settings.items():
#             setattr(litellm, key, value)

#     ## MODEL LIST
#     model_list = config.get('model_list', None)
#     if model_list:
#         router = litellm.Router(model_list=model_list)

#     ## ENVIRONMENT VARIABLES
#     environment_variables = config.get('environment_variables', None)
#     if environment_variables:
#         for key, value in environment_variables.items():
#             os.environ[key] = value

#     return router, model_list, server_settings

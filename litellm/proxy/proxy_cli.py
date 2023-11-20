import click
import subprocess, traceback, json
import os, sys
import random, appdirs
from datetime import datetime
from dotenv import load_dotenv
import operator
sys.path.append(os.getcwd())

config_filename = "litellm.secrets"
# Using appdirs to determine user-specific config path
config_dir = appdirs.user_config_dir("litellm")
user_config_path = os.getenv("LITELLM_CONFIG_PATH", os.path.join(config_dir, config_filename))

load_dotenv()
from importlib import resources
import shutil
telemetry = None

def run_ollama_serve():
    command = ['ollama', 'serve']
    
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(command, stdout=devnull, stderr=devnull)

def clone_subfolder(repo_url, subfolder, destination):
  # Clone the full repo
  repo_name = repo_url.split('/')[-1]  
  repo_master = os.path.join(destination, "repo_master")
  subprocess.run(['git', 'clone', repo_url, repo_master])

  # Move into the subfolder 
  subfolder_path = os.path.join(repo_master, subfolder)

  # Copy subfolder to destination
  for file_name in os.listdir(subfolder_path):
    source = os.path.join(subfolder_path, file_name)
    if os.path.isfile(source):
        shutil.copy(source, destination)
    else:
        dest_path = os.path.join(destination, file_name)
        shutil.copytree(source, dest_path)

  # Remove cloned repo folder
  subprocess.run(['rm', '-rf', os.path.join(destination, "repo_master")])
  feature_telemetry(feature="create-proxy")

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

@click.command()
@click.option('--host', default='0.0.0.0', help='Host for the server to listen on.')
@click.option('--port', default=8000, help='Port to bind the server to.')
@click.option('--num_workers', default=1, help='Number of uvicorn workers to spin up')
@click.option('--api_base', default=None, help='API base URL.')
@click.option('--api_version', default="2023-07-01-preview", help='For azure - pass in the api version.')
@click.option('--model', '-m', default=None, help='The model name to pass to litellm expects') 
@click.option('--alias', default=None, help='The alias for the model - use this to give a litellm model name (e.g. "huggingface/codellama/CodeLlama-7b-Instruct-hf") a more user-friendly name ("codellama")') 
@click.option('--add_key', default=None, help='The model name to pass to litellm expects') 
@click.option('--headers', default=None, help='headers for the API call') 
@click.option('--save', is_flag=True, type=bool, help='Save the model-specific config')
@click.option('--debug', default=False, is_flag=True, type=bool, help='To debug the input') 
@click.option('--temperature', default=None, type=float, help='Set temperature for the model') 
@click.option('--max_tokens', default=None, type=int, help='Set max tokens for the model') 
@click.option('--request_timeout', default=600, type=int, help='Set timeout in seconds for completion calls') 
@click.option('--drop_params', is_flag=True, help='Drop any unmapped params') 
@click.option('--add_function_to_prompt', is_flag=True, help='If function passed but unsupported, pass it as prompt') 
@click.option('--config', '-c', default=None, help='Configure Litellm')  
@click.option('--file', '-f', help='Path to config file')
@click.option('--max_budget', default=None, type=float, help='Set max budget for API calls - works for hosted models like OpenAI, TogetherAI, Anthropic, etc.`') 
@click.option('--telemetry', default=True, type=bool, help='Helps us know if people are using this feature. Turn this off by doing `--telemetry False`') 
@click.option('--logs', flag_value=False, type=int, help='Gets the "n" most recent logs. By default gets most recent log.') 
@click.option('--test', flag_value=True, help='proxy chat completions url to make a test request to')
@click.option('--local', is_flag=True, default=False, help='for local debugging')
def run_server(host, port, api_base, api_version, model, alias, add_key, headers, save, debug, temperature, max_tokens, request_timeout, drop_params, add_function_to_prompt, config, file, max_budget, telemetry, logs, test, local, num_workers):
    global feature_telemetry
    args = locals()
    if local:
        from proxy_server import app, save_worker_config, usage_telemetry
    else:
        try:
            from .proxy_server import app, save_worker_config, usage_telemetry
        except ImportError as e: 
            from proxy_server import app, save_worker_config, usage_telemetry
    feature_telemetry = usage_telemetry
    if logs is not None:
        if logs == 0: # default to 1
            logs = 1
        try:
            with open('api_log.json') as f:
                data = json.load(f)

            # convert keys to datetime objects    
            log_times = {datetime.strptime(k, "%Y%m%d%H%M%S%f"): v for k, v in data.items()}

            # sort by timestamp    
            sorted_times = sorted(log_times.items(), key=operator.itemgetter(0), reverse=True)

            # get n recent logs
            recent_logs = {k.strftime("%Y%m%d%H%M%S%f"): v for k, v in sorted_times[:logs]}

            print(json.dumps(recent_logs, indent=4))
        except:
            print("LiteLLM: No logs saved!")
        return
    if model and "ollama" in model: 
        run_ollama_serve()
    if test != False:
        click.echo('\nLiteLLM: Making a test ChatCompletions request to your proxy')
        import openai
        if test == True: # flag value set
            api_base = f"http://{host}:{port}"
        else: 
            api_base = test
        client = openai.OpenAI(
            api_key="My API Key",
            base_url=api_base
        )

        response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
            {
                "role": "user",
                "content": "this is a test request, write a short poem"
            }
        ])
        click.echo(f'\nLiteLLM: response from proxy {response}')

        print("\n Making streaming request to proxy")

        response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
            {
                "role": "user",
                "content": "this is a test request, write a short poem"
            }
        ],
        stream=True,
        )
        for chunk in response:
            click.echo(f'LiteLLM: streaming response from proxy {chunk}')
        print("\n making completion request to proxy")
        response = client.completions.create(model="gpt-3.5-turbo", prompt='this is a test request, write a short poem')
        print(response)

        return
    else:
        if headers:
            headers = json.loads(headers)
        save_worker_config(model=model, alias=alias, api_base=api_base, api_version=api_version, debug=debug, temperature=temperature, max_tokens=max_tokens, request_timeout=request_timeout, max_budget=max_budget, telemetry=telemetry, drop_params=drop_params, add_function_to_prompt=add_function_to_prompt, headers=headers, save=save, config=config)
        try:
            import uvicorn
        except:
            raise ImportError("Uvicorn needs to be imported. Run - `pip install uvicorn`")
        if port == 8000 and is_port_in_use(port):
            port = random.randint(1024, 49152)
        uvicorn.run("litellm.proxy.proxy_server:app", host=host, port=port, workers=num_workers)


if __name__ == "__main__":
    run_server()

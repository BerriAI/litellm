import click
import subprocess, traceback, json
import os, sys
import random, appdirs
from datetime import datetime
from dotenv import load_dotenv
import operator

config_filename = "litellm.secrets.toml"
# Using appdirs to determine user-specific config path
config_dir = appdirs.user_config_dir("litellm")
user_config_path = os.path.join(config_dir, config_filename)

load_dotenv()
from importlib import resources
import shutil
telemetry = None

def run_ollama_serve():
    command = ['ollama', 'serve']
    
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(command, stdout=devnull, stderr=devnull)

def open_config(file_path=None):
    # Create the .env file if it doesn't exist
    if file_path: 
        # Ensure the user-specific directory exists
        os.makedirs(config_dir, exist_ok=True)
        # Copying the file using shutil.copy
        try:
            shutil.copy(file_path, user_config_path)
            with open(file_path) as f:
                print(f"Source file: {file_path}")
                print(f.read())

            with open(user_config_path) as f:
                print(f"Dest file: {user_config_path}")
                print(f.read())
            print("\033[1;32mDone successfully\033[0m")
        except Exception as e:
            print(f"Failed to copy {file_path}: {e}")
    else: 
        if os.path.exists(user_config_path):
            if os.path.getsize(user_config_path) == 0:
                print(f"{user_config_path} exists but is empty")
                print(f"To create a config (save keys, modify model prompt), copy the template located here: https://docs.litellm.ai/docs/proxy_server")
            else: 
                with open(user_config_path) as f:
                    print(f"Saved Config file: {user_config_path}")
                    print(f.read())
        else:
            print(f"{user_config_path} hasn't been created yet.")
            print(f"To create a config (save keys, modify model prompt), copy the template located here: https://docs.litellm.ai/docs/proxy_server")
    print(f"LiteLLM: config location - {user_config_path}")


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
@click.option('--drop_params', is_flag=True, help='Drop any unmapped params') 
@click.option('--create_proxy', is_flag=True, help='Creates a local OpenAI-compatible server template') 
@click.option('--add_function_to_prompt', is_flag=True, help='If function passed but unsupported, pass it as prompt') 
@click.option('--config', '-c', is_flag=True, help='Configure Litellm')  
@click.option('--file', '-f', help='Path to config file')
@click.option('--max_budget', default=None, type=float, help='Set max budget for API calls - works for hosted models like OpenAI, TogetherAI, Anthropic, etc.`') 
@click.option('--telemetry', default=True, type=bool, help='Helps us know if people are using this feature. Turn this off by doing `--telemetry False`') 
@click.option('--logs', flag_value=False, type=int, help='Gets the "n" most recent logs. By default gets most recent log.') 
@click.option('--test', flag_value=True, help='proxy chat completions url to make a test request to')
@click.option('--local', is_flag=True, default=False, help='for local debugging')
@click.option('--cost', is_flag=True, default=False, help='for viewing cost logs')
def run_server(host, port, api_base, api_version, model, alias, add_key, headers, save, debug, temperature, max_tokens, drop_params, create_proxy, add_function_to_prompt, config, file, max_budget, telemetry, logs, test, local, cost):
    global feature_telemetry
    args = locals()
    if local:
        from proxy_server import app, initialize, print_cost_logs, usage_telemetry, add_keys_to_config
        debug = True
    else:
        try:
            from .proxy_server import app, initialize, print_cost_logs, usage_telemetry, add_keys_to_config
        except ImportError as e: 
            from proxy_server import app, initialize, print_cost_logs, usage_telemetry, add_keys_to_config
    feature_telemetry = usage_telemetry
    if create_proxy == True: 
        repo_url = 'https://github.com/BerriAI/litellm'
        subfolder = 'litellm/proxy' 
        destination = os.path.join(os.getcwd(), 'litellm-proxy')

        clone_subfolder(repo_url, subfolder, destination)
        return
    if config:
        if file: 
            open_config(file_path=file)
        else: 
            open_config()
        return
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
    if add_key:
        key_name, key_value = add_key.split("=")
        add_keys_to_config(key_name, key_value)
        with open(user_config_path) as f:
            print(f.read())
        print("\033[1;32mDone successfully\033[0m")
        return
    if model and "ollama" in model: 
        print(f"ollama called")
        run_ollama_serve()
    if cost == True:
        print_cost_logs()
        return
    if test != False:
        click.echo('LiteLLM: Making a test ChatCompletions request to your proxy')
        import openai
        if test == True: # flag value set
            api_base = f"http://{host}:{port}"
        else: 
            api_base = test
        openai.api_base = api_base
        openai.api_key = "temp-key"
        print(openai.api_base)

        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
            {
                "role": "user",
                "content": "this is a test request, acknowledge that you got it"
            }
        ])
        click.echo(f'LiteLLM: response from proxy {response}')

        click.echo(f'LiteLLM: response from proxy with streaming {response}')
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
            {
                "role": "user",
                "content": "this is a test request, acknowledge that you got it"
            }
        ],
        stream=True,
        )
        for chunk in response:
            click.echo(f'LiteLLM: streaming response from proxy {chunk}')
        return
    else:
        if headers:
            headers = json.loads(headers)
        initialize(model=model, alias=alias, api_base=api_base, api_version=api_version, debug=debug, temperature=temperature, max_tokens=max_tokens, max_budget=max_budget, telemetry=telemetry, drop_params=drop_params, add_function_to_prompt=add_function_to_prompt, headers=headers, save=save)
        try:
            import uvicorn
        except:
            raise ImportError("Uvicorn needs to be imported. Run - `pip install uvicorn`")
        print(f"\033[32mLiteLLM: Test your local endpoint with: \"litellm --test\" [In a new terminal tab]\033[0m\n")
        print(f"\033[32mLiteLLM: View available endpoints for this server on: http://{host}:{port}\033[0m\n")
        print(f"\033[32mLiteLLM: Self-host your proxy using the following: https://docs.litellm.ai/docs/proxy_server#deploy-proxy \033[0m\n")
        
        if port == 8000 and is_port_in_use(port):
            port = random.randint(1024, 49152)
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

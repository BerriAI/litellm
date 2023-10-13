import click
import subprocess, traceback
import os, sys
import random
from dotenv import load_dotenv

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
@click.option('--api_base', default=None, help='API base URL.')
@click.option('--model', default=None, help='The model name to pass to litellm expects') 
@click.option('--deploy', is_flag=True, type=bool, help='Get a deployed proxy endpoint - api.litellm.ai')
@click.option('--debug', default=False, is_flag=True, type=bool, help='To debug the input') 
@click.option('--temperature', default=None, type=float, help='Set temperature for the model') 
@click.option('--max_tokens', default=None, type=int, help='Set max tokens for the model') 
@click.option('--drop_params', is_flag=True, help='Drop any unmapped params') 
@click.option('--create_proxy', is_flag=True, help='Creates a local OpenAI-compatible server template') 
@click.option('--add_function_to_prompt', is_flag=True, help='If function passed but unsupported, pass it as prompt') 
@click.option('--max_budget', default=None, type=float, help='Set max budget for API calls - works for hosted models like OpenAI, TogetherAI, Anthropic, etc.`') 
@click.option('--telemetry', default=True, type=bool, help='Helps us know if people are using this feature. Turn this off by doing `--telemetry False`') 
@click.option('--test', flag_value=True, help='proxy chat completions url to make a test request to')
@click.option('--local', is_flag=True, default=False, help='for local debugging')
@click.option('--cost', is_flag=True, default=False, help='for viewing cost logs')
def run_server(host, port, api_base, model, deploy, debug, temperature, max_tokens, drop_params, create_proxy, add_function_to_prompt, max_budget, telemetry, test, local, cost):
    global feature_telemetry
    if local:
        from proxy_server import app, initialize, deploy_proxy, print_cost_logs, usage_telemetry
        debug = True
    else:
        try:
            from .proxy_server import app, initialize, deploy_proxy, print_cost_logs, usage_telemetry
        except ImportError as e: 
            from proxy_server import app, initialize, deploy_proxy, print_cost_logs, usage_telemetry
    feature_telemetry = usage_telemetry
    if create_proxy == True: 
        repo_url = 'https://github.com/BerriAI/litellm'
        subfolder = 'litellm/proxy' 
        destination = os.path.join(os.getcwd(), 'litellm-proxy')

        clone_subfolder(repo_url, subfolder, destination)

        return
    if deploy == True:
        print(f"\033[32mLiteLLM: Deploying your proxy to api.litellm.ai\033[0m\n")
        print(f"\033[32mLiteLLM: Deploying proxy for model: {model}\033[0m\n")
        url = deploy_proxy(model, api_base, debug, temperature, max_tokens, telemetry, deploy)
        print(f"\033[32mLiteLLM: Deploy Successfull\033[0m\n")
        print(f"\033[32mLiteLLM: Your deployed url: {url}\033[0m\n")

        print(f"\033[32mLiteLLM: Test your URL using the following: \"litellm --test {url}\"\033[0m")
        return
    if model and "ollama" in model: 
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
        initialize(model, api_base, debug, temperature, max_tokens, max_budget, telemetry, drop_params, add_function_to_prompt)
        try:
            import uvicorn
        except:
            raise ImportError("Uvicorn needs to be imported. Run - `pip install uvicorn`")
        print(f"\033[32mLiteLLM: Deployed Proxy Locally\033[0m\n")
        print(f"\033[32mLiteLLM: Test your local endpoint with: \"litellm --test\" [In a new terminal tab]\033[0m\n")
        print(f"\033[32mLiteLLM: Deploy your proxy using the following: \"litellm --model claude-instant-1 --deploy\" Get an https://api.litellm.ai/chat/completions endpoint \033[0m\n")
        
        if port == 8000 and is_port_in_use(port):
            port = random.randint(1024, 49152)
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

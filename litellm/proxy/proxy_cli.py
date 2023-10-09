import click
import subprocess
import os, appdirs
from dotenv import load_dotenv

load_dotenv()
from importlib import resources
import shutil

config_filename = ".env.litellm"

# Using appdirs to determine user-specific config path
config_dir = appdirs.user_config_dir("litellm")
user_config_path = os.path.join(config_dir, config_filename)

def run_ollama_serve():
    command = ['ollama', 'serve']
    
    with open(os.devnull, 'w') as devnull:
        process = subprocess.Popen(command, stdout=devnull, stderr=devnull)

def load_config():
    try: 
        if not os.path.exists(user_config_path):
            # If user's config doesn't exist, copy the default config from the package
            here = os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.dirname(here)
            default_config_path = os.path.join(parent_dir, '.env.template')
            # Ensure the user-specific directory exists
            os.makedirs(config_dir, exist_ok=True)
            # Copying the file using shutil.copy
            shutil.copy(default_config_path, user_config_path)

        # As the .env file is typically much simpler in structure, we use load_dotenv here directly
        load_dotenv(dotenv_path=user_config_path)
    except:
        pass

def open_config():
    # Create the .env file if it doesn't exist
    if not os.path.exists(user_config_path):
        # If user's env doesn't exist, copy the default env from the package
        here = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.dirname(here)
        default_env_path = os.path.join(parent_dir, '.env.template')
        # Ensure the user-specific directory exists
        os.makedirs(config_dir, exist_ok=True)
        # Copying the file using shutil.copy
        try:
            shutil.copy(default_env_path, user_config_path)
        except Exception as e:
            print(f"Failed to copy .env.template: {e}")

    # Open the .env file in the default editor 
    if os.name == 'nt': # For Windows
        os.startfile(user_config_path)
    elif os.name == 'posix': # For MacOS, Linux, and anything using Bash
        subprocess.call(('open', '-t', user_config_path)) 

@click.command()
@click.option('--host', default='0.0.0.0', help='Host for the server to listen on.')
@click.option('--port', default=8000, help='Port to bind the server to.')
@click.option('--api_base', default=None, help='API base URL.')
@click.option('--model', default=None, help='The model name to pass to litellm expects') 
@click.option('--deploy', is_flag=True, type=bool, help='Get a deployed proxy endpoint - api.litellm.ai')
@click.option('--debug', is_flag=True, help='To debug the input') 
@click.option('--temperature', default=None, type=float, help='Set temperature for the model') 
@click.option('--max_tokens', default=None, type=int, help='Set max tokens for the model') 
@click.option('--drop_params', is_flag=True, help='Drop any unmapped params') 
@click.option('--add_function_to_prompt', is_flag=True, help='If function passed but unsupported, pass it as prompt') 
@click.option('--max_tokens', default=None, type=int, help='Set max tokens for the model') 
@click.option('--telemetry', default=True, type=bool, help='Helps us know if people are using this feature. Turn this off by doing `--telemetry False`') 
@click.option('--config', is_flag=True, help='Create and open .env file from .env.template')
@click.option('--test', flag_value=True, help='proxy chat completions url to make a test request to')
@click.option('--local', is_flag=True, default=False, help='for local debugging')
def run_server(host, port, api_base, model, deploy, debug, temperature, max_tokens, drop_params, add_function_to_prompt, telemetry, config, test, local):
    if config:
        open_config()
    
    if local:
        from proxy_server import app, initialize, deploy_proxy
        debug = True
    else:
        from .proxy_server import app, initialize, deploy_proxy

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
        load_config()
        initialize(model, api_base, debug, temperature, max_tokens, telemetry, drop_params, add_function_to_prompt)


        try:
            import uvicorn
        except:
            raise ImportError("Uvicorn needs to be imported. Run - `pip install uvicorn`")
        print(f"\033[32mLiteLLM: Deployed Proxy Locally\033[0m\n")
        print(f"\033[32mLiteLLM: Test your local endpoint with: \"litellm --test\" [In a new terminal tab]\033[0m\n")
        print(f"\033[32mLiteLLM: Deploy your proxy using the following: \"litellm --model claude-instant-1 --deploy\" Get an https://api.litellm.ai/chat/completions endpoint \033[0m\n")
        
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

import click
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

@click.command()
@click.option('--port', default=8000, help='Port to bind the server to.')
@click.option('--api_base', default=None, help='API base URL.')
@click.option('--model', default=None, help='The model name to pass to litellm expects') 
@click.option('--deploy', is_flag=True, type=bool, help='Get a deployed proxy endpoint - api.litellm.ai')
@click.option('--debug', is_flag=True, help='To debug the input') 
@click.option('--temperature', default=None, type=float, help='Set temperature for the model') 
@click.option('--max_tokens', default=None, help='Set max tokens for the model') 
@click.option('--telemetry', default=True, type=bool, help='Helps us know if people are using this feature. Turn this off by doing `--telemetry False`') 
@click.option('--config', is_flag=True, help='Create and open .env file from .env.template')
@click.option('--test', default=None, help='proxy chat completions url to make a test request to')
def run_server(port, api_base, model, deploy, debug, temperature, max_tokens, telemetry, config, test):
    if config:
        if os.path.exists('.env.template'):
            if not os.path.exists('.env'):
                with open('.env.template', 'r') as source:
                    data = source.read()
                    with open('.env', 'w') as destination:
                        destination.write(data)
            
            click.echo('Opening .env file...')
            subprocess.call(['open', '.env'])  # replace `open` with `start` on Windows
        else:
            click.echo('No .env.template file found.')
    
    from .proxy_server import app, initialize, deploy_proxy
    # from proxy_server import app, initialize, deploy_proxy
    if deploy == True:
        click.echo('LiteLLM: Deploying your proxy server')
        url = deploy_proxy(model, api_base, debug, temperature, max_tokens, telemetry, deploy)
        click.echo(f'LiteLLM: Your deployed url: {url}')
    if test != None:
        click.echo('LiteLLM: Making a test ChatCompletions request to your proxy')
        import openai
        openai.api_base = test
        openai.api_key = "temp-key"
        print(openai.api_base)

        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages = [
            {
                "role": "user",
                "content": "this is a test request, acknowledge that you got it"
            }
        ])
        click.echo(f'LiteLLM: response from proxy {response}')
    else:
        initialize(model, api_base, debug, temperature, max_tokens, telemetry)


        try:
            import uvicorn
        except:
            raise ImportError("Uvicorn needs to be imported. Run - `pip install uvicorn`")
        uvicorn.run(app, host='0.0.0.0', port=port)


if __name__ == "__main__":
    run_server()
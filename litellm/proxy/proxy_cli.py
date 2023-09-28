import click
from dotenv import load_dotenv
load_dotenv()

@click.command()
@click.option('--port', default=8000, help='Port to bind the server to.')
@click.option('--api_base', default=None, help='API base URL.')
@click.option('--model', required=True, help='The model name to pass to litellm expects') 
@click.option('--debug', is_flag=True, help='To debug the input') 
@click.option('--temperature', default=None, type=float, help='Set temperature for the model') 
@click.option('--max_tokens', default=None, help='Set max tokens for the model') 
def run_server(port, api_base, model, debug, temperature, max_tokens):
    from .proxy_server import app, initialize
    initialize(model, api_base, debug, temperature, max_tokens)
    try:
        import uvicorn
    except:
        raise ImportError("Uvicorn needs to be imported. Run - `pip install uvicorn`")
    uvicorn.run(app, host='0.0.0.0', port=port)


if __name__ == "__main__":
    run_server()
import click
from dotenv import load_dotenv
load_dotenv()

@click.command()
@click.option('--port', default=8000, help='Port to bind the server to.')
@click.option('--api_base', default=None, help='API base URL.')
@click.option('--model', required=True, help='The model name to pass to litellm expects') 
def run_server(port, api_base, model):
    from .proxy_server import app, initialize
    initialize(model, api_base)
    try:
        import uvicorn
    except:
        raise ImportError("Uvicorn needs to be imported. Run - `pip install uvicorn`")
    uvicorn.run(app, host='0.0.0.0', port=port)


if __name__ == "__main__":
    run_server()
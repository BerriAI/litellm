import click
import subprocess, traceback, json
import os, sys
import random
from datetime import datetime
import importlib
from dotenv import load_dotenv
import operator

sys.path.append(os.getcwd())

config_filename = "litellm.secrets"

load_dotenv()
from importlib import resources
import shutil

telemetry = None


def run_ollama_serve():
    try:
        command = ["ollama", "serve"]

        with open(os.devnull, "w") as devnull:
            process = subprocess.Popen(command, stdout=devnull, stderr=devnull)
    except Exception as e:
        print(
            f"""
            LiteLLM Warning: proxy started with `ollama` model\n`ollama serve` failed with Exception{e}. \nEnsure you run `ollama serve`
        """
        )  # noqa


def clone_subfolder(repo_url, subfolder, destination):
    # Clone the full repo
    repo_name = repo_url.split("/")[-1]
    repo_master = os.path.join(destination, "repo_master")
    subprocess.run(["git", "clone", repo_url, repo_master])

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
    subprocess.run(["rm", "-rf", os.path.join(destination, "repo_master")])
    feature_telemetry(feature="create-proxy")


def is_port_in_use(port):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@click.command()
@click.option("--host", default="0.0.0.0", help="Host for the server to listen on.")
@click.option("--port", default=8000, help="Port to bind the server to.")
@click.option("--num_workers", default=1, help="Number of uvicorn workers to spin up")
@click.option("--api_base", default=None, help="API base URL.")
@click.option(
    "--api_version",
    default="2023-07-01-preview",
    help="For azure - pass in the api version.",
)
@click.option(
    "--model", "-m", default=None, help="The model name to pass to litellm expects"
)
@click.option(
    "--alias",
    default=None,
    help='The alias for the model - use this to give a litellm model name (e.g. "huggingface/codellama/CodeLlama-7b-Instruct-hf") a more user-friendly name ("codellama")',
)
@click.option(
    "--add_key", default=None, help="The model name to pass to litellm expects"
)
@click.option("--headers", default=None, help="headers for the API call")
@click.option("--save", is_flag=True, type=bool, help="Save the model-specific config")
@click.option(
    "--debug", default=False, is_flag=True, type=bool, help="To debug the input"
)
@click.option(
    "--detailed_debug",
    default=False,
    is_flag=True,
    type=bool,
    help="To view detailed debug logs",
)
@click.option(
    "--use_queue",
    default=False,
    is_flag=True,
    type=bool,
    help="To use celery workers for async endpoints",
)
@click.option(
    "--temperature", default=None, type=float, help="Set temperature for the model"
)
@click.option(
    "--max_tokens", default=None, type=int, help="Set max tokens for the model"
)
@click.option(
    "--request_timeout",
    default=600,
    type=int,
    help="Set timeout in seconds for completion calls",
)
@click.option("--drop_params", is_flag=True, help="Drop any unmapped params")
@click.option(
    "--add_function_to_prompt",
    is_flag=True,
    help="If function passed but unsupported, pass it as prompt",
)
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to the proxy configuration file (e.g. config.yaml). Usage `litellm --config config.yaml`",
)
@click.option(
    "--max_budget",
    default=None,
    type=float,
    help="Set max budget for API calls - works for hosted models like OpenAI, TogetherAI, Anthropic, etc.`",
)
@click.option(
    "--telemetry",
    default=True,
    type=bool,
    help="Helps us know if people are using this feature. Turn this off by doing `--telemetry False`",
)
@click.option(
    "--version",
    "-v",
    default=False,
    is_flag=True,
    type=bool,
    help="Print LiteLLM version",
)
@click.option(
    "--health",
    flag_value=True,
    help="Make a chat/completions request to all llms in config.yaml",
)
@click.option(
    "--test",
    flag_value=True,
    help="proxy chat completions url to make a test request to",
)
@click.option(
    "--test_async",
    default=False,
    is_flag=True,
    help="Calls async endpoints /queue/requests and /queue/response",
)
@click.option(
    "--num_requests",
    default=10,
    type=int,
    help="Number of requests to hit async endpoint with",
)
@click.option("--local", is_flag=True, default=False, help="for local debugging")
def run_server(
    host,
    port,
    api_base,
    api_version,
    model,
    alias,
    add_key,
    headers,
    save,
    debug,
    detailed_debug,
    temperature,
    max_tokens,
    request_timeout,
    drop_params,
    add_function_to_prompt,
    config,
    max_budget,
    telemetry,
    test,
    local,
    num_workers,
    test_async,
    num_requests,
    use_queue,
    health,
    version,
):
    global feature_telemetry
    args = locals()
    if local:
        from proxy_server import app, save_worker_config, usage_telemetry
    else:
        try:
            from .proxy_server import app, save_worker_config, usage_telemetry
        except ImportError as e:
            if "litellm[proxy]" in str(e):
                # user is missing a proxy dependency, ask them to pip install litellm[proxy]
                raise e
            else:
                # this is just a local/relative import error, user git cloned litellm
                from proxy_server import app, save_worker_config, usage_telemetry
    feature_telemetry = usage_telemetry
    if version == True:
        pkg_version = importlib.metadata.version("litellm")
        click.echo(f"\nLiteLLM: Current Version = {pkg_version}\n")
        return
    if model and "ollama" in model and api_base is None:
        run_ollama_serve()
    if test_async is True:
        import requests, concurrent, time

        api_base = f"http://{host}:{port}"

        def _make_openai_completion():
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "Write a short poem about the moon"}
                ],
            }

            response = requests.post("http://0.0.0.0:8000/queue/request", json=data)

            response = response.json()

            while True:
                try:
                    url = response["url"]
                    polling_url = f"{api_base}{url}"
                    polling_response = requests.get(polling_url)
                    polling_response = polling_response.json()
                    print("\n RESPONSE FROM POLLING JOB", polling_response)
                    status = polling_response["status"]
                    if status == "finished":
                        llm_response = polling_response["result"]
                        break
                    print(
                        f"POLLING JOB{polling_url}\nSTATUS: {status}, \n Response {polling_response}"
                    )  # noqa
                    time.sleep(0.5)
                except Exception as e:
                    print("got exception in polling", e)
                    break

        # Number of concurrent calls (you can adjust this)
        concurrent_calls = num_requests

        # List to store the futures of concurrent calls
        futures = []
        start_time = time.time()
        # Make concurrent calls
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrent_calls
        ) as executor:
            for _ in range(concurrent_calls):
                futures.append(executor.submit(_make_openai_completion))

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

        # Summarize the results
        successful_calls = 0
        failed_calls = 0

        for future in futures:
            if future.done():
                if future.result() is not None:
                    successful_calls += 1
                else:
                    failed_calls += 1
        end_time = time.time()
        print(f"Elapsed Time: {end_time-start_time}")
        print(f"Load test Summary:")
        print(f"Total Requests: {concurrent_calls}")
        print(f"Successful Calls: {successful_calls}")
        print(f"Failed Calls: {failed_calls}")
        return
    if health != False:
        import requests

        print("\nLiteLLM: Health Testing models in config")
        response = requests.get(url=f"http://{host}:{port}/health")
        print(json.dumps(response.json(), indent=4))
        return
    if test != False:
        request_model = model or "gpt-3.5-turbo"
        click.echo(
            f"\nLiteLLM: Making a test ChatCompletions request to your proxy. Model={request_model}"
        )
        import openai

        if test == True:  # flag value set
            api_base = f"http://{host}:{port}"
        else:
            api_base = test
        client = openai.OpenAI(api_key="My API Key", base_url=api_base)

        response = client.chat.completions.create(
            model=request_model,
            messages=[
                {
                    "role": "user",
                    "content": "this is a test request, write a short poem",
                }
            ],
            max_tokens=256,
        )
        click.echo(f"\nLiteLLM: response from proxy {response}")

        print(
            f"\n LiteLLM: Making a test ChatCompletions + streaming request to proxy. Model={request_model}"
        )

        response = client.chat.completions.create(
            model=request_model,
            messages=[
                {
                    "role": "user",
                    "content": "this is a test request, write a short poem",
                }
            ],
            stream=True,
        )
        for chunk in response:
            click.echo(f"LiteLLM: streaming response from proxy {chunk}")
        print("\n making completion request to proxy")
        response = client.completions.create(
            model=request_model, prompt="this is a test request, write a short poem"
        )
        print(response)

        return
    else:
        if headers:
            headers = json.loads(headers)
        save_worker_config(
            model=model,
            alias=alias,
            api_base=api_base,
            api_version=api_version,
            debug=debug,
            detailed_debug=detailed_debug,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            max_budget=max_budget,
            telemetry=telemetry,
            drop_params=drop_params,
            add_function_to_prompt=add_function_to_prompt,
            headers=headers,
            save=save,
            config=config,
            use_queue=use_queue,
        )
        try:
            import uvicorn
        except:
            raise ImportError(
                "Uvicorn needs to be imported. Run - `pip install uvicorn`"
            )
        if port == 8000 and is_port_in_use(port):
            port = random.randint(1024, 49152)
        uvicorn.run(
            "litellm.proxy.proxy_server:app", host=host, port=port, workers=num_workers
        )


if __name__ == "__main__":
    run_server()

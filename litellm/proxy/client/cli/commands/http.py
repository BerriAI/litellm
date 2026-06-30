import json as json_lib
from typing import Optional

import click
import rich
import requests

from ...http_client import HTTPClient


@click.group()
def http():
    """Make HTTP requests to the LiteLLM proxy server"""
    pass


@http.command()
@click.argument("method")
@click.argument("uri")
@click.option(
    "--data",
    "-d",
    type=str,
    help="Data to send in the request body (as JSON string)",
)
@click.option(
    "--json",
    "-j",
    type=str,
    help="JSON data to send in the request body (as JSON string)",
)
@click.option(
    "--header",
    "-H",
    multiple=True,
    help="HTTP headers in 'key:value' format. Can be specified multiple times.",
)
@click.pass_context
def request(
    ctx: click.Context,
    method: str,
    uri: str,
    data: Optional[str] = None,
    json: Optional[str] = None,
    header: tuple[str, ...] = (),
):
    """Make an HTTP request to the LiteLLM proxy server

    METHOD: HTTP method (GET, POST, PUT, DELETE, etc.)
    URI: URI path (will be appended to base_url)

    Examples:
        litellm http request GET /models
        litellm http request POST /chat/completions -j '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
        litellm http request GET /health/test_connection -H "X-Custom-Header:value"
    """
    # Parse headers from key:value format
    headers = {}
    for h in header:
        try:
            key, value = h.split(":", 1)
            headers[key.strip()] = value.strip()
        except ValueError:
            raise click.BadParameter(f"Invalid header format: {h}. Expected format: 'key:value'")

    # Parse JSON data if provided
    json_data = None
    if json:
        try:
            json_data = json_lib.loads(json)
        except ValueError as e:
            raise click.BadParameter(f"Invalid JSON format: {e}")

    # Parse data if provided
    request_data = None
    if data:
        try:
            request_data = json_lib.loads(data)
        except ValueError:
            # If not JSON, use as raw data
            request_data = data

    client = HTTPClient(ctx.obj["base_url"], ctx.obj["api_key"])
    try:
        response = client.request(
            method=method,
            uri=uri,
            data=request_data,
            json=json_data,
            headers=headers,
        )
        rich.print_json(data=response)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body = e.response.json()
            rich.print_json(data=error_body)
        except json_lib.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()

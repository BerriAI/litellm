import json
from typing import Optional

import click
import rich
import requests

from ...chat import ChatClient


@click.group()
def chat():
    """Chat with models through the LiteLLM proxy server"""
    pass


@chat.command()
@click.argument("model")
@click.option(
    "--message",
    "-m",
    multiple=True,
    help="Messages in 'role:content' format (e.g. 'user:Hello'). Can be specified multiple times.",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    help="Sampling temperature between 0 and 2",
)
@click.option(
    "--top-p",
    type=float,
    help="Nucleus sampling parameter between 0 and 1",
)
@click.option(
    "--n",
    type=int,
    help="Number of completions to generate",
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum number of tokens to generate",
)
@click.option(
    "--presence-penalty",
    type=float,
    help="Presence penalty between -2.0 and 2.0",
)
@click.option(
    "--frequency-penalty",
    type=float,
    help="Frequency penalty between -2.0 and 2.0",
)
@click.option(
    "--user",
    type=str,
    help="Unique identifier for the end user",
)
@click.pass_context
def completions(
    ctx: click.Context,
    model: str,
    message: tuple[str, ...],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    user: Optional[str] = None,
):
    """Create a chat completion"""
    if not message:
        raise click.UsageError("At least one message is required")

    # Parse messages from role:content format
    messages = []
    for msg in message:
        try:
            role, content = msg.split(":", 1)
            messages.append({"role": role.strip(), "content": content.strip()})
        except ValueError:
            raise click.BadParameter(f"Invalid message format: {msg}. Expected format: 'role:content'")

    client = ChatClient(ctx.obj["base_url"], ctx.obj["api_key"])
    try:
        response = client.completions(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            user=user,
        )
        rich.print_json(data=response)
    except requests.exceptions.HTTPError as e:
        click.echo(f"Error: HTTP {e.response.status_code}", err=True)
        try:
            error_body = e.response.json()
            rich.print_json(data=error_body)
        except json.JSONDecodeError:
            click.echo(e.response.text, err=True)
        raise click.Abort()

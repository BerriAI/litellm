import json
import sys
from typing import Any, Dict, List, Optional

import click
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ... import Client
from ...chat import ChatClient


def _get_available_models(ctx: click.Context) -> List[Dict[str, Any]]:
    """Get list of available models from the proxy server"""
    try:
        client = Client(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])
        models_list = client.models.list()
        # Ensure we return a list of dictionaries
        if isinstance(models_list, list):
            # Filter to ensure all items are dictionaries
            return [model for model in models_list if isinstance(model, dict)]
        return []
    except Exception as e:
        click.echo(f"Warning: Could not fetch models list: {e}", err=True)
        return []


def _select_model(console: Console, available_models: List[Dict[str, Any]]) -> Optional[str]:
    """Interactive model selection"""
    if not available_models:
        console.print("[yellow]No models available or could not fetch models list.[/yellow]")
        model_name = Prompt.ask("Please enter a model name")
        return model_name if model_name.strip() else None
    
    # Display available models in a table
    table = Table(title="Available Models")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Model ID", style="green")
    table.add_column("Owned By", style="yellow")
    MAX_MODELS_TO_DISPLAY = 200
    
    models_to_display: List[Dict[str, Any]] = available_models[:MAX_MODELS_TO_DISPLAY]
    for i, model in enumerate(models_to_display):  # Limit to first 200 models
        table.add_row(
            str(i + 1),
            str(model.get("id", "")),
            str(model.get("owned_by", ""))
        )
    
    if len(available_models) > MAX_MODELS_TO_DISPLAY:
        console.print(f"\n[dim]... and {len(available_models) - MAX_MODELS_TO_DISPLAY} more models[/dim]")
    
    console.print(table)
    
    while True:
        try:
            choice = Prompt.ask(
                "\nSelect a model by entering the index number (or type a model name directly)",
                default="1"
            ).strip()
            
            # Try to parse as index
            try:
                index = int(choice) - 1
                if 0 <= index < len(available_models):
                    return available_models[index]["id"]
                else:
                    console.print(f"[red]Invalid index. Please enter a number between 1 and {len(available_models)}[/red]")
                    continue
            except ValueError:
                # Not a number, treat as model name
                if choice:
                    return choice
                else:
                    console.print("[red]Please enter a valid model name or index[/red]")
                    continue
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Model selection cancelled.[/yellow]")
            return None


@click.command()
@click.argument("model", required=False)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.7,
    help="Sampling temperature between 0 and 2 (default: 0.7)",
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum number of tokens to generate",
)
@click.option(
    "--system",
    "-s",
    type=str,
    help="System message to set the behavior of the assistant",
)
@click.pass_context
def chat(
    ctx: click.Context,
    model: Optional[str],
    temperature: float,
    max_tokens: Optional[int] = None,
    system: Optional[str] = None,
):
    """Interactive chat with streaming responses
    
    Examples:
    
        # Chat with a specific model
        litellm-proxy chat gpt-4
        
        # Chat without specifying model (will show model selection)
        litellm-proxy chat
        
        # Chat with custom settings
        litellm-proxy chat gpt-4 --temperature 0.9 --system "You are a helpful coding assistant"
    """
    console = Console()
    
    # If no model specified, show model selection
    if not model:
        available_models = _get_available_models(ctx)
        model = _select_model(console, available_models)
        if not model:
            console.print("[red]No model selected. Exiting.[/red]")
            return
    
    client = ChatClient(ctx.obj["base_url"], ctx.obj["api_key"])
    
    # Initialize conversation history
    messages: List[Dict[str, Any]] = []
    
    # Add system message if provided
    if system:
        messages.append({"role": "system", "content": system})
    
    # Display welcome message
    console.print(Panel.fit(
        f"[bold blue]LiteLLM Interactive Chat[/bold blue]\n"
        f"Model: [green]{model}[/green]\n"
        f"Temperature: [yellow]{temperature}[/yellow]\n"
        f"Max Tokens: [yellow]{max_tokens or 'unlimited'}[/yellow]\n\n"
        f"Type your messages and press Enter. Type '/quit' or '/exit' to end the session.\n"
        f"Type '/help' for more commands.",
        title="🤖 Chat Session"
    ))
    
    try:
        while True:
            # Get user input
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Chat session ended.[/yellow]")
                break
            
            # Handle special commands
            should_exit, messages, new_model = _handle_special_commands(
                console, user_input, messages, system, ctx
            )
            
            if should_exit:
                break
            if new_model:
                model = new_model
            
            # Check if this was a special command that was handled (not a normal message)
            if user_input.lower().startswith(('/quit', '/exit', '/q', '/help', '/clear', '/history', '/save', '/load', '/model')) or not user_input:
                continue
            
            # Add user message to conversation
            messages.append({"role": "user", "content": user_input})
            
            # Display assistant label
            console.print("\n[bold green]Assistant:[/bold green]")
            
            # Stream the response
            assistant_content = _stream_response(
                console=console,
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Add assistant message to conversation history
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            else:
                console.print("[red]Error: No content received from the model[/red]")
                    
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat session interrupted.[/yellow]")


def _show_help(console: Console):
    """Show help for interactive chat commands"""
    help_text = """
[bold]Interactive Chat Commands:[/bold]

[cyan]/help[/cyan]     - Show this help message
[cyan]/quit[/cyan]     - Exit the chat session (also /exit, /q)
[cyan]/clear[/cyan]    - Clear conversation history
[cyan]/history[/cyan]  - Show conversation history
[cyan]/model[/cyan]    - Switch to a different model
[cyan]/save <name>[/cyan] - Save conversation to file
[cyan]/load <name>[/cyan] - Load conversation from file

[bold]Tips:[/bold]
- Your conversation history is maintained during the session
- Use Ctrl+C to interrupt at any time
- Responses are streamed in real-time
- You can switch models mid-conversation with /model
    """
    console.print(Panel(help_text, title="Help"))


def _show_history(console: Console, messages: List[Dict[str, Any]]):
    """Show conversation history"""
    if not messages:
        console.print("[yellow]No conversation history.[/yellow]")
        return
    
    console.print(Panel.fit("[bold]Conversation History[/bold]", title="History"))
    
    for i, message in enumerate(messages, 1):
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            console.print(f"[dim]{i}. [bold magenta]System:[/bold magenta] {content}[/dim]")
        elif role == "user":
            console.print(f"{i}. [bold cyan]You:[/bold cyan] {content}")
        elif role == "assistant":
            console.print(f"{i}. [bold green]Assistant:[/bold green] {content[:100]}{'...' if len(content) > 100 else ''}")


def _save_conversation(console: Console, messages: List[Dict[str, Any]], command: str):
    """Save conversation to a file"""
    parts = command.split()
    if len(parts) < 2:
        console.print("[red]Usage: /save <filename>[/red]")
        return
    
    filename = parts[1]
    if not filename.endswith('.json'):
        filename += '.json'
    
    try:
        with open(filename, 'w') as f:
            json.dump(messages, f, indent=2)
        console.print(f"[green]Conversation saved to {filename}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving conversation: {e}[/red]")


def _load_conversation(console: Console, command: str, system: Optional[str]) -> List[Dict[str, Any]]:
    """Load conversation from a file"""
    parts = command.split()
    if len(parts) < 2:
        console.print("[red]Usage: /load <filename>[/red]")
        return []
    
    filename = parts[1]
    if not filename.endswith('.json'):
        filename += '.json'
    
    try:
        with open(filename, 'r') as f:
            messages = json.load(f)
        console.print(f"[green]Conversation loaded from {filename}[/green]")
        return messages
    except FileNotFoundError:
        console.print(f"[red]File not found: {filename}[/red]")
    except Exception as e:
        console.print(f"[red]Error loading conversation: {e}[/red]")
    
    # Return empty list or just system message if load failed
    if system:
        return [{"role": "system", "content": system}]
    return []


def _handle_special_commands(
    console: Console, 
    user_input: str, 
    messages: List[Dict[str, Any]], 
    system: Optional[str],
    ctx: click.Context
) -> tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """Handle special chat commands. Returns (should_exit, updated_messages, updated_model)"""
    if user_input.lower() in ['/quit', '/exit', '/q']:
        console.print("[yellow]Chat session ended.[/yellow]")
        return True, messages, None
    elif user_input.lower() == '/help':
        _show_help(console)
        return False, messages, None
    elif user_input.lower() == '/clear':
        new_messages = []
        if system:
            new_messages.append({"role": "system", "content": system})
        console.print("[green]Conversation history cleared.[/green]")
        return False, new_messages, None
    elif user_input.lower() == '/history':
        _show_history(console, messages)
        return False, messages, None
    elif user_input.lower().startswith('/save'):
        _save_conversation(console, messages, user_input)
        return False, messages, None
    elif user_input.lower().startswith('/load'):
        new_messages = _load_conversation(console, user_input, system)
        return False, new_messages, None
    elif user_input.lower() == '/model':
        available_models = _get_available_models(ctx)
        new_model = _select_model(console, available_models)
        if new_model:
            console.print(f"[green]Switched to model: {new_model}[/green]")
            return False, messages, new_model
        return False, messages, None
    elif not user_input:
        return False, messages, None
    
    # Not a special command
    return False, messages, None


def _stream_response(console: Console, client: ChatClient, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int]) -> Optional[str]:
    """Stream the model response and return the complete content"""
    try:
        assistant_content = ""
        for chunk in client.completions_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    assistant_content += content
                    console.print(content, end="")
                    sys.stdout.flush()
        
        console.print()  # Add newline after streaming
        return assistant_content if assistant_content else None
        
    except requests.exceptions.HTTPError as e:
        console.print(f"\n[red]Error: HTTP {e.response.status_code}[/red]")
        try:
            error_body = e.response.json()
            console.print(f"[red]{error_body.get('error', {}).get('message', 'Unknown error')}[/red]")
        except json.JSONDecodeError:
            console.print(f"[red]{e.response.text}[/red]")
        return None
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        return None
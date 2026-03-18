import click
from litellm.llms.gemini.authenticator import GeminiAuthenticator


@click.group(name="gemini")
def gemini():
    """Gemini-specific management commands"""
    pass


@gemini.command(name="login")
def login():
    """Login to Gemini using OAuth Device Flow (Loopback)"""
    try:
        auth = GeminiAuthenticator()
        auth.get_token()
        click.echo("✅ Successfully authenticated with Gemini.")
    except Exception as e:
        raise click.ClickException(f"Gemini authentication failed: {e}")

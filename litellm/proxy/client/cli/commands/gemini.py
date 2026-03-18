import click
from litellm.llms.gemini.authenticator import GeminiAuthenticator


@click.group(name="gemini")
def gemini():
    """Gemini-specific management commands"""
    pass


@gemini.command(name="login")
def login():
    """Authenticate with Gemini using the OAuth Authorization Code flow (loopback redirect)"""
    try:
        auth = GeminiAuthenticator()
        auth.get_token()
        click.echo("✅ Successfully authenticated with Gemini.")
    except Exception as e:
        raise click.ClickException(f"Gemini authentication failed: {e}")

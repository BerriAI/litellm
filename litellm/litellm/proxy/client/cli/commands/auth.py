import json
import os
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

import click


# Token storage utilities
def get_token_file_path() -> str:
    """Get the path to store the authentication token"""
    home_dir = Path.home()
    config_dir = home_dir / ".litellm"
    config_dir.mkdir(exist_ok=True)
    return str(config_dir / "token.json")

def save_token(token_data: Dict[str, Any]) -> None:
    """Save token data to file"""
    token_file = get_token_file_path()
    with open(token_file, 'w') as f:
        json.dump(token_data, f, indent=2)
    # Set file permissions to be readable only by owner
    os.chmod(token_file, 0o600)

def load_token() -> Optional[Dict[str, Any]]:
    """Load token data from file"""
    token_file = get_token_file_path()
    if not os.path.exists(token_file):
        return None
    
    try:
        with open(token_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

def clear_token() -> None:
    """Clear stored token"""
    token_file = get_token_file_path()
    if os.path.exists(token_file):
        os.remove(token_file)

def get_stored_api_key() -> Optional[str]:
    """Get the stored API key from token file"""
    token_data = load_token()
    if token_data and 'key' in token_data:
        return token_data['key']
    return None

# Polling-based authentication - no local server needed

@click.command(name="login")
@click.pass_context
def login(ctx: click.Context):
    """Login to LiteLLM proxy using SSO authentication"""
    import uuid

    import requests

    from litellm.constants import LITELLM_CLI_SOURCE_IDENTIFIER
    from litellm.proxy.client.cli.interface import show_commands
    
    base_url = ctx.obj["base_url"]
    
    # Generate unique key ID for this login session
    key_id = f"sk-{str(uuid.uuid4())}"
    
    try:
        # Construct SSO login URL with CLI source and pre-generated key
        sso_url = f"{base_url}/sso/key/generate?source={LITELLM_CLI_SOURCE_IDENTIFIER}&key={key_id}"
        
        click.echo(f"Opening browser to: {sso_url}")
        click.echo("Please complete the SSO authentication in your browser...")
        click.echo(f"Session ID: {key_id}")
        
        # Open browser
        webbrowser.open(sso_url)
        
        # Poll for key creation
        click.echo("Waiting for authentication...")
        
        poll_url = f"{base_url}/sso/cli/poll/{key_id}"
        timeout = 300  # 5 minute timeout
        poll_interval = 2  # Poll every 2 seconds
        
        for attempt in range(timeout // poll_interval):
            try:
                response = requests.get(poll_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ready":
                        # Key is ready - save it
                        api_key = data.get("key")
                        if api_key:
                            # Save token data (simplified for CLI - we just need the key)
                            save_token({
                                'key': api_key,
                                'user_id': 'cli-user',
                                'user_email': 'unknown',
                                'user_role': 'cli',
                                'auth_header_name': 'Authorization',
                                'jwt_token': '',
                                'timestamp': time.time()
                            })
                            
                            click.echo("✅ Login successful!")
                            click.echo(f"API Key: {api_key[:20]}...")
                            click.echo("You can now use the CLI without specifying --api-key")
                            
                            # Show available commands after successful login
                            click.echo("\n" + "="*60)
                            show_commands()
                            return
                elif response.status_code == 200:
                    # Still pending
                    if attempt % 10 == 0:  # Show progress every 20 seconds
                        click.echo("Still waiting for authentication...")
                else:
                    click.echo(f"Polling error: HTTP {response.status_code}")
                    
            except requests.RequestException as e:
                if attempt % 10 == 0:
                    click.echo(f"Connection error (will retry): {e}")
            
            time.sleep(poll_interval)
        
        click.echo("❌ Authentication timed out. Please try again.")
        return
            
    except KeyboardInterrupt:
        click.echo("\n❌ Authentication cancelled by user.")
        return
    except Exception as e:
        click.echo(f"❌ Authentication failed: {e}")
        return

@click.command(name="logout")
def logout():
    """Logout and clear stored authentication"""
    clear_token()
    click.echo("✅ Logged out successfully. Authentication token cleared.")

@click.command(name="whoami")
def whoami():
    """Show current authentication status"""
    token_data = load_token()
    
    if not token_data:
        click.echo("❌ Not authenticated. Run 'litellm-proxy login' to authenticate.")
        return
    
    click.echo("✅ Authenticated")
    click.echo(f"User Email: {token_data.get('user_email', 'Unknown')}")
    click.echo(f"User ID: {token_data.get('user_id', 'Unknown')}")
    click.echo(f"User Role: {token_data.get('user_role', 'Unknown')}")
    
    # Check if token is still valid (basic timestamp check)
    timestamp = token_data.get('timestamp', 0)
    age_hours = (time.time() - timestamp) / 3600
    click.echo(f"Token age: {age_hours:.1f} hours")
    
    if age_hours > 24:
        click.echo("⚠️ Warning: Token is more than 24 hours old and may have expired.")

# Export individual commands instead of grouping them
# login, logout, and whoami will be added as top-level commands 
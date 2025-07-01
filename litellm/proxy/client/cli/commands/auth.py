import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote, urlparse

import click
import jwt


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

# Callback server for SSO flow
class CallbackHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, token_received_callback=None, **kwargs):
        self.token_received_callback = token_received_callback
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        parsed_url = urlparse(self.path)
        
        if parsed_url.path == '/callback':
            # Extract JWT token from URL parameters
            query_params = parse_qs(parsed_url.query)
            token = query_params.get('token', [None])[0]
            
            if token:
                try:
                    # Decode JWT token (without verification for now, just to extract data)
                    # Note: In production, you might want to verify the token signature
                    decoded_token = jwt.decode(token, options={"verify_signature": False})
                    
                    # Save token data
                    save_token({
                        'key': decoded_token.get('key'),
                        'user_id': decoded_token.get('user_id'),
                        'user_email': decoded_token.get('user_email'),
                        'user_role': decoded_token.get('user_role'),
                        'auth_header_name': decoded_token.get('auth_header_name', 'Authorization'),
                        'jwt_token': token,
                        'timestamp': time.time()
                    })
                    
                    # Send success response
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"""
                    <html>
                    <head><title>Login Success</title></head>
                    <body>
                        <h1>Login Successful!</h1>
                        <p>You have successfully logged in to LiteLLM CLI.</p>
                        <p>You can now close this browser window and return to the CLI.</p>
                        <script>
                            setTimeout(function() {
                                window.close();
                            }, 2000);
                        </script>
                    </body>
                    </html>
                    """)
                    
                    if self.token_received_callback:
                        self.token_received_callback(decoded_token)
                        
                except Exception as e:
                    click.echo(f"Error processing callback: {e}", err=True)
                    self.send_response(400)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"<html><body><h1>Login Failed</h1><p>Error processing authentication.</p></body></html>")
            else:
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Login Failed</h1><p>No authentication token received.</p></body></html>")
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

class CallbackServer:
    def __init__(self, port: int = 8765):
        self.port = port
        self.server = None
        self.token_received = threading.Event()
        self.received_token = None
        
    def token_callback(self, token_data):
        self.received_token = token_data
        self.token_received.set()
        
    def start(self):
        """Start the callback server"""
        def create_handler(*args, **kwargs):
            return CallbackHandler(*args, token_received_callback=self.token_callback, **kwargs)
            
        self.server = HTTPServer(('localhost', self.port), create_handler)
        
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        return f"http://localhost:{self.port}/callback"
    
    def stop(self):
        """Stop the callback server"""
        if self.server:
            self.server.shutdown()
            self.server = None
    
    def wait_for_token(self, timeout: int = 300):
        """Wait for token to be received"""
        return self.token_received.wait(timeout)

@click.command(name="login")
@click.option(
    '--port',
    default=8765,
    help='Port to run the local callback server on (default: 8765)'
)
@click.pass_context
def login(ctx: click.Context, port: int):
    """Login to LiteLLM proxy using SSO authentication"""
    
    base_url = ctx.obj["base_url"]
    
    # Start local callback server
    callback_server = CallbackServer(port=port)
    
    try:
        callback_url = callback_server.start()
        click.echo(f"Started local callback server at {callback_url}")
        
        # Construct CLI SSO login URL with redirect parameter
        sso_url = f"{base_url}/sso/cli/key/generate?redirect_url={quote(callback_url)}"
        
        click.echo(f"Opening browser to: {sso_url}")
        click.echo("Please complete the SSO authentication in your browser...")
        
        # Open browser
        webbrowser.open(sso_url)
        
        # Wait for callback
        click.echo("Waiting for authentication callback...")
        
        if callback_server.wait_for_token(timeout=300):  # 5 minute timeout
            click.echo("✅ Login successful!")
            
            token_data = load_token()
            if token_data:
                click.echo(f"Authenticated as: {token_data.get('user_email', 'Unknown User')}")
                click.echo(f"User Role: {token_data.get('user_role', 'Unknown')}")
                click.echo("You can now use the CLI without specifying --api-key")
            else:
                click.echo("⚠️ Warning: Token was processed but could not be loaded from storage")
        else:
            click.echo("❌ Authentication timed out. Please try again.")
            return
            
    except KeyboardInterrupt:
        click.echo("\n❌ Authentication cancelled by user.")
        return
    except Exception as e:
        click.echo(f"❌ Authentication failed: {e}")
        return
    finally:
        callback_server.stop()

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
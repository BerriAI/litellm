"""
HTML rendering utilities for LiteLLM proxy management endpoints
"""

def _get_litellm_banner() -> str:
    """
    Gets the LiteLLM ASCII banner string
    
    Returns:
        str: The LiteLLM ASCII banner
    """
    # Reuse the exact banner from banner.py
    return """   ██╗     ██╗████████╗███████╗██╗     ██╗     ███╗   ███╗
   ██║     ██║╚══██╔══╝██╔════╝██║     ██║     ████╗ ████║
   ██║     ██║   ██║   █████╗  ██║     ██║     ██╔████╔██║
   ██║     ██║   ██║   ██╔══╝  ██║     ██║     ██║╚██╔╝██║
   ███████╗██║   ██║   ███████╗███████╗███████╗██║ ╚═╝ ██║
   ╚══════╝╚═╝   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝"""


def render_cli_sso_success_page() -> str:
    """
    Renders the CLI SSO authentication success page with the LiteLLM banner
    
    Returns:
        str: HTML content for the success page
    """
    banner = _get_litellm_banner()
    
    html_content = f"""
    <html>
    <head>
        <title>CLI Authentication Successful</title>
        <style>
            body {{
                font-family: monospace;
                background-color: #1a1a1a;
                color: #00ff00;
                text-align: center;
                padding: 20px;
                margin: 0;
            }}
            .banner {{
                white-space: pre;
                font-size: 12px;
                line-height: 1.2;
                margin: 20px 0;
                color: #00ff00;
            }}
            .success-message {{
                color: #ffffff;
                margin: 20px 0;
            }}
            .countdown {{
                color: #ffaa00;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="banner">{banner}</div>
        <div class="success-message">
            <h1>CLI Authentication Successful!</h1>
            <p>Your CLI authentication is complete.</p>
            <p>Return to your terminal - the CLI will automatically detect the successful authentication.</p>
        </div>
        <div class="countdown" id="countdown">This window will close in 3 seconds...</div>
        <script>
            let seconds = 3;
            const countdownElement = document.getElementById('countdown');
            
            const countdown = setInterval(function() {{
                seconds--;
                if (seconds > 0) {{
                    countdownElement.textContent = `This window will close in ${{seconds}} second${{seconds === 1 ? '' : 's'}}...`;
                }} else {{
                    countdownElement.textContent = 'Closing...';
                    clearInterval(countdown);
                    window.close();
                }}
            }}, 1000);
        </script>
    </body>
    </html>
    """
    return html_content 
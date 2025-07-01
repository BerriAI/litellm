import os

from litellm.proxy.client.cli.banner import LITELLM_BANNER


def render_cli_sso_success_page() -> str:
    """
    Renders the CLI SSO authentication success page with modern UI styling
    matching the LiteLLM login page theme
    
    Returns:
        str: HTML content for the success page
    """
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>CLI Authentication Successful - LiteLLM</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background-color: #f8fafc;
                margin: 0;
                padding: 20px;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                color: #333;
            }

            .container {
                background-color: #fff;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                width: 450px;
                max-width: 100%;
                text-align: center;
            }
            
            .logo-container {
                margin-bottom: 30px;
            }
            
            .logo {
                font-size: 24px;
                font-weight: 600;
                color: #1e293b;
            }
            
            h1 {
                margin: 0 0 10px;
                color: #1e293b;
                font-size: 28px;
                font-weight: 600;
            }
            
            .subtitle {
                color: #64748b;
                margin: 0 0 20px;
                font-size: 16px;
            }

            .success-box {
                background-color: #f0fdf4;
                border-radius: 6px;
                padding: 20px;
                margin-bottom: 30px;
                border-left: 4px solid #22c55e;
            }
            
            .success-header {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 12px;
                color: #16a34a;
                font-weight: 600;
                font-size: 16px;
            }
            
            .success-header svg {
                margin-right: 8px;
            }
            
            .success-box p {
                color: #475569;
                margin: 8px 0;
                line-height: 1.5;
                font-size: 14px;
            }

            .countdown {
                color: #f59e0b;
                font-size: 14px;
                font-weight: 500;
                padding: 12px;
                background-color: #fffbeb;
                border-radius: 6px;
                border: 1px solid #fde68a;
            }

            .banner {
                background-color: #1e293b;
                color: #00ff00;
                font-family: 'Courier New', Consolas, monospace;
                font-size: 10px;
                line-height: 1.1;
                white-space: pre;
                padding: 20px;
                border-radius: 6px;
                margin: 20px 0;
                text-align: center;
                border: 1px solid #334155;
                overflow-x: auto;
            }

            .instructions {
                background-color: #f1f5f9;
                border-radius: 6px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 4px solid #3b82f6;
            }
            
            .instructions-header {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 12px;
                color: #1e40af;
                font-weight: 600;
                font-size: 16px;
            }
            
            .instructions-header svg {
                margin-right: 8px;
            }
            
            .instructions p {
                color: #475569;
                margin: 8px 0;
                line-height: 1.5;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo-container">
                <div class="logo">
                    🚅 LiteLLM
                </div>
            </div>
            
            <div class="banner">{LITELLM_BANNER}</div>
            
            <h1>Authentication Successful!</h1>
            <p class="subtitle">Your CLI authentication is complete.</p>
            
            <div class="success-box">
                <div class="success-header">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M9 12l2 2 4-4"></path>
                        <circle cx="12" cy="12" r="10"></circle>
                    </svg>
                    CLI Authentication Complete
                </div>
                <p>Your LiteLLM CLI has been successfully authenticated and is ready to use.</p>
            </div>
            
            <div class="instructions">
                <div class="instructions-header">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                    Next Steps
                </div>
                <p>Return to your terminal - the CLI will automatically detect the successful authentication.</p>
                <p>You can now use LiteLLM CLI commands with your authenticated session.</p>
            </div>
            
            <div class="countdown" id="countdown">This window will close in 3 seconds...</div>
        </div>
        
        <script>
            let seconds = 3;
            const countdownElement = document.getElementById('countdown');
            
            const countdown = setInterval(function() {
                seconds--;
                if (seconds > 0) {
                    countdownElement.textContent = `This window will close in ${seconds} second${seconds === 1 ? '' : 's'}...`;
                } else {
                    countdownElement.textContent = 'Closing...';
                    clearInterval(countdown);
                    window.close();
                }
            }, 1000);
        </script>
    </body>
    </html>
    """
    return html_content 
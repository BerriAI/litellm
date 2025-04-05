import os


def show_missing_vars_in_env():
    from fastapi.responses import HTMLResponse

    from litellm.proxy.proxy_server import master_key, prisma_client

    if prisma_client is None and master_key is None:
        return HTMLResponse(
            content=missing_keys_form(
                missing_key_names="DATABASE_URL, LITELLM_MASTER_KEY"
            ),
            status_code=200,
        )
    if prisma_client is None:
        return HTMLResponse(
            content=missing_keys_form(missing_key_names="DATABASE_URL"), status_code=200
        )

    if master_key is None:
        return HTMLResponse(
            content=missing_keys_form(missing_key_names="LITELLM_MASTER_KEY"),
            status_code=200,
        )
    return None


# LiteLLM Admin UI - Non SSO Login
url_to_redirect_to = os.getenv("PROXY_BASE_URL", "")
url_to_redirect_to += "/login"
html_form = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LiteLLM Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #ffffff;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            color: #333;
        }}

        form {{
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.1);
            width: 450px;
            max-width: 100%;
        }}
        
        .logo-container {{
            text-align: center;
            margin-bottom: 40px;
        }}
        
        .logo {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: 600;
            color: #1e293b;
        }}
        
        .logo img {{
            margin-right: 8px;
        }}
        
        h2 {{
            margin: 0 0 8px 0;
            color: #1e293b;
            font-size: 28px;
            font-weight: 600;
        }}
        
        .subtitle {{
            color: #64748b;
            margin-top: 0;
            margin-bottom: 30px;
            font-size: 16px;
            font-weight: normal;
        }}

        .info-box {{
            background-color: #f1f5f9;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 30px;
            border-left: 4px solid #2563eb;
        }}
        
        .info-header {{
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            color: #1e40af;
            font-weight: 600;
            font-size: 16px;
        }}
        
        .info-header svg {{
            margin-right: 8px;
        }}
        
        .info-box p {{
            color: #475569;
            margin: 8px 0;
            line-height: 1.5;
            font-size: 14px;
        }}

        label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #334155;
            font-size: 14px;
        }}
        
        .required {{
            color: #dc2626;
            margin-left: 2px;
        }}

        input {{
            width: 100%;
            padding: 10px 14px;
            margin-bottom: 20px;
            box-sizing: border-box;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            font-size: 15px;
            color: #1e293b;
            background-color: #fff;
        }}
        
        input:focus {{
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 1px #3b82f6;
        }}

        input[type="submit"] {{
            background-color: #3b82f6;
            color: #fff;
            cursor: pointer;
            font-weight: 500;
            border: none;
            padding: 10px 16px;
            transition: background-color 0.2s;
            border-radius: 6px;
            margin-top: 10px;
            font-size: 14px;
        }}

        input[type="submit"]:hover {{
            background-color: #2563eb;
        }}
        
        a {{
            color: #3b82f6;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}
        
        code {{
            background-color: #f1f5f9;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 13px;
            color: #334155;
        }}
        
        .help-text {{
            color: #64748b;
            font-size: 14px;
            margin-top: -12px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <form action="{url_to_redirect_to}" method="post">
        <div class="logo-container">
            <div class="logo">
                <span>🚅 LiteLLM</span>
            </div>
        </div>
        
        <h2>Login</h2>
        <p class="subtitle">Access your LiteLLM Admin UI.</p>
        
        <div class="info-box">
            <div class="info-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                Default Credentials
            </div>
            <p>By default Username is <code>admin</code> and Password is your set LiteLLM Proxy <code>MASTER_KEY</code></p>
            <p>Need to set UI credentials or SSO? <a href="https://docs.litellm.ai/docs/proxy/ui" target="_blank">Check the documentation</a></p>
        </div>
        
        <label for="username">Username</label>
        <input type="text" id="username" name="username" required placeholder="Enter username">
        
        <label for="password">Password</label>
        <input type="password" id="password" name="password" required placeholder="Enter password">
        
        <input type="submit" value="Login">
    </form>
</body>
</html>
"""


def missing_keys_form(missing_key_names: str):
    missing_keys_html_form = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    color: #333;
                    margin: 20px;
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 800px;
                    margin: auto;
                    padding: 20px;
                    background: #fff;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    font-size: 24px;
                    margin-bottom: 20px;
                }}
                pre {{
                    background: #f8f8f8;
                    padding: 1px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    overflow-x: auto;
                    font-size: 14px;
                }}
                .env-var {{
                    font-weight: normal;
                }}
                .comment {{
                    font-weight: normal;
                    color: #777;
                }}
            </style>
            <title>Environment Setup Instructions</title>
        </head>
        <body>
            <div class="container">
                <h1>Environment Setup Instructions</h1>
                <p>Please add the following variables to your environment variables:</p>
                <pre>
    <span class="env-var">LITELLM_MASTER_KEY="sk-1234"</span> <span class="comment"># Your master key for the proxy server. Can use this to send /chat/completion requests etc</span>
    <span class="env-var">LITELLM_SALT_KEY="sk-XXXXXXXX"</span> <span class="comment"># Can NOT CHANGE THIS ONCE SET - It is used to encrypt/decrypt credentials stored in DB. If value of 'LITELLM_SALT_KEY' changes your models cannot be retrieved from DB</span>
    <span class="env-var">DATABASE_URL="postgres://..."</span> <span class="comment"># Need a postgres database? (Check out Supabase, Neon, etc)</span>
    <span class="comment">## OPTIONAL ##</span>
    <span class="env-var">PORT=4000</span> <span class="comment"># DO THIS FOR RENDER/RAILWAY</span>
    <span class="env-var">STORE_MODEL_IN_DB="True"</span> <span class="comment"># Allow storing models in db</span>
                </pre>
                <h1>Missing Environment Variables</h1>
                <p>{missing_keys}</p>
            </div>

            <div class="container">
            <h1>Need Help? Support</h1>
            <p>Discord: <a href="https://discord.com/invite/wuPM9dRgDw" target="_blank">https://discord.com/invite/wuPM9dRgDw</a></p>
            <p>Docs: <a href="https://docs.litellm.ai/docs/" target="_blank">https://docs.litellm.ai/docs/</a></p>
            </div>
        </body>
        </html>
    """
    return missing_keys_html_form.format(missing_keys=missing_key_names)


def admin_ui_disabled():
    from fastapi.responses import HTMLResponse

    ui_disabled_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    color: #333;
                    margin: 20px;
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 800px;
                    margin: auto;
                    padding: 20px;
                    background: #fff;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    font-size: 24px;
                    margin-bottom: 20px;
                }}
                pre {{
                    background: #f8f8f8;
                    padding: 1px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    overflow-x: auto;
                    font-size: 14px;
                }}
                .env-var {{
                    font-weight: normal;
                }}
                .comment {{
                    font-weight: normal;
                    color: #777;
                }}
            </style>
            <title>Admin UI Disabled</title>
        </head>
        <body>
            <div class="container">
                <h1>Admin UI is Disabled</h1>
                <p>The Admin UI has been disabled by the administrator. To re-enable it, please update the following environment variable:</p>
                <pre>
    <span class="env-var">DISABLE_ADMIN_UI="False"</span> <span class="comment"># Set this to "False" to enable the Admin UI.</span>
                </pre>
                <p>After making this change, restart the application for it to take effect.</p>
            </div>

            <div class="container">
            <h1>Need Help? Support</h1>
            <p>Discord: <a href="https://discord.com/invite/wuPM9dRgDw" target="_blank">https://discord.com/invite/wuPM9dRgDw</a></p>
            <p>Docs: <a href="https://docs.litellm.ai/docs/" target="_blank">https://docs.litellm.ai/docs/</a></p>
            </div>
        </body>
        </html>
    """

    return HTMLResponse(
        content=ui_disabled_html,
        status_code=200,
    )

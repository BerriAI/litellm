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

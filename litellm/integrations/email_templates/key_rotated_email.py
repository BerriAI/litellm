"""
Modern Email Templates for LiteLLM Email Service with professional styling
"""

KEY_ROTATED_EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your API Key Has Been Rotated</title>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #333333;
            background-color: #f8fafc;
            line-height: 1.5;
        }}
        .container {{
            max-width: 560px;
            margin: 20px auto;
            background-color: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header {{
            padding: 24px 0;
            text-align: center;
            border-bottom: 1px solid #f1f5f9;
        }}
        .content {{
            padding: 32px 40px;
        }}
        .greeting {{
            font-size: 16px;
            margin-bottom: 20px;
            color: #333333;
        }}
        .message {{
            font-size: 16px;
            color: #333333;
            margin-bottom: 20px;
        }}
        .key-container {{
            margin: 28px 0;
        }}
        .key-label {{
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            color: #4b5563;
        }}
        .key {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            word-break: break-all;
            background-color: #f9fafb;
            border-radius: 6px;
            padding: 16px;
            font-size: 14px;
            border: 1px solid #e5e7eb;
            color: #4338ca;
        }}
        h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-top: 36px;
            margin-bottom: 16px;
            color: #333333;
        }}
        .budget-info {{
            background-color: #f0fdf4;
            border-radius: 6px;
            padding: 14px 16px;
            margin: 24px 0;
            font-size: 14px;
            border: 1px solid #dcfce7;
        }}
        .code-block {{
            background-color: #f8fafc;
            color: #334155;
            border-radius: 8px;
            padding: 20px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
            overflow-x: auto;
            margin: 20px 0;
            line-height: 1.6;
            border: 1px solid #e2e8f0;
        }}
        .code-comment {{
            color: #64748b;
        }}
        .code-string {{
            color: #0369a1;
        }}
        .code-keyword {{
            color: #7e22ce;
        }}
        .btn {{
            display: inline-block;
            padding: 8px 20px;
            background-color: #6366f1;
            color: #ffffff !important;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            margin-top: 24px;
            text-align: center;
            font-size: 14px;
            transition: background-color 0.2s;
        }}
        .btn:hover {{
            background-color: #4f46e5;
            color: #ffffff !important;
        }}
        .separator {{
            height: 1px;
            background-color: #f1f5f9;
            margin: 40px 0 30px;
        }}
        .footer {{
            padding: 24px 40px 32px;
            text-align: center;
            color: #64748b;
            font-size: 13px;
            background-color: #f8fafc;
            border-top: 1px solid #f1f5f9;
        }}
        .social-links {{
            margin-top: 12px;
        }}
        .social-links a {{
            display: inline-block;
            margin: 0 8px;
            color: #64748b;
            text-decoration: none;
        }}
        @media only screen and (max-width: 620px) {{
            .container {{
                width: 100%;
                margin: 0;
                border-radius: 0;
            }}
            .content {{
                padding: 24px 20px;
            }}
            .footer {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="{email_logo_url}" alt="LiteLLM Logo" style="height: 32px; width: auto;">
        </div>
        <div class="content">
            <div class="greeting">
                <p>Hi {recipient_email},</p>
            </div>
            
            <div class="message">
                <p><strong>Your LiteLLM API key has been rotated</strong> as part of our ongoing commitment to security best practices.</p>
                <p style="margin-top: 16px;">Your previous API key has been deactivated and will no longer work. Please update your applications with the new key below.</p>
            </div>
            
            <div class="key-container">
                <div class="key-label">Your New API Key</div>
                <div class="key">{key_token}</div>
            </div>
            
            <div class="budget-info">
                <p style="margin: 0;"><strong>Monthly Budget:</strong> {key_budget}</p>
            </div>
            
            <h2>Action Required</h2>
            <p>Update your applications and systems with the new API key. Here's an example:</p>
            
            <div class="code-block">
<span class="code-keyword">import</span> openai<br>
<br>
client = openai.OpenAI(<br>
&nbsp;&nbsp;api_key=<span class="code-string">"{key_token}"</span>,<br>
&nbsp;&nbsp;base_url=<span class="code-string">"{base_url}"</span><br>
)<br>
<br>
response = client.chat.completions.create(<br>
&nbsp;&nbsp;model=<span class="code-string">"gpt-3.5-turbo"</span>, <span class="code-comment"># model to send to the proxy</span><br>
&nbsp;&nbsp;messages = [<br>
&nbsp;&nbsp;&nbsp;&nbsp;{{<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="code-string">"role"</span>: <span class="code-string">"user"</span>,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="code-string">"content"</span>: <span class="code-string">"this is a test request, write a short poem"</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;}}<br>
&nbsp;&nbsp;]<br>
)
            </div>
            
            <div class="separator"></div>
            
            <h2>Security Best Practices</h2>
            <p style="margin-bottom: 12px;">To keep your API key secure:</p>
            <ul style="margin: 0; padding-left: 20px; color: #333333;">
                <li style="margin-bottom: 8px;">Never share your API key publicly or commit it to version control</li>
                <li style="margin-bottom: 8px;">Store it securely using environment variables or secret management systems</li>
                <li style="margin-bottom: 8px;">Monitor your API usage regularly for any unusual activity</li>
                <li style="margin-bottom: 8px;">Rotate your keys periodically as a security best practice</li>
            </ul>
            
            <a href="https://docs.litellm.ai/docs/proxy/user_keys" class="btn" style="color: #ffffff;">View Documentation</a>
            
            <div class="separator"></div>
            
            <h2>Need Help?</h2>
            <p>If you have any questions or need assistance updating your systems, please contact us at {email_support_contact}.</p>
        </div>
        {email_footer}
    </div>
</body>
</html>
"""


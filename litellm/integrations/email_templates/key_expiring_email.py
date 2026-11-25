from typing import Final

KEY_EXPIRING_EMAIL_TEMPLATE: Final = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your API Key Is Expiring</title>
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
        .header {{ padding: 24px 0; text-align: center; border-bottom: 1px solid #f1f5f9; }}
        .content {{ padding: 32px 40px; }}
        .greeting, .message {{ font-size: 16px; margin-bottom: 20px; }}
        .key {{
            background-color: #f9fafb;
            border-radius: 6px;
            padding: 16px;
            border: 1px solid #e5e7eb;
            color: #4338ca;
            word-break: break-all;
        }}
        .btn {{
            display: inline-block;
            padding: 8px 20px;
            background-color: #6366f1;
            color: #ffffff !important;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            margin: 24px 0;
        }}
        .footer {{
            padding: 24px 40px 32px;
            text-align: center;
            color: #64748b;
            font-size: 13px;
            background-color: #f8fafc;
            border-top: 1px solid #f1f5f9;
        }}
        @media only screen and (max-width: 620px) {{
            .container {{ width: 100%; margin: 0; border-radius: 0; }}
            .content {{ padding: 24px 20px; }}
            .footer {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="{email_logo_url}" alt="LiteLLM Logo" style="height: 32px; width: auto;">
        </div>
        <div class="content">
            <div class="greeting"><p>Hi {recipient_email},</p></div>
            <div class="message"><p><strong>Your LiteLLM API key is expiring.</strong></p></div>
            <div class="key">{key_display}</div>
            <p>Expires at {expiry_display} ({days_remaining} {day_label} remaining).</p>
            <p>Regenerate your key before it expires and update your applications to avoid interruptions.</p>
            <a href="{base_url}" class="btn">Regenerate your key in the dashboard</a>
            <h2>Need Help?</h2>
            <p>If you need assistance, please contact us at {email_support_contact}.</p>
        </div>
        {email_footer}
    </div>
</body>
</html>
"""

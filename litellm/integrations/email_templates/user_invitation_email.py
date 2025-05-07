"""
Modern Email Templates for LiteLLM Email Service with professional styling
"""

USER_INVITATION_EMAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Invitation to LiteLLM</title>
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
        h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-top: 36px;
            margin-bottom: 16px;
            color: #333333;
        }}
        .btn {{
            display: inline-block;
            padding: 12px 24px;
            background-color: #6366f1;
            color: #ffffff !important;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            margin-top: 24px;
            text-align: center;
            font-size: 15px;
            transition: background-color 0.2s;
        }}
        .btn:hover {{
            background-color: #4f46e5;
            color: #ffffff !important;
        }}
        .highlight-box {{
            background-color: #f0f9ff;
            border-radius: 6px;
            padding: 16px;
            margin: 24px 0;
            font-size: 14px;
            border: 1px solid #bae6fd;
        }}
        .separator {{
            height: 1px;
            background-color: #f1f5f9;
            margin: 40px 0 30px;
        }}
        .features {{
            margin: 30px 0;
        }}
        .feature {{
            display: flex;
            margin-bottom: 16px;
            align-items: flex-start;
        }}
        .feature-icon {{
            width: 24px;
            height: 24px;
            background-color: #e0f2fe;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 12px;
            flex-shrink: 0;
            color: #0284c7;
            font-weight: bold;
        }}
        .feature-text {{
            font-size: 14px;
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
                <p>You've been invited to join LiteLLM! We're excited to have you onboard our platform for unified API access to all LLM providers.</p>
            </div>
            
            <div class="highlight-box">
                <p style="margin: 0;"><strong>Get Started:</strong> Click the button below to set up your account and start using LiteLLM.</p>
            </div>
            
            <div style="text-align: center; margin: 36px 0;">
                <a href="{base_url}" class="btn" style="color: #ffffff;">Accept Invitation</a>
            </div>
            
            <h2>Why use LiteLLM?</h2>
            
            <div class="features">
                <div class="feature">
                    <div class="feature-icon">✓</div>
                    <div class="feature-text">Single API for all LLM providers (OpenAI, Anthropic, Cohere, etc.)</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">✓</div>
                    <div class="feature-text">Automatic fallbacks between models and providers</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">✓</div>
                    <div class="feature-text">Detailed usage tracking and analytics</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">✓</div>
                    <div class="feature-text">Cost management and budget controls</div>
                </div>
            </div>
            
            <div class="separator"></div>
            
            <h2>Need Help?</h2>
            <p>If you have any questions or need assistance, please contact us at {email_support_contact}.</p>
        </div>
        <div class="footer">
            <p>© 2023 LiteLLM. All rights reserved.</p>
            <div class="social-links">
                <a href="https://twitter.com/litellm">Twitter</a> • 
                <a href="https://github.com/BerriAI/litellm">GitHub</a> • 
                <a href="https://litellm.ai">Website</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

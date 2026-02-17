"""
Email Templates used by the LiteLLM Email Service in slack_alerting.py
"""

KEY_CREATED_EMAIL_TEMPLATE = """
                    <img src="{email_logo_url}" alt="LiteLLM Logo" width="150" height="50" />

                    <p> Hi {recipient_email}, <br/>
        
                    I'm happy to provide you with an OpenAI Proxy API Key, loaded with ${key_budget} per month. <br /> <br />

                    <b>
                    Key: <pre>{key_token}</pre> <br>
                    </b>

                    <h2>Usage Example</h2>

                    Detailed Documentation on <a href="https://docs.litellm.ai/docs/proxy/user_keys">Usage with OpenAI Python SDK, Langchain, LlamaIndex, Curl</a>

                    <pre>

                    import openai
                    client = openai.OpenAI(
                        api_key="{key_token}",
                        base_url={{base_url}}
                    )

                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo", # model to send to the proxy
                        messages = [
                            {{
                                "role": "user",
                                "content": "this is a test request, write a short poem"
                            }}
                        ]
                    )

                    </pre>


                    If you have any questions, please send an email to {email_support_contact} <br /> <br />

                    Best, <br />
                    The LiteLLM team <br />
"""


USER_INVITED_EMAIL_TEMPLATE = """
                    <img src="{email_logo_url}" alt="LiteLLM Logo" width="150" height="50" />

                    <p> Hi {recipient_email}, <br/>

                    You were invited to use OpenAI Proxy API for team {team_name}  <br /> <br />

                    <a href="{base_url}" style="display: inline-block; padding: 10px 20px; background-color: #87ceeb; color: #fff; text-decoration: none; border-radius: 20px;">Get Started here</a> <br /> <br />

                    
                    If you have any questions, please send an email to {email_support_contact} <br /> <br />

                    Best, <br />
                    The LiteLLM team <br />
"""

SOFT_BUDGET_ALERT_EMAIL_TEMPLATE = """
                    <img src="{email_logo_url}" alt="LiteLLM Logo" width="150" height="50" />

                    <p> Hi {recipient_email}, <br/>

                    Your LiteLLM API key has crossed its <b>soft budget limit of {soft_budget}</b>. <br /> <br />

                    <b>Current Spend:</b> {spend} <br />
                    <b>Soft Budget:</b> {soft_budget} <br />
                    {max_budget_info}

                    <p style="color: #dc2626; font-weight: 500;">
                    ⚠️ Note: Your API requests will continue to work, but you should monitor your usage closely. 
                    If you reach your maximum budget, requests will be rejected.
                    </p>

                    You can view your usage and manage your budget in the <a href="{base_url}">LiteLLM Dashboard</a>. <br /> <br />

                    If you have any questions, please send an email to {email_support_contact} <br /> <br />

                    Best, <br />
                    The LiteLLM team <br />
"""

TEAM_SOFT_BUDGET_ALERT_EMAIL_TEMPLATE = """
                    <img src="{email_logo_url}" alt="LiteLLM Logo" width="150" height="50" />

                    <p> Hi {team_alias} team member, <br/>

                    Your LiteLLM team has crossed its <b>soft budget limit of {soft_budget}</b>. <br /> <br />

                    <b>Current Spend:</b> {spend} <br />
                    <b>Soft Budget:</b> {soft_budget} <br />
                    {max_budget_info}

                    <p style="color: #dc2626; font-weight: 500;">
                    ⚠️ Note: Your API requests will continue to work, but you should monitor your usage closely. 
                    If you reach your maximum budget, requests will be rejected.
                    </p>

                    You can view your usage and manage your budget in the <a href="{base_url}">LiteLLM Dashboard</a>. <br /> <br />

                    If you have any questions, please send an email to {email_support_contact} <br /> <br />

                    Best, <br />
                    The LiteLLM team <br />
"""

MAX_BUDGET_ALERT_EMAIL_TEMPLATE = """
                    <img src="{email_logo_url}" alt="LiteLLM Logo" width="150" height="50" />

                    <p> Hi {recipient_email}, <br/>

                    Your LiteLLM API key has reached <b>{percentage}% of its maximum budget</b>. <br /> <br />

                    <b>Current Spend:</b> {spend} <br />
                    <b>Maximum Budget:</b> {max_budget} <br />
                    <b>Alert Threshold:</b> {alert_threshold} ({percentage}%) <br />

                    <p style="color: #dc2626; font-weight: 500;">
                    ⚠️ Warning: You are approaching your maximum budget limit. 
                    Once you reach your maximum budget of {max_budget}, all API requests will be rejected.
                    </p>

                    You can view your usage and manage your budget in the <a href="{base_url}">LiteLLM Dashboard</a>. <br /> <br />

                    If you have any questions, please send an email to {email_support_contact} <br /> <br />

                    Best, <br />
                    The LiteLLM team <br />
"""
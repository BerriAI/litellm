"""Template for the model deprecation digest sent to team admins"""

from typing import Final

MODEL_DEPRECATION_EMAIL_TEMPLATE: Final = """
<img src="{email_logo_url}" alt="LiteLLM Logo" width="150" height="50" /> <br/><br/>

<p>Hi team admin,</p>

<p>The following models available to team <b>{team_name}</b> are being deprecated by their provider.
Requests to a deprecated model may start failing on its deprecation date.</p>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
    <thead>
        <tr>
            <th align="left">Model</th>
            <th align="left">Provider</th>
            <th align="left">Deprecation date</th>
            <th align="left">Days left</th>
            <th align="left">Status</th>
        </tr>
    </thead>
    <tbody>
{model_rows}
    </tbody>
</table>

<p>Plan a migration to a supported model. See
<a href="https://docs.litellm.ai/docs/proxy/model_management">the LiteLLM model management docs</a> for guidance.</p>

<p>If you have any questions, please send an email to {email_support_contact}</p>

<p>Best,<br/>The LiteLLM team</p>
"""

MODEL_DEPRECATION_EMAIL_ROW_TEMPLATE: Final = """        <tr>
            <td><code>{model_name}</code></td>
            <td>{provider}</td>
            <td>{deprecation_date}</td>
            <td>{days_left}</td>
            <td>{status}</td>
        </tr>"""

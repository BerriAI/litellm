import os
import sys
import requests
import json

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

def test_zixun_api_gateway_auth():
    """Test zixun api gateway auth."""
    
    try:
        from litellm.proxy.zx.auth.zixun_api_gateway import zixun_api_gateway_auth
        
        client_id = os.environ['ZIXUN_API_GATEWAY_APP_ID']
        client_secret = os.environ['ZIXUN_API_GATEWAY_SECRET']
        base_url = "https://test-etools.ziniao.com"
        if client_id and client_secret:
            url, params, json_body, headers = zixun_api_gateway_auth.request_sign(
                client_id,
                client_secret,
                base_url,
                "post",
                "/api/v1/ding-connector/ding-talk/robot-send-user-message",
                "https://test-etools.ziniao.com/api/v1/ding-connector/ding-talk/robot-send-user-message",
                {},
                {},
                {}
            )
            resp = requests.request('POST', url, json=json_body, headers=headers)
            response_dict = json.loads(resp.text)
            print(response_dict)
    except Exception as e:
        pytest.fail(
            f"Error occurred on zixun api gateway Auth: {e}"
        )


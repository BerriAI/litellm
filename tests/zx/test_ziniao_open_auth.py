import os
import sys
import requests
import json

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

def test_ziniao_open_auth():
    """Test ziniao open api auth."""
    
    try:
        from litellm.proxy.zx.auth.ziniao_open import ziniao_open_auth
        
        client_id = os.environ['ZINIAO_OPEN_APP_ID']
        client_secret = os.environ['ZINIAO_OPEN_APP_SECRET']
        base_url = "https://test-sbappstoreapi.ziniao.com"
        if client_id and client_secret:
            url, params, json_body, headers = ziniao_open_auth.request_sign(
                client_id,
                client_secret,
                base_url,
                "post",
                "/superbrowser/rest/v1/erp/inner/sims/repair_by_admin",
                "https://test-sbappstoreapi.ziniao.com/superbrowser/rest/v1/erp/inner/sims/repair_by_admin",
                {},
                {"companyId":"1","companyName":"1","errorCode":"TIMED_OUT","hostingAccountName":"10"},
                {}
            )
            resp = requests.request('POST', url, json=json_body)
            response_dict = json.loads(resp.text)
            print(response_dict)
    except Exception as e:
        pytest.fail(
            f"Error occurred on ZiNiao Open Auth: {e}"
        )


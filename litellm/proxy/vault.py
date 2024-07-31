import os
import traceback

import hvac

VAULT_ADDR = str(os.environ.get("VAULT_ADDR", "")).strip()
VAULT_TOKEN = str(os.environ.get("VAULT_TOKEN", "")).strip()
client = hvac.Client(url=VAULT_ADDR, token=VAULT_TOKEN, verify=False)


def get_api_keys(user_id):
    try:
        path = f'user_api_keys/{user_id}'
        secrets = client.secrets.kv.v2.read_secret_version(path=path, raise_on_deleted_version=True)
        data = secrets['data']['data']
        return data
    except Exception as e:
        print(f"exception : {str(e)}")
        traceback.print_exc()
        return {}

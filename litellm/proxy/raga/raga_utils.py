from fastapi import HTTPException

API_KEY = "api_key"
API_BASE = "api_base"
API_VERSION = "api_version"

AZURE_API_KEY = "AZURE_API_KEY"
AZURE_API_BASE = "AZURE_API_BASE"
AZURE_API_VERSION = "AZURE_API_VERSION"

AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
AWS_REGION_NAME = "AWS_REGION_NAME"
OLLAMA_API_BASE = "OLLAMA_API_BASE"


def modify_user_request(data):
    try:
        if "provider" in data:
            data["model"] = data["provider"] + "/" + data["model"]
            del data["provider"]
        if "user_id" in data:
            set_api_keys_from_vault(data)
            del data["user_id"]
        return data
    except HTTPException as e:
        raise e
    except Exception as e:
        return HTTPException(status_code=500, detail=e)


def set_api_keys_from_vault(data):
    print(f"getting api keys for user: {data['user_id']}")
    import litellm.proxy.raga.vault as vault

    vault_secrets = vault.get_api_keys(data["user_id"])

    model_name = data["model"]
    if model_name.startswith("azure"):
        validate_api_keys(vault_secrets, model_name, [AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION])
        data[API_KEY] = vault_secrets.get(AZURE_API_KEY)
        data[API_BASE] = vault_secrets.get(AZURE_API_BASE)
        data[API_VERSION] = vault_secrets.get(AZURE_API_VERSION)
    elif model_name.startswith("bedrock"):
        validate_api_keys(vault_secrets, model_name, [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME])
        data["aws_access_key_id"] = vault_secrets.get(AWS_ACCESS_KEY_ID)
        data["aws_secret_access_key"] = vault_secrets.get(AWS_SECRET_ACCESS_KEY)
        data["aws_region_name"] = vault_secrets.get(AWS_REGION_NAME)
    elif model_name.startswith("ollama"):
        validate_api_keys(vault_secrets, model_name, [OLLAMA_API_BASE])
        data[API_BASE] = vault_secrets.get(OLLAMA_API_BASE)
    else:
        from litellm.proxy.raga.data import get_model_keys

        keys = get_model_keys(model_name)
        if len(keys) == 1:
            validate_api_keys(vault_secrets, model_name, keys)
            data[API_KEY] = vault_secrets.get(keys[0])
        else:
            raise Exception(f"Model {model_name} is not supported")


def validate_api_keys(vault_secrets, model_name, required_keys):
    not_set_keys = []
    for key in required_keys:
        if vault_secrets.get(key, "") == "":
            not_set_keys.append(key)

    if len(not_set_keys) > 0:
        raise HTTPException(status_code=401, detail=f"Required API Keys are not set for {model_name}: {not_set_keys}")

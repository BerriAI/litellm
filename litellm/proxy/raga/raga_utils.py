
API_KEY = "api_key"
API_BASE = "api_base"
API_VERSION = "api_version"

OPENAI_API_KEY = "OPENAI_API_KEY"
AZURE_API_KEY = "AZURE_API_KEY"
AZURE_API_BASE = "AZURE_API_BASE"
AZURE_API_VERSION = "AZURE_API_VERSION"
GROQ_API_KEY = "GROQ_API_KEY"
GEMINI_API_KEY = "GEMINI_API_KEY"


def modify_user_request(data):
    try:
        if "user_id" in data:
            set_api_keys_from_vault(data)
            del data["user_id"]
        return data, None
    except Exception as e:
        return None, {
            "error": {
                "message": f"Error: {str(e)}",
                "code": 400
            }
        }


def set_api_keys_from_vault(data):
    print(f"getting api keys for user: {data['user_id']}")
    import litellm.proxy.raga.vault as vault
    vault_secrets = vault.get_api_keys(data['user_id'])
    if data["model"].startswith("gpt"):
        validate_api_keys(vault_secrets, [OPENAI_API_KEY])
        data[API_KEY] = vault_secrets.get(OPENAI_API_KEY)

    elif data["model"].startswith("azure"):
        validate_api_keys(vault_secrets, [AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION])
        data[API_KEY] = vault_secrets.get(AZURE_API_KEY)
        data[API_BASE] = vault_secrets.get(AZURE_API_BASE)
        data[API_VERSION] = vault_secrets.get(AZURE_API_VERSION)

    elif data["model"].startswith("groq"):
        validate_api_keys(vault_secrets, [GROQ_API_KEY])
        data[API_KEY] = vault_secrets.get(GROQ_API_KEY)

    elif data["model"].startswith("gemini"):
        validate_api_keys(vault_secrets, [GEMINI_API_KEY])
        data[API_KEY] = vault_secrets.get(GEMINI_API_KEY)


def validate_api_keys(vault_secrets, required_keys):
    not_set_keys = []
    for key in required_keys:
        if vault_secrets.get(key, "") == "":
            not_set_keys.append(key)

    if len(not_set_keys) > 0:
        raise Exception(f"Required API Keys are not set: {not_set_keys}")



def modify_user_request(data):
    # set user api keys from vault
    if "user_id" in data:
        print(f"getting api keys for user: {data['user_id']}")
        import litellm.proxy.raga.vault as vault
        vault_secrets = vault.get_api_keys(data['user_id'])
        if data["model"].startswith("gpt"):
            data["api_key"] = vault_secrets.get("OPENAI_API_KEY", "abcd")
        elif data["model"].startswith("azure"):
            data["api_key"] = vault_secrets.get("AZURE_API_KEY", "abcd")
            data["api_base"] = vault_secrets.get("AZURE_API_BASE", "abcd")
            data["api_version"] = vault_secrets.get("AZURE_API_VERSION", "abcd")
        elif data["model"].startswith("groq"):
            data["api_key"] = vault_secrets.get("GROQ_API_KEY", "abcd")
        elif data["model"].startswith("gemini"):
            data["api_key"] = vault_secrets.get("GEMINI_API_KEY", "abcd")

        del data["user_id"]

    return data

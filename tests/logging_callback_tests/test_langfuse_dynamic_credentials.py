from litellm.integrations.langfuse.langfuse import resolve_langfuse_credentials


def test_resolve_langfuse_credentials_does_not_use_env_for_dynamic_host(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "global-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "global-secret")

    public_key, secret_key, host = resolve_langfuse_credentials(
        langfuse_host="https://attacker.example",
        allow_env_credentials=False,
    )

    assert public_key is None
    assert secret_key is None
    assert host == "https://attacker.example"


def test_resolve_langfuse_credentials_accepts_secret_key_alias_for_dynamic_host(
    monkeypatch,
):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "global-secret")

    public_key, secret_key, host = resolve_langfuse_credentials(
        langfuse_public_key="dynamic-public",
        langfuse_secret_key="dynamic-secret",
        langfuse_host="https://team-langfuse.example",
        allow_env_credentials=False,
    )

    assert public_key == "dynamic-public"
    assert secret_key == "dynamic-secret"
    assert host == "https://team-langfuse.example"


def test_resolve_langfuse_credentials_keeps_env_for_global_config(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "global-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "global-secret")

    public_key, secret_key, host = resolve_langfuse_credentials(
        langfuse_host="https://admin-configured.example",
        allow_env_credentials=True,
    )

    assert public_key == "global-public"
    assert secret_key == "global-secret"
    assert host == "https://admin-configured.example"

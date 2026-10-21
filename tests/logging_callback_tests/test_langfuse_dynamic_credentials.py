import litellm








def test_upstream_langfuse_env_only_warns_and_opens_no_second_channel(monkeypatch, caplog):
    """UPSTREAM_LANGFUSE_* configured a second v2 ingestion client. v4 has one export channel per
    credential set, so the values are ignored with a startup warning and never build anything."""
    from litellm.integrations.langfuse import langfuse_sdk
    from litellm.integrations.langfuse.langfuse import LangFuseLogger

    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    monkeypatch.setattr(langfuse_sdk, "_TRACING", {})
    monkeypatch.setenv("LANGFUSE_MOCK", "true")
    monkeypatch.setenv("UPSTREAM_LANGFUSE_SECRET_KEY", "upstream-secret")
    monkeypatch.setenv("UPSTREAM_LANGFUSE_PUBLIC_KEY", "upstream-public")
    monkeypatch.setenv("UPSTREAM_LANGFUSE_HOST", "https://upstream.example")

    with caplog.at_level("WARNING", logger="LiteLLM"):
        logger = LangFuseLogger(
            langfuse_public_key="public",
            langfuse_secret="secret",
            langfuse_host="https://langfuse.example",
        )

    assert any("UPSTREAM_LANGFUSE_* is no longer supported" in record.getMessage() for record in caplog.records)
    assert [lease.tracing for lease in langfuse_sdk._TRACING.values()] == [logger.tracing]
    assert all(key.public_key == "public" for key in langfuse_sdk._TRACING)

from litellm.proxy.gateway.mcp.authn.resolver import anonymous_subject, resolve_subject


def test_resolve_subject_stub_ignores_bearer_and_returns_anonymous():
    assert resolve_subject("any-token") == anonymous_subject()
    assert resolve_subject(None) == anonymous_subject()


def test_anonymous_subject_shape():
    subject = anonymous_subject()
    assert subject.subject_id == "anonymous"
    assert subject.tenant == "anonymous"

import pytest
import time
import hmac
import hashlib
import base64
import json


def make_token(secret: str, score: float = 0.88, expired: bool = False) -> str:
    now = int(time.time())
    payload = {"sub": "test", "score": score, "iat": now, "exp": now + (- 1 if expired else 120)}
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    sig = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"v1.{payload_b64}.{sig_b64}"


SECRET = "test_secret_htl"


@pytest.fixture
def hook():
    from litellm.proxy.custom_hooks.x_trust_hook import XTrustHook
    return XTrustHook(secret=SECRET, min_score=0.0)


@pytest.mark.asyncio
async def test_valid_token_annotates_trusted(hook):
    token = make_token(SECRET, score=0.88)
    data = {"metadata": {"headers": {"x-trust": token}}}
    result = await hook.async_pre_call_hook(None, None, data, None)
    assert result["metadata"]["x_trust"]["trusted"] is True
    assert result["metadata"]["x_trust"]["score"] == 0.88


@pytest.mark.asyncio
async def test_wrong_secret_annotates_untrusted(hook):
    token = make_token("wrong_secret", score=0.88)
    data = {"metadata": {"headers": {"x-trust": token}}}
    result = await hook.async_pre_call_hook(None, None, data, None)
    assert result["metadata"]["x_trust"]["trusted"] is False


@pytest.mark.asyncio
async def test_no_token_annotates_untrusted(hook):
    data = {"metadata": {"headers": {}}}
    result = await hook.async_pre_call_hook(None, None, data, None)
    assert result["metadata"]["x_trust"]["trusted"] is False


@pytest.mark.asyncio
async def test_expired_token_annotates_untrusted(hook):
    token = make_token(SECRET, score=0.88, expired=True)
    data = {"metadata": {"headers": {"x-trust": token}}}
    result = await hook.async_pre_call_hook(None, None, data, None)
    assert result["metadata"]["x_trust"]["trusted"] is False

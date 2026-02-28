#!/usr/bin/env python3
"""
Start mock JWKS/OIDC server (:19900) and fake LLM backend (:19901).
Also writes signed JWTs to /tmp/tokens.env for use in shell demos.
Runs forever — Ctrl-C to stop.
"""

import base64, json, os, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import jwt as pyjwt

# ── RSA key pair ──────────────────────────────────────────────────────────────
RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
RSA_PUB = RSA_KEY.public_key()
PUB_NUMBERS = RSA_PUB.public_numbers()

def _b64url(n, length):
    return base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode()

KID = "demo-kid-1"
JWKS = {"keys": [{"kty": "RSA", "kid": KID, "use": "sig", "alg": "RS256",
                   "n": _b64url(PUB_NUMBERS.n, 256), "e": _b64url(PUB_NUMBERS.e, 3)}]}

PEM_PRIVATE = RSA_KEY.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())

ISSUER = "https://demo-idp.example.com"
AUDIENCE = "litellm-proxy-demo"

def sign_token(claims):
    payload = {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 3600, **claims}
    return pyjwt.encode(payload, PEM_PRIVATE, algorithm="RS256", headers={"kid": KID})

# ── Write tokens to file ─────────────────────────────────────────────────────
valid_token = sign_token({"sub": "user-1", "roles": ["team-beta", "team-gamma"]})
tampered_token = valid_token[:-5] + "XXXXX"
empty_roles_token = sign_token({"sub": "user-2", "roles": []})
single_role_token = sign_token({"sub": "user-3", "roles": ["team-alpha"]})

with open("/tmp/tokens.env", "w") as f:
    f.write(f'VALID_TOKEN="{valid_token}"\n')
    f.write(f'TAMPERED_TOKEN="{tampered_token}"\n')
    f.write(f'EMPTY_ROLES_TOKEN="{empty_roles_token}"\n')
    f.write(f'SINGLE_ROLE_TOKEN="{single_role_token}"\n')

print("[tokens] Written to /tmp/tokens.env")

# ── Mock JWKS + OIDC discovery server ────────────────────────────────────────
class JWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/.well-known/openid-configuration":
            body = json.dumps({"issuer": ISSUER,
                               "jwks_uri": "http://127.0.0.1:19900/jwks",
                               "authorization_endpoint": "https://example.com/authorize",
                               "token_endpoint": "https://example.com/token"}).encode()
        elif self.path == "/jwks":
            body = json.dumps(JWKS).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *_): pass

# ── Fake LLM backend ─────────────────────────────────────────────────────────
class FakeLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.dumps({"id": "chatcmpl-fake", "object": "chat.completion",
            "created": 1700000000, "model": "fake-model",
            "choices": [{"index": 0, "message": {"role": "assistant",
                         "content": "Hello from fake LLM!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if "/models" in self.path:
            body = json.dumps({"data": [{"id": "fake-model", "object": "model"}], "object": "list"}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *_): pass

# ── Start servers ─────────────────────────────────────────────────────────────
print("[JWKS]  Starting on :19900 (OIDC discovery + JWKS)")
jwks_srv = HTTPServer(("127.0.0.1", 19900), JWKSHandler)
threading.Thread(target=jwks_srv.serve_forever, daemon=True).start()

print("[LLM]   Starting fake LLM backend on :19901")
llm_srv = HTTPServer(("127.0.0.1", 19901), FakeLLMHandler)
threading.Thread(target=llm_srv.serve_forever, daemon=True).start()

print("[READY] Mock servers running. Press Ctrl-C to stop.")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\nShutting down.")

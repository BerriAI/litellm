#!/usr/bin/env python3
"""
Integration demo for JWT auth fixes on branch fix/jwt-auth-oidc-array-roles.

Spins up:
  - Mock JWKS + OIDC discovery server on port 19900
  - Fake LLM backend on port 19901
  - LiteLLM proxy on port 19902  (premium_user patched to bypass license)

Tests three scenarios:
  Fix 1  OIDC discovery URL resolution  (valid JWT → 200, tampered → 401)
  Fix 2  roles claim as JSON array      (["team-beta","team-gamma"] → 200, [] → 401)
  Fix 3  Helpful error for dot-notation  (roles.0 → 401 with hint)
"""

import asyncio, base64, json, os, signal, subprocess, sys, textwrap, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import jwt as pyjwt

# ── 1. RSA key pair ──────────────────────────────────────────────────────────

RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
RSA_PUB = RSA_KEY.public_key()
PUB_NUMBERS = RSA_PUB.public_numbers()


def _b64url(n: int, length: int) -> str:
    return base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode()


KID = "demo-kid-1"
JWKS = {
    "keys": [{
        "kty": "RSA", "kid": KID, "use": "sig", "alg": "RS256",
        "n": _b64url(PUB_NUMBERS.n, 256),
        "e": _b64url(PUB_NUMBERS.e, 3),
    }]
}

PEM_PRIVATE = RSA_KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)

ISSUER = "https://demo-idp.example.com"
AUDIENCE = "litellm-proxy-demo"


def sign_token(claims: dict) -> str:
    payload = {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 3600, **claims}
    return pyjwt.encode(payload, PEM_PRIVATE, algorithm="RS256", headers={"kid": KID})


# ── 2. Mock JWKS + OIDC discovery server (port 19900) ────────────────────────

class JWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/.well-known/openid-configuration":
            body = json.dumps({
                "issuer": ISSUER,
                "jwks_uri": "http://127.0.0.1:19900/jwks",
                "authorization_endpoint": "https://example.com/authorize",
                "token_endpoint": "https://example.com/token",
            }).encode()
        elif self.path == "/jwks":
            body = json.dumps(JWKS).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


# ── 3. Fake LLM backend (port 19901) ─────────────────────────────────────────

class FakeLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.dumps({
            "id": "chatcmpl-fake", "object": "chat.completion", "created": 1700000000,
            "model": "fake-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from fake LLM!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if "/models" in self.path:
            body = json.dumps({"data": [{"id": "fake-model", "object": "model"}], "object": "list"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *_):
        pass


# ── 4. Proxy startup that patches premium_user ───────────────────────────────

PROXY_PORT = 19902

CONFIG_FIX12 = textwrap.dedent("""\
    model_list:
      - model_name: fake-model
        litellm_params:
          model: openai/fake-model
          api_key: fake-key
          api_base: http://127.0.0.1:19901/
    general_settings:
      enable_jwt_auth: true
      litellm_jwtauth:
        team_id_jwt_field: "roles"
        team_id_default: ""
    litellm_settings:
      drop_params: true
""")

CONFIG_FIX3 = textwrap.dedent("""\
    model_list:
      - model_name: fake-model
        litellm_params:
          model: openai/fake-model
          api_key: fake-key
          api_base: http://127.0.0.1:19901/
    general_settings:
      enable_jwt_auth: true
      litellm_jwtauth:
        team_id_jwt_field: "roles.0"
        team_id_default: ""
    litellm_settings:
      drop_params: true
""")


def _write_config(content: str, path: str):
    with open(path, "w") as f:
        f.write(content)


PROXY_WRAPPER = textwrap.dedent("""\
    import sys, os, unittest.mock
    os.environ["JWT_PUBLIC_KEY_URL"] = "http://127.0.0.1:19900/.well-known/openid-configuration"
    os.environ["JWT_AUDIENCE"] = "{audience}"
    os.environ["DISABLE_SCHEMA_UPDATE"] = "true"
    # Patch premium_user BEFORE proxy imports the variable
    import litellm.proxy.proxy_server as _ps
    _ps.premium_user = True
    from litellm.proxy.proxy_server import app, initialize
    import asyncio, uvicorn
    asyncio.get_event_loop().run_until_complete(initialize(config="{config_path}"))
    uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
""")


def start_proxy(config_path: str) -> subprocess.Popen:
    wrapper_code = PROXY_WRAPPER.format(
        audience=AUDIENCE,
        config_path=config_path,
        port=PROXY_PORT,
    )
    wrapper_path = "/tmp/_proxy_wrapper.py"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_code)
    proc = subprocess.Popen(
        [sys.executable, wrapper_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_proxy(timeout: int = 90) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/health/readiness", timeout=2)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def stop_proxy(proc: subprocess.Popen):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def proxy_request(token: str, model: str = "fake-model") -> tuple:
    import urllib.request, urllib.error
    data = json.dumps({"model": model, "messages": [{"role": "user", "content": "Hi"}]}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=15)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ── 5. Pretty output ─────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
results = []


def header(text: str):
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")


def check(label: str, condition: bool, detail: str = ""):
    mark = f"{GREEN}PASS{RESET}" if condition else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")
    results.append((label, condition))


# ── 6. Main ──────────────────────────────────────────────────────────────────

def main():
    _write_config(CONFIG_FIX12, "/tmp/config_fix12.yaml")
    _write_config(CONFIG_FIX3,  "/tmp/config_fix3.yaml")

    print(f"{BOLD}Starting mock JWKS + OIDC discovery server on :19900 ...{RESET}")
    jwks_srv = HTTPServer(("127.0.0.1", 19900), JWKSHandler)
    threading.Thread(target=jwks_srv.serve_forever, daemon=True).start()

    print(f"{BOLD}Starting fake LLM backend on :19901 ...{RESET}")
    llm_srv = HTTPServer(("127.0.0.1", 19901), FakeLLMHandler)
    threading.Thread(target=llm_srv.serve_forever, daemon=True).start()

    # ── Fix 1 + 2 ────────────────────────────────────────────────────────
    header("Fix 1 + Fix 2: OIDC Discovery & Array Roles")
    proxy = start_proxy("/tmp/config_fix12.yaml")
    try:
        print(f"  Waiting for proxy on :{PROXY_PORT} (may take ~30s for Prisma) ...")
        if not wait_for_proxy():
            print(f"{RED}Proxy failed to start!{RESET}")
            out = proxy.stdout.read().decode() if proxy.stdout else ""
            print(out[-3000:])
            raise SystemExit(1)
        print(f"  {GREEN}Proxy is up.{RESET}\n")

        # Fix 1a
        token_ok = sign_token({"sub": "user-1", "roles": ["team-beta", "team-gamma"]})
        status, body = proxy_request(token_ok)
        check("Fix 1 — OIDC discovery: valid JWT → HTTP 200", status == 200, f"status={status}")

        # Fix 1b
        tampered = token_ok[:-5] + "XXXXX"
        status, body = proxy_request(tampered)
        check("Fix 1 — OIDC discovery: tampered JWT → HTTP 401", status == 401, f"status={status}")

        # Fix 2a
        token_arr = sign_token({"sub": "user-2", "roles": ["team-beta", "team-gamma"]})
        status, body = proxy_request(token_arr)
        check("Fix 2 — roles array ['team-beta','team-gamma'] → HTTP 200", status == 200, f"status={status}")

        # Fix 2b
        token_empty = sign_token({"sub": "user-3", "roles": []})
        status, body = proxy_request(token_empty)
        check("Fix 2 — empty roles [] → HTTP 401", status == 401, f"status={status}")

    finally:
        stop_proxy(proxy)
        time.sleep(2)

    # ── Fix 3 ─────────────────────────────────────────────────────────────
    header("Fix 3: Dot-notation hint (roles.0)")
    proxy = start_proxy("/tmp/config_fix3.yaml")
    try:
        print(f"  Waiting for proxy on :{PROXY_PORT} ...")
        if not wait_for_proxy():
            print(f"{RED}Proxy failed to start!{RESET}")
            out = proxy.stdout.read().decode() if proxy.stdout else ""
            print(out[-3000:])
            raise SystemExit(1)
        print(f"  {GREEN}Proxy is up.{RESET}\n")

        token_dot = sign_token({"sub": "user-4", "roles": ["team-alpha"]})
        status, body = proxy_request(token_dot)
        hint_text = "Use 'roles' instead"
        has_hint = hint_text in body
        check(
            "Fix 3 — roles.0 config → HTTP 401 with helpful hint",
            status == 401 and has_hint,
            f"status={status}, hint_present={has_hint}\n"
            f"body excerpt: {body[:400]}",
        )

    finally:
        stop_proxy(proxy)

    # ── Summary ───────────────────────────────────────────────────────────
    all_pass = all(ok for _, ok in results)
    header("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print()
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

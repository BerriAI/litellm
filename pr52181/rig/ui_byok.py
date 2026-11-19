"""Browser leg of the BYOK A/B: a signed-in dashboard user connects the BYOK server the way an MCP
client sends them to, through the proxy's own authorize page, and a loopback callback (the MCP client's
side) redeems the code. Then the same user calls the tool from the dashboard.

usage: python ui_byok.py <out_dir> <users.json>
"""
import base64
import hashlib
import html
import json
import secrets
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright

OUT, USERS = sys.argv[1], json.load(open(sys.argv[2]))
PORT = {"base": 16292, "head": 16293}
SHA = {"base": "49b2058c1a7", "head": "d3e196d654c"}
CALLBACK = "http://127.0.0.1:47777/callback"
ATTEMPTS = 6
PENDING: dict[str, tuple[str, str]] = {}
TALLY: dict[str, list[tuple[int, str]]] = {"base": [], "head": []}


def exchange(side: str, code: str, verifier: str) -> tuple[int, str]:
    data = urllib.parse.urlencode({"grant_type": "authorization_code", "code": code, "redirect_uri": CALLBACK,
                                   "code_verifier": verifier, "client_id": "rig-mcp-client"}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT[side]}/v1/mcp/oauth/token", data=data, method="POST",
                                 headers={"Connection": "close"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            return resp.status, "access token issued" if body.get("access_token") else str(body)[:80]
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()).get("error", "")


class Callback(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        return

    def do_GET(self) -> None:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        state, code = q.get("state", [""])[0], q.get("code", [""])[0]
        side, verifier = PENDING.pop(state)
        status, detail = exchange(side, code, verifier)
        TALLY[side].append((status, detail))
        ok = sum(1 for s, _ in TALLY[side] if s == 200)
        rows = "".join(
            f"<tr><td>{i}</td><td class={'ok' if s == 200 else 'bad'}>{s}</td><td>{html.escape(d)}</td></tr>"
            for i, (s, d) in enumerate(TALLY[side], 1))
        page = f"""<!doctype html><html><head><meta charset=utf-8><title>MCP client callback</title>
<style>body{{font:16px -apple-system,sans-serif;margin:40px;color:#111}}h1{{font-size:22px}}
table{{border-collapse:collapse;margin-top:16px}}td,th{{border:1px solid #ccc;padding:6px 14px;text-align:left}}
.ok{{color:#0a7d32;font-weight:600}}.bad{{color:#b3261e;font-weight:600}}.big{{font-size:20px;margin:12px 0}}</style></head>
<body><h1>MCP client loopback callback, {side} proxy at {SHA[side]} (4 workers, shared Redis)</h1>
<div class=big>This attempt: POST /v1/mcp/oauth/token returned <span class={'ok' if status == 200 else 'bad'}>{status} {html.escape(detail)}</span></div>
<div class=big>Connects that worked so far: {ok} of {len(TALLY[side])}</div>
<table><tr><th>attempt</th><th>token status</th><th>result</th></tr>{rows}</table></body></html>"""
        body = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    return verifier, base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 47777), Callback)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    report = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for side in ("base", "head"):
            user = USERS[side]
            key_value = f"sk-ui-{side}-{secrets.token_hex(4)}"
            ctx = browser.new_context(viewport={"width": 1280, "height": 900})
            pg = ctx.new_page()
            pg.goto(f"http://127.0.0.1:{PORT[side]}/ui/", wait_until="networkidle")
            pg.fill("input[name=username]", user["email"])
            pg.fill("input[type=password]", user["password"])
            pg.get_by_role("button", name="Login", exact=True).click()
            pg.wait_for_timeout(4000)
            for i in range(1, ATTEMPTS + 1):
                verifier, challenge = pkce()
                state = f"{side}-{i}-{secrets.token_hex(3)}"
                PENDING[state] = (side, verifier)
                query = urllib.parse.urlencode({
                    "client_id": "rig-mcp-client", "redirect_uri": CALLBACK, "response_type": "code",
                    "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
                    "server_id": user["server_id"]})
                pg.goto(f"http://127.0.0.1:{PORT[side]}/v1/mcp/oauth/authorize?{query}", wait_until="networkidle")
                if i == 1:
                    pg.screenshot(path=f"{OUT}/01_authorize_page_{side}_{SHA[side]}.png")
                pg.locator("button.btn-primary").click()
                pg.locator("#apiKey").fill(key_value)
                if i == 1:
                    pg.screenshot(path=f"{OUT}/02_api_key_entered_{side}_{SHA[side]}.png")
                pg.locator("#connectBtn").click()
                pg.wait_for_url("http://127.0.0.1:47777/callback*", timeout=30000)
                pg.wait_for_load_state("load")
                if TALLY[side][-1][0] != 200 and not report.get(f"{side}_fail_shot"):
                    pg.screenshot(path=f"{OUT}/03_callback_failed_{side}_{SHA[side]}.png", full_page=True)
                    report[f"{side}_fail_shot"] = i
            pg.screenshot(path=f"{OUT}/04_callback_tally_{side}_{SHA[side]}.png", full_page=True)
            # The same signed-in user calls the BYOK tool from the dashboard
            pg.goto(f"http://127.0.0.1:{PORT[side]}/ui/mcp-servers", wait_until="networkidle")
            pg.wait_for_timeout(3000)
            pg.screenshot(path=f"{OUT}/05_mcp_servers_{side}_{SHA[side]}.png", full_page=True)
            got = None
            try:
                pg.get_by_text("byokecho", exact=True).first.click()
                pg.wait_for_timeout(3000)
                pg.get_by_role("tab", name="MCP Tools").click()
                pg.get_by_text("whoami", exact=True).first.wait_for(timeout=60000)
                pg.get_by_text("whoami", exact=True).first.click()
                pg.wait_for_timeout(1500)
                pg.get_by_role("button", name="Call Tool").click()
                pg.get_by_text("fp=", exact=False).first.wait_for(timeout=30000)
                pg.wait_for_timeout(800)
                got = pg.get_by_text("fp=", exact=False).first.inner_text()
                pg.screenshot(path=f"{OUT}/06_tool_call_{side}_{SHA[side]}.png", full_page=True)
            except Exception as e:  # noqa: BLE001
                pg.screenshot(path=f"{OUT}/06_tool_call_attempt_{side}_{SHA[side]}.png", full_page=True)
                print(side, "dashboard tool call step failed:", str(e)[:200])
                print("buttons", [t.strip()[:30] for t in pg.locator("button, [role=tab], a").all_inner_texts() if t.strip()][:60])
            report[side] = {"key_fp": hashlib.sha256(key_value.encode()).hexdigest()[:12], "dashboard_tool": got,
                            "tally": TALLY[side]}
            ctx.close()
        browser.close()
    server.shutdown()
    print(json.dumps(report, indent=1))


main()

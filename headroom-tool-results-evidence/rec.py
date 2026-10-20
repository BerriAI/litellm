import http.server, json, os, sys, urllib.error, urllib.request
R = {"/hr": "http://127.0.0.1:8787", "/anthropic": "https://api.anthropic.com", "/openai": "https://api.openai.com"}
FAULTS = {"http500": (500, b'{"error":"x"}'), "nonjson": (200, b"<html>"), "nomessages": (200, b"{}")}
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        p = next(k for k in R if self.path.startswith(k + "/")); body = self.rfile.read(int(self.headers["content-length"]))
        hdr = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "accept-encoding")}
        try: r = urllib.request.urlopen(urllib.request.Request(R[p] + self.path[len(p):], body, hdr), timeout=180); st, rb, ct = r.status, r.read(), r.headers.get("content-type", "application/json")
        except urllib.error.HTTPError as e: st, rb, ct = e.code, e.read(), "application/json"
        f = open("fault").read().strip() if os.path.exists("fault") else ""
        rec = lambda: json.dump({"req": json.loads(body), "resp": json.loads(rb) if rb[:1] == b"{" else None, "resp_raw": None if rb[:1] == b"{" else rb[:200000].decode("utf-8", "replace")},
                                open(f"tap/{open('case').read().strip()}.{p[1:]}.json", "w"))
        if p == "/hr" and f == "close": rb = b""; rec(); self.close_connection = True; return
        if p == "/hr" and f in FAULTS: st, rb = FAULTS[f]
        if p == "/hr" and f == "drop_row": rb = json.dumps({**json.loads(rb), "messages": json.loads(rb)["messages"][:-1]}).encode()
        rec()
        self.send_response(st); self.send_header("content-type", ct if not (p == "/hr" and f) else "application/json"); self.send_header("content-length", str(len(rb)))
        self.end_headers(); self.wfile.write(rb)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()

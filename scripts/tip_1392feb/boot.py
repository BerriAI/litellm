import json, re
from playwright.sync_api import sync_playwright
pw = sync_playwright().start()
br = pw.chromium.connect_over_cdp("http://localhost:29229")
ctx = br.contexts[0]; page = ctx.pages[0] if ctx.pages else ctx.new_page()
page.set_viewport_size({"width":1440,"height":1000})
REQS = []
page.on("request", lambda r: REQS.append((r.method, r.url, r.post_data)) if r.method in ("POST","PATCH","PUT") and ":4000" in r.url else None)
KEY = re.search(r"master_key: (\S+)", open("/home/ubuntu/qa_ar/config.yaml").read()).group(1)
def shot(n): page.screenshot(path=f"/home/ubuntu/qa_ar/shots_base/{n}.png", full_page=False)
def login():
    ctx.clear_cookies(); page.goto("http://localhost:3000/login"); page.evaluate("()=>{localStorage.clear();sessionStorage.clear()}")
    page.goto("http://localhost:3000/login"); page.wait_for_timeout(3000)
    u = page.locator("input[name=username], input#username, input[type=text]").first
    p = page.locator("input[type=password]").first
    if p.count():
        u.fill("admin"); p.fill(KEY); page.get_by_role("button", name=re.compile("log ?in|sign in", re.I)).first.click(); page.wait_for_timeout(4000)
    print("url after login", page.url)

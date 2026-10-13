import json, re, time
page.set_default_timeout(8000)
FREQ = "button:has-text('Every request'), button:has-text('Every new user message'), button:has-text('Once per session')"
OUT = "/home/ubuntu/qa_ar/shots_base"
LOG = []
def D(): return page.locator("[role=dialog]").last
def outline(locs):
    page.evaluate("()=>document.querySelectorAll('[data-qa-mark]').forEach(e=>{e.style.outline='';e.removeAttribute('data-qa-mark')})")
    for l in locs:
        try:
            if l.count(): l.first.evaluate("e=>{e.setAttribute('data-qa-mark','1');e.style.outline='3px solid red';e.style.outlineOffset='2px'}")
        except Exception: pass
def snap(name, locs=()):
    outline(list(locs))
    if locs:
        try: locs[0].first.scroll_into_view_if_needed(timeout=2000)
        except Exception: pass
    page.screenshot(path=f"{OUT}/{name}.png"); outline([])
def open_add(name):
    page.goto("http://localhost:3000/models-and-endpoints"); page.wait_for_timeout(2500)
    t = page.get_by_role("tab", name="Auto-Routers")
    if t.count(): t.first.click(); page.wait_for_timeout(800)
    page.get_by_role("button", name="Add Auto Router").first.click(); page.wait_for_timeout(1500)
    D().get_by_label("Auto Router Name").fill(name)
def L_family(): return D().locator("[role=radio][aria-checked=true]").locator("xpath=ancestor::label[1]")
def L_freq(): return D().locator(FREQ)
def L_tpl(): return D().get_by_test_id("template-selector")
def L_dec(): return D().get_by_role("combobox", name="Decision model")
def L_judge(): return D().get_by_role("combobox", name="Judge model")
def L_heur(): return D().get_by_role("button", name="Heuristic")
def L_appr(): return D().get_by_role("button", name="Routing approach")
def st(tag):
    d = D()
    fam = d.locator("[role=radio][aria-checked=true]").evaluate_all("els=>els.map(e=>(e.closest('label')?.innerText||document.getElementById(e.getAttribute('aria-labelledby')||'')?.innerText||'').split('\\n')[0])")
    def cv(c):
        if not c.count(): return "<none>"
        try: return c.first.input_value()
        except Exception: return c.first.inner_text()
    def bt(c): return c.first.inner_text().split("\n")[0] if c.count() else "<none>"
    tpl = L_tpl()
    s = {"family": fam, "heur": bt(L_heur()), "approach": bt(L_appr()), "decision": cv(L_dec()), "judge": cv(L_judge()),
         "freq": bt(L_freq()), "template": (tpl.first.inner_text().split("\n")[0] if tpl.count() else "<hidden>"),
         "tpl_disabled": (tpl.first.is_disabled() or tpl.first.get_attribute("aria-disabled")=="true" or tpl.first.get_attribute("data-disabled") is not None) if tpl.count() else None,
         "tier_names": d.locator("[aria-label^='Name for tier']").evaluate_all("els=>els.map(e=>e.value)"),
         "tier_models": d.locator("[aria-label*='Tier'] , [aria-label*='tier']").count()}
    LOG.append((tag, s)); print(tag, json.dumps(s)); return s
def freq(n):
    L_freq().first.click(); page.wait_for_timeout(400); page.get_by_role("option", name=n).click(); page.wait_for_timeout(400)
def tpl(n):
    L_tpl().first.click(); page.wait_for_timeout(500); page.get_by_role("option", name=re.compile("^"+re.escape(n))).first.click(); page.wait_for_timeout(1000)
def fam(n): D().get_by_text(n, exact=True).first.click(); page.wait_for_timeout(800)
def pick(combo, opt=None):
    D().get_by_role("combobox", name=combo).first.click(); page.wait_for_timeout(500)
    o = page.get_by_role("option", name=opt) if opt else page.get_by_role("option")
    v = o.first.inner_text(); o.first.click(); page.wait_for_timeout(500); return v
def menu(btn, item):
    D().get_by_role("button", name=btn).first.click(); page.wait_for_timeout(500)
    page.get_by_role("menuitemradio", name=re.compile(item)).first.click(); page.wait_for_timeout(1000)
def confirm_dialog():
    a = page.locator("[role=alertdialog]")
    return a.first.inner_text()[:400] if a.count() else None
def submit(fn):
    REQS.clear(); D().get_by_role("button", name=re.compile("^(Add Auto Router|Save Changes)$")).last.click(timeout=5000); page.wait_for_timeout(3500)
    for m,u,d in REQS:
        if d and ("model/new" in u or "/update" in u or "model/" in u):
            try: p=json.loads(d)
            except Exception: continue
            c=(p.get("litellm_params") or {}).get("complexity_router_config")
            if c is None: continue
            open(f"{OUT}/{fn}.json","w").write(json.dumps(p,indent=2))
            r={k:c.get(k) for k in ("classifier_type","jev_classifier_config","classifier_llm_config","classification_mode","tiers","custom_tier_set","adaptive")}
            print("PAYLOAD", m, u.split('?')[0].split(':4000')[1], json.dumps(r)[:1000]); return r
    print("NO PAYLOAD", [(m,u) for m,u,_ in REQS])
def persisted(name):
    r = page.request.get("http://localhost:4000/v2/model/info", headers={"Authorization": f"Bearer {KEY}"})
    for m in r.json()["data"]:
        if m["model_name"]==name:
            c = m["litellm_params"].get("complexity_router_config"); c = json.loads(c) if isinstance(c,str) else c
            r2 = {"db_model": m["model_info"].get("db_model"), **{k:(c or {}).get(k) for k in ("classifier_type","jev_classifier_config","classifier_llm_config","classification_mode","tiers","custom_tier_set")}}
            open(f"{OUT}/persisted_{name}.json","w").write(json.dumps(r2,indent=2)); print("PERSISTED", json.dumps(r2)[:900]); return r2
    print("NOT FOUND", name)
def reopen(name):
    page.goto("http://localhost:3000/models-and-endpoints"); page.wait_for_timeout(2500)
    t = page.get_by_role("tab", name="Auto-Routers")
    if t.count(): t.first.click(); page.wait_for_timeout(1000)
    page.get_by_text(name, exact=True).first.click(); page.wait_for_timeout(2500)
    page.get_by_role("button", name=re.compile("Edit Auto Router")).first.click(); page.wait_for_timeout(2500)
def case(n, f):
    try: f()
    except Exception as e: print("CASE-ERR", n, str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_{n}.png")

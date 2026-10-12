FAM = {"jev": "Decisions Model", "heur": "Heuristics", "llm": "LLM"}
TPLS = ["1M Context", "Anthropic Family", "Gemini Family", "Lite", "OpenAI Family"]
TIERKEYS = ("tiers", "tier_model_params", "tier_model_configs")
RESULTS = {}
def ui_tiers():
    return D().locator("input[aria-label^='Select model(s) for']").evaluate_all("""els=>Object.fromEntries(els.map(e=>{
      const box=e.closest('[data-slot=combobox-chips]')||e.parentElement;
      const chips=[...box.querySelectorAll('[aria-label]')].map(c=>c.getAttribute('aria-label')).filter(a=>a&&a!==e.getAttribute('aria-label')&&a!=='Clear all'&&!a.startsWith('Remove'));
      return [e.getAttribute('aria-label').replace('Select model(s) for ','').replace(' queries',''), chips];}))""")
def ui_efforts():
    return D().locator("[aria-label^='Reasoning effort for']").evaluate_all("els=>Object.fromEntries(els.map(e=>[e.getAttribute('aria-label'),e.innerText.trim()]))")
def setup_family(f):
    if f == "jev": fam("Decisions Model"); pick("Decision model", "jev-decider")
    elif f == "llm": fam("LLM"); pick("Judge model", "claude-fable-5-1")
    freq("Every new user message")
def full_payload(fn):
    p = json.load(open(f"{OUT}/{fn}.json")); return p["litellm_params"]["complexity_router_config"]
def nontier(c): return {k: v for k, v in c.items() if k not in TIERKEYS}
def c2m():
    for f in ["jev", "heur", "llm"]:
        for t in TPLS:
            tag = f"c2m_{f}_{t.split()[0].lower()}"; n = f"{P}-{tag.replace('_','-')}"
            try:
                open_add(n); setup_family(f); before = st(f"{tag} before"); tpl(t); after = st(f"{tag} after")
                ui = ui_tiers(); eff = ui_efforts()
                lk = {"jev": L_dec, "heur": L_heur, "llm": L_judge}[f]
                snap(f"{P}_{tag}", [L_family(), lk(), L_freq(), L_tpl()])
                r = submit(f"{P}_{tag}_payload"); c = full_payload(f"{P}_{tag}_payload"); per = persisted(n)
                pt = {k: v for k, v in c["tiers"].items() if v}
                ut = {k: v for k, v in ui.items() if v}
                RESULTS[tag] = {"before": before, "after": after, "ui_tiers": ui, "ui_efforts": eff, "payload_nontier": nontier(c),
                                "payload_tiers": c["tiers"], "payload_tier_model_params": c.get("tier_model_params"), "persisted": per,
                                "ui_unchanged_nontier": {k: before[k] == after[k] for k in ("family", "heur", "decision", "judge", "freq")}}
                print("C2M", tag, "ui_nontier_same", RESULTS[tag]["ui_unchanged_nontier"], "ui_tiers", json.dumps(ut)[:300], "payload_tiers", json.dumps(pt)[:300])
            except Exception as e:
                print("C2M-ERR", tag, str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_{tag}.png")
        tag = f"c2r_{f}"; n = f"{P}-c2r-{f}"
        try:
            open_add(n); setup_family(f); b = st(f"{tag} before")
            for t in ["OpenAI Family", "Gemini Family", "Anthropic Family"]: tpl(t); st(f"{tag} after {t}")
            snap(f"{P}_{tag}_o_g_a", [L_family(), {"jev": L_dec, "heur": L_heur, "llm": L_judge}[f](), L_freq(), L_tpl()])
            submit(f"{P}_{tag}_payload"); RESULTS[tag] = {"payload_nontier": nontier(full_payload(f"{P}_{tag}_payload")), "persisted": persisted(n)}
        except Exception as e:
            print("C2R-ERR", tag, str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_{tag}.png")
def lite_copy():
    open_add(f"{P}-lite-copy"); L_tpl().first.click(); page.wait_for_timeout(700)
    opts = page.get_by_role("option").evaluate_all("els=>els.map(e=>e.innerText.replace(/\\n/g,' | '))")
    print("TPL-OPTS", json.dumps(opts)); RESULTS["template_options"] = opts
    lite = page.get_by_role("option", name=re.compile("^Lite"))
    snap(f"{P}_lite_copy_dropdown", [lite]); page.keyboard.press("Escape"); page.wait_for_timeout(300)
    tpl("Lite"); s = D().get_by_text(re.compile("Cost-optimized routing", re.I))
    RESULTS["lite_visible_desc"] = s.all_inner_texts()[:3]; print("LITE-DESC", RESULTS["lite_visible_desc"])
    snap(f"{P}_lite_copy_selected", [L_tpl(), s, L_family()])
def adaptive_loc(): return D().locator("label:has-text('Enable adaptive bandit selection')").locator("[role=switch]")
def adaptive_state():
    a = adaptive_loc()
    if not a.count():
        adv = D().get_by_role("button", name="Advanced settings")
        if adv.count() and not D().get_by_text("Sessions and efficiency").count(): adv.first.scroll_into_view_if_needed(); adv.first.click(); page.wait_for_timeout(800)
        h = D().get_by_text("Sessions and efficiency")
        if h.count(): h.first.scroll_into_view_if_needed(); h.first.click(); page.wait_for_timeout(800)
    a = adaptive_loc(); return a.first.get_attribute("aria-checked") if a.count() else "<none>"
def c5ad():
    for appr in ["Fuse v2", "Capability"]:
        tag = f"c5ad_{appr[:3].lower()}"; n = f"{P}-{tag.replace('_','-')}"
        try:
            open_add(n); fam("LLM"); pick("Judge model", "claude-fable-5-1"); freq("Every new user message")
            adaptive_state(); adaptive_loc().first.click(); page.wait_for_timeout(600)
            b = st(f"{tag} before"); b["adaptive"] = adaptive_state(); b["ui_tiers"] = {k: v for k, v in ui_tiers().items() if v}; print(tag, "before adaptive", b["adaptive"], "tiers", b["ui_tiers"])
            snap(f"{P}_{tag}_a_before", [adaptive_loc(), L_appr(), L_judge(), L_freq()])
            menu("Routing approach", appr); cd = confirm_dialog(); print(tag, "confirm:", cd); res = {"before": b, "confirm": cd}
            if cd:
                snap(f"{P}_{tag}_b_confirm", [page.locator("[role=alertdialog]")])
                page.locator("[role=alertdialog]").get_by_role("button", name=re.compile("cancel|keep", re.I)).first.click(); page.wait_for_timeout(800)
                c = st(f"{tag} cancelled"); c["adaptive"] = adaptive_state(); res["cancelled"] = c; print(tag, "cancel adaptive", c["adaptive"])
                snap(f"{P}_{tag}_c_cancelled", [adaptive_loc(), L_appr(), L_judge(), L_freq()])
                menu("Routing approach", appr); page.locator("[role=alertdialog]").get_by_role("button").last.click(); page.wait_for_timeout(1000)
            a = st(f"{tag} applied"); a["adaptive"] = adaptive_state(); res["applied"] = a; print(tag, "applied adaptive", a["adaptive"])
            snap(f"{P}_{tag}_d_applied", [L_appr(), adaptive_loc(), L_family()])
            menu("Routing approach", "Complexity"); cd2 = confirm_dialog(); res["back_confirm"] = cd2
            if cd2: page.locator("[role=alertdialog]").get_by_role("button").last.click(); page.wait_for_timeout(800)
            k = st(f"{tag} back"); k["adaptive"] = adaptive_state(); res["back"] = k; print(tag, "back adaptive", k["adaptive"])
            snap(f"{P}_{tag}_e_back", [L_appr(), L_judge(), L_freq(), adaptive_loc()])
            RESULTS[tag] = res
        except Exception as e:
            print("C5AD-ERR", tag, str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_{tag}.png")
def clear_tier(label):
    inp = D().locator(f"input[aria-label='Select model(s) for {label} queries']")
    box = inp.first.locator("xpath=..")
    for _ in range(3):
        c = box.get_by_role("button", name="Clear all")
        if c.count(): c.first.click(); page.wait_for_timeout(500); break
        box = box.locator("xpath=..")
def c5ad2():
    for appr in ["Fuse v2", "Capability"]:
        tag = f"c5ao_{appr[:3].lower()}"; n = f"{P}-{tag.replace('_','-')}"
        try:
            open_add(n); fam("LLM"); pick("Judge model", "claude-fable-5-1"); freq("Every new user message")
            clear_tier("medium"); clear_tier("complex")
            adaptive_state(); adaptive_loc().first.click(); page.wait_for_timeout(600)
            def S(ph):
                s = st(f"{tag} {ph}"); s["adaptive"] = adaptive_state(); s["ui_tiers"] = {k: v for k, v in ui_tiers().items() if v}
                print(tag, ph, json.dumps({x: s[x] for x in ("approach", "judge", "freq", "adaptive", "ui_tiers")})); return s
            res = {"before": S("before")}
            snap(f"{P}_{tag}_a_before", [adaptive_loc(), L_appr(), D().locator("input[aria-label^='Select model(s) for']")])
            menu("Routing approach", appr); cd = confirm_dialog(); res["confirm"] = cd; print(tag, "confirm:", (cd or "NONE")[:200].replace("\n", " / "))
            if cd:
                snap(f"{P}_{tag}_b_confirm", [page.locator("[role=alertdialog]")])
                page.locator("[role=alertdialog]").get_by_role("button", name=re.compile("cancel|keep", re.I)).first.click(); page.wait_for_timeout(800)
                res["cancelled"] = S("cancelled"); snap(f"{P}_{tag}_c_cancelled", [adaptive_loc(), L_appr(), L_judge(), L_freq()])
                menu("Routing approach", appr); page.locator("[role=alertdialog]").get_by_role("button").last.click(); page.wait_for_timeout(1000)
            res["applied"] = S("applied"); snap(f"{P}_{tag}_d_applied", [L_appr(), L_family()])
            menu("Routing approach", "Complexity"); res["back_confirm"] = confirm_dialog()
            if res["back_confirm"]: page.locator("[role=alertdialog]").get_by_role("button").last.click(); page.wait_for_timeout(800)
            res["back"] = S("back"); snap(f"{P}_{tag}_e_back", [L_appr(), L_judge(), L_freq(), adaptive_loc()])
            RESULTS[tag] = res
        except Exception as e:
            print("C5AO-ERR", tag, str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_{tag}.png")

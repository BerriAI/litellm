H = {"Authorization": f"Bearer {KEY}"}
def api_post(path, body):
    r = page.request.post("http://localhost:4000" + path, headers=H, data=body); print("API", path, r.status); return r.status
def all_models(): return page.request.get("http://localhost:4000/v2/model/info", headers=H).json()["data"]
def purge():
    n = 0
    for m in all_models():
        if m["model_info"].get("db_model") and str(m["litellm_params"].get("model", "")).startswith("auto_router"):
            api_post("/model/delete", {"id": m["model_info"]["id"]}); n += 1
    print("PURGED", n)
def full_cfg(name):
    for m in all_models():
        if m["model_name"] == name:
            c = m["litellm_params"].get("complexity_router_config"); return json.loads(c) if isinstance(c, str) else c
def accept_any():
    if page.locator("[role=alertdialog]").count():
        page.locator("[role=alertdialog]").get_by_role("button").last.click(); page.wait_for_timeout(900); return True
    return False
def appr(a): menu("Routing approach", a); return accept_any()
def payload_body(fn, name, drop_rubric=False):
    b = json.load(open(f"{OUT}/{fn}.json")); b["model_name"] = name
    c = b["litellm_params"]["complexity_router_config"]
    if drop_rubric and c.get("classifier_llm_config"): c["classifier_llm_config"].pop("classification_rubric", None)
    return b
def rubric(c): return (c or {}).get("classifier_llm_config")
def add_custom_tier(nm="LEGAL"):
    D().get_by_role("button", name="Edit tiers").click(); page.wait_for_timeout(600)
    D().get_by_role("button", name="Add tier").click(); page.wait_for_timeout(600)
    D().locator("[aria-label^='Name for tier']").last.fill(nm); D().locator("[aria-label^='Definition for tier']").last.fill("Legal questions"); page.wait_for_timeout(500)
def mk_forecast(tag, a):
    n = f"{P}-{tag}"; open_add(n); fam("LLM"); pick("Judge model", "claude-fable-5-1"); appr(a)
    if a == "Capability":
        t = D().get_by_text("Solve probability threshold", exact=True).locator("xpath=following::input[1]"); t.fill("0.5"); page.wait_for_timeout(500)
    st(f"{tag} created {a}")
    submit(f"{P}_{tag}_create_payload"); return n, full_cfg(n)
def r1h(): r1([("r1d", "Capability", "Heuristics")])
def r1(cases=None):
    purge()
    for tag, a, via in cases or [("r1a", "Capability", None), ("r1b", "Fuse v2", "Heuristics"), ("r1c", "Capability", "Decisions Model")]:
        try:
            n, created = mk_forecast(tag, a); reopen(n); s0 = st(f"{tag} reopened")
            sv = None
            if via: fam(via); st(f"{tag} via {via}"); fam("LLM"); sv = st(f"{tag} llm again")
            appr("Complexity"); s2 = st(f"{tag} complexity")
            snap(f"{P}_{tag}_reloaded_{a[:3]}_to_complexity", [L_appr(), L_family(), L_judge()])
            submit(f"{P}_{tag}_save_payload"); saved = full_cfg(n)
            RESULTS[tag] = {"created_type": created["classifier_type"], "created_llm_config": rubric(created), "reopened": s0,
                            "approach_after_llm_again": sv and sv["approach"], "saved_type": saved["classifier_type"], "saved_llm_config": rubric(saved)}
            print("R1", tag, json.dumps(RESULTS[tag])[:700])
        except Exception as e: print("R-ERR", tag, str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_{tag}.png")
def r5():
    for tag, src, a in [("r5cap", f"{P}_r1a_create_payload", "Capability"), ("r5fus", f"{P}_r1b_create_payload", "Fuse v2")]:
        try:
            purge(); n = f"{P}-{tag}"; api_post("/model/new", payload_body(src, n, drop_rubric=True)); created = full_cfg(n)
            reopen(n); s0 = st(f"{tag} reopened"); appr("Complexity"); st(f"{tag} complexity")
            snap(f"{P}_{tag}_implicit_reload_to_complexity", [L_appr(), L_family(), L_judge()])
            submit(f"{P}_{tag}_save_payload"); saved = full_cfg(n)
            RESULTS[tag] = {"created_type": created["classifier_type"], "created_llm_config": rubric(created), "reopened_approach": s0["approach"],
                            "saved_type": saved["classifier_type"], "saved_llm_config": rubric(saved)}
            print("R5", tag, json.dumps(RESULTS[tag]))
        except Exception as e: print("R-ERR", tag, str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_{tag}.png")
def r2():
    try:
        purge(); n = f"{P}-r2-implicit"; api_post("/model/new", payload_body(f"{P}_c2m_llm_openai_payload", n, drop_rubric=True)); created = full_cfg(n)
        reopen(n); st("r2 reopened")
        for a in ["Capability", "Complexity", "Fuse v2", "Complexity"]: d = appr(a); st(f"r2 {a} dialog={d}")
        snap(f"{P}_r2_implicit_roundtrip", [L_appr(), L_family(), L_judge()])
        submit(f"{P}_r2_save_payload"); saved = full_cfg(n)
        RESULTS["r2"] = {"created_type": created["classifier_type"], "created_llm_config": rubric(created), "saved_type": saved["classifier_type"], "saved_llm_config": rubric(saved)}
        print("R2", json.dumps(RESULTS["r2"]))
    except Exception as e: print("R-ERR r2", str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_r2.png")
def disabled_state():
    radios = D().get_by_role("radio").evaluate_all("els=>els.map(e=>(e.getAttribute('aria-label')||e.closest('label')?.innerText.split('\\n')[0]||e.id)+':'+(e.disabled||e.getAttribute('data-disabled')!==null))")
    return radios
def menu_items(btn):
    D().get_by_role("button", name=btn).first.click(); page.wait_for_timeout(500)
    it = page.get_by_role("menuitemradio").evaluate_all("els=>els.map(e=>e.innerText.split('\\n')[0]+':'+(e.getAttribute('aria-disabled')||e.getAttribute('data-disabled')||''))")
    return it
def force_item(item):
    try: page.get_by_role("menuitemradio", name=re.compile(item)).first.click(force=True, timeout=2000); page.wait_for_timeout(700)
    except Exception as e: print("force click blocked", item, str(e).split("\n")[0][:120])
    if page.get_by_role("menu").count(): page.keyboard.press("Escape"); page.wait_for_timeout(400)
    accept_any()
def r3():
    try:
        purge(); n = f"{P}-r3-custom"; open_add(n)
        menu("Heuristic", "Heuristic v2"); fam("LLM"); pick("Judge model", "claude-fable-5-1"); appr("Fuse v2")
        fam("Decisions Model"); pick("Decision model", "jev-decider"); st("r3 jev after fuse")
        add_custom_tier(); s1 = st("r3 jev custom")
        fam("LLM"); dlg = confirm_dialog(); s2 = st("r3 llm again")
        snap(f"{P}_r3a_llm_after_fuse_keeps_custom", [L_appr(), L_family(), D().locator("[aria-label^='Name for tier']").last])
        rad = disabled_state(); hr = D().get_by_role("radio", name="Heuristics")
        try: hr.first.click(force=True, timeout=2000); page.wait_for_timeout(700)
        except Exception as e: print("heur force blocked", str(e).split("\n")[0][:120])
        s3 = st("r3 after forced heuristics")
        items = menu_items("Routing approach"); snap(f"{P}_r3b_forecast_items_disabled", [page.get_by_role("menuitemradio", name=re.compile("Capability|Fuse"))])
        force_item("Fuse v2"); s4 = st("r3 after forced fuse")
        RESULTS["r3"] = {"jev_custom": s1, "dialog_on_llm": dlg, "llm_again": s2, "radios": rad, "after_forced_heur": s3, "approach_items": items, "after_forced_fuse": s4,
                         "save_note": "not submitted: LEGAL tier has no model, Save correctly blocked by custom tier validation; UI-state evidence only"}
        print("R3", json.dumps(RESULTS["r3"])[:1500])
    except Exception as e: print("R-ERR r3", str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_r3.png")
def r4():
    for tag, method in [("r4f", "Fuse v2"), ("r4c", "Capability")]:
        try:
            purge(); n = f"{P}-{tag}"; open_add(n)
            menu("Heuristic", "Heuristic v2"); fam("LLM"); pick("Judge model", "claude-fable-5-1"); appr(method)
            fam("Decisions Model"); pick("Decision model", "jev-decider"); s0 = st(f"{tag} jev with history")
            for hn, src in [("hv2", f"{P}_c3_payload"), ("cap", f"{P}_r1a_create_payload"), ("fus", f"{P}_r1b_create_payload")]:
                api_post("/model/new", payload_body(src, f"{P}-{tag}-holder-{hn}"))
            freq("Every new user message"); page.wait_for_timeout(2500)
            labels = D().get_by_text(re.compile("of 1 available|Used by this router|0 of")).all_inner_texts()[:8]
            fam("Heuristics"); sh = st(f"{tag} heuristics"); snap(f"{P}_{tag}_a_heuristics_not_v2", [L_family(), L_heur()])
            hitems = menu_items("Heuristic"); force_item("Heuristic v2"); shf = st(f"{tag} after forced v2")
            fam("LLM"); sl = st(f"{tag} llm"); snap(f"{P}_{tag}_b_llm_not_{method[:3]}", [L_family(), L_appr()])
            aitems = menu_items("Routing approach"); force_item(method); slf = st(f"{tag} after forced {method}")
            RESULTS[tag] = {"history_state": s0, "labels": labels, "heuristics": sh, "heur_items": hitems, "after_forced_v2": shf,
                            "llm": sl, "approach_items": aitems, "after_forced_method": slf}
            print("R4", tag, json.dumps(RESULTS[tag])[:1500])
        except Exception as e: print("R-ERR", tag, str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_{tag}.png")

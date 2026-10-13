AV = []
def _on_resp(r):
    if "/auto_router/availability" in r.url:
        try: AV.append({"t": time.strftime("%H:%M:%S"), "status": r.status, "req_classifier": (json.loads(r.request.post_data or "{}").get("complexity_router_config") or {}).get("classifier_type"), "body": r.json()})
        except Exception: pass
page.on("response", _on_resp)
REQLOG = []
page.on("request", lambda r: REQLOG.append((time.strftime("%H:%M:%S"), ((json.loads(r.post_data or "{}").get("complexity_router_config") or {}).get("classification_mode")))) if "/auto_router/availability" in r.url else None)
def av_of(key):
    for a in reversed(AV):
        for e in (a["body"] or {}).get("allowances", []):
            if e["key"] == key: return {"t": a["t"], **e}
def save_btn():
    b = D().get_by_role("button", name=re.compile("^(Add Auto Router|Save Changes)$")).last
    return {"disabled": b.is_disabled()}
FUSE = [("Efficient solver profile", "Short reasoning budget"), ("Capable solver profile", "Larger reasoning budget"), ("Harness and budget", "Shell and test runner, one attempt")]
def L_fuse(): return [D().get_by_label(l, exact=True) for l, _ in FUSE] + [D().get_by_label("Maximum quality gap", exact=True)]
def fill_fuse():
    for l, v in FUSE: D().get_by_label(l, exact=True).first.fill(v)
    q = D().get_by_label("Maximum quality gap")
    if not q.count(): q = D().get_by_text("Maximum quality gap", exact=True).locator("xpath=following::input[1]")
    q.first.fill("0.05"); page.wait_for_timeout(700)
def fuse_vals():
    out = {}
    for l, _ in FUSE + [("Maximum quality gap", "")]:
        c = D().get_by_label(l, exact=True); out[l] = c.first.input_value() if c.count() else "<none>"
    return out
def g1():
    for tag, via in [("g1a", None), ("g1b", "Heuristics"), ("g1c", "Decisions Model")]:
        try:
            purge(); n = f"{P}-{tag}"; open_add(n); fam("LLM"); pick("Judge model", "claude-fable-5-1"); dlg = confirm_dialog(); appr("Fuse v2"); fill_fuse()
            s = st(f"{tag} fuse filled"); sb = save_btn()
            snap(f"{P}_{tag}_a_fuse_filled", [L_appr()] + L_fuse())
            submit(f"{P}_{tag}_create_payload"); created = full_cfg(n)
            if not created: RESULTS[tag] = {"error": "fuse create not persisted", "save_btn": sb}; print("G1", tag, RESULTS[tag]); continue
            reopen(n); s0 = st(f"{tag} reopened"); fv = fuse_vals(); snap(f"{P}_{tag}_b_reopened_fuse", [L_appr()] + L_fuse())
            sv = None
            if via: fam(via); st(f"{tag} via {via}"); fam("LLM"); sv = st(f"{tag} llm again")
            cd = appr("Complexity"); s2 = st(f"{tag} complexity")
            snap(f"{P}_{tag}_c_reloaded_fuse_to_complexity", [L_appr(), L_family(), L_judge()])
            submit(f"{P}_{tag}_save_payload"); saved = full_cfg(n); persisted(n)
            RESULTS[tag] = {"via": via, "created_type": created["classifier_type"], "created_llm_v2_config": created.get("llm_v2_config"), "created_adaptive": created.get("adaptive"),
                            "created_tiers": created.get("tiers"), "created_llm_config": rubric(created), "reopened": s0, "reopened_fuse_fields": fv,
                            "approach_after_llm_again": sv and sv["approach"], "complexity_confirm_dialog": cd, "after_complexity": s2,
                            "saved_type": saved["classifier_type"], "saved_llm_config": rubric(saved), "saved_llm_v2_config": saved.get("llm_v2_config")}
            print("G1", tag, json.dumps(RESULTS[tag])[:900])
        except Exception as e: print("G-ERR", tag, str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_{tag}.png")
def holder(src, name):
    api_post("/model/new", payload_body(src, name)); c = full_cfg(name); print("HOLDER", name, c and c["classifier_type"]); return c and c["classifier_type"]
def holder_id(name):
    for m in all_models():
        if m["model_name"] == name: return m["model_info"]["id"]
def wait_exhausted(key, want=0):
    for i in range(40):
        r = page.request.post("http://localhost:4000/auto_router/availability", headers=H, data={"complexity_router_config": None})
        e = [x for x in r.json().get("allowances", []) if x["key"] == key]
        if e and e[0]["remaining"] == want: print("PROXY", key, "remaining", want, "after", i, "polls"); return e[0]
        page.wait_for_timeout(1000)
    print("PROXY never reached", key, want)
def g2():
    try:
        purge(); AV.clear(); n = f"{P}-g2-draft"; open_add(n)
        menu("Heuristic", "Heuristic v2"); s_v2 = st("g2 v2 selected")
        fam("LLM"); pick("Judge model", "claude-fable-5-1"); freq("Every new user message"); page.wait_for_timeout(2500); av0 = av_of("heuristic_v2")
        fam("Heuristics"); sA = st("g2 available: back to heuristics"); snap(f"{P}_g2a_available_restores_v2", [L_family(), L_heur()])
        fam("LLM"); page.wait_for_timeout(1500)
        ht = holder("t1392_c3_payload", f"{P}-g2-holder-hv2")
        freq("Once per session"); page.wait_for_timeout(3000); av1 = av_of("heuristic_v2")
        fam("Heuristics"); sE = st("g2 exhausted: back to heuristics"); sbE = save_btn()
        snap(f"{P}_g2b_exhausted_falls_back_rule_based", [L_family(), L_heur(), D().get_by_text(re.compile("0 of 1 available"))])
        hitems = menu_items("Heuristic"); snap(f"{P}_g2c_exhausted_v2_disabled_menu", [page.get_by_role("menuitemradio", name=re.compile("Heuristic v2"))])
        force_item("Heuristic v2"); sF = st("g2 after forced v2")
        fam("LLM"); sJ = st("g2 llm draft retained")
        api_post("/model/delete", {"id": holder_id(f"{P}-g2-holder-hv2")})
        freq("Every request"); page.wait_for_timeout(3000); av2 = av_of("heuristic_v2")
        fam("Heuristics"); sR = st("g2 after holder removed: back to heuristics")
        hitems2 = menu_items("Heuristic"); page.keyboard.press("Escape"); page.wait_for_timeout(400)
        if sR["heur"] != "Heuristic v2": menu("Heuristic", "Heuristic v2"); sR2 = st("g2 v2 reselected after removal")
        else: sR2 = sR
        snap(f"{P}_g2d_available_after_holder_removed", [L_family(), L_heur()])
        submit(f"{P}_g2_save_payload"); saved = full_cfg(n); persisted(n)
        RESULTS["g2"] = {"v2_selected": s_v2, "avail_before": av0, "available_restore": sA, "holder_type": ht, "avail_exhausted": av1, "exhausted_restore": sE,
                         "save_btn_exhausted": sbE, "heur_menu_exhausted": hitems, "after_forced_v2": sF, "llm_draft_after": sJ, "avail_after_removal": av2,
                         "restore_after_removal": sR, "heur_menu_after_removal": hitems2, "final_state": sR2, "saved_type": saved and saved["classifier_type"],
                         "saved_mode": saved and saved.get("classification_mode")}
        print("G2", json.dumps(RESULTS["g2"])[:2500])
    except Exception as e: print("G-ERR g2", str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_g2.png")
def g3():
    try:
        purge(); AV.clear(); n = f"{P}-g3-draft"; open_add(n)
        fam("LLM"); pick("Judge model", "claude-fable-5-1"); appr("Fuse v2"); fill_fuse(); freq("Every new user message"); s0 = st("g3 fuse draft")
        fam("Heuristics"); page.wait_for_timeout(2500); fam("LLM"); page.wait_for_timeout(800); av0 = av_of("llm_v2"); sA = st("g3 available: back to llm"); fv0 = fuse_vals()
        snap(f"{P}_g3a_available_restores_fuse", [L_family(), L_appr()] + L_fuse())
        fam("Heuristics"); page.wait_for_timeout(1500)
        ht = holder(f"{P}_g1a_create_payload", f"{P}-g3-holder-fuse"); px = wait_exhausted("llm_v2", 0)
        n_before = len(AV); tpl("OpenAI"); av1 = None
        for i in range(20):
            page.wait_for_timeout(1000); new = AV[n_before:]
            av1 = next((dict(t=a["t"], req_classifier=a["req_classifier"], **e) for a in reversed(new) for e in a["body"].get("allowances", []) if e["key"] == "llm_v2"), None)
            if av1 and av1["remaining"] == 0: break
        print("REFRESH", REQLOG[-3:], av1); sT = st("g3 heuristics after template (valid away draft)")
        snap(f"{P}_g3a2_heuristics_valid_draft_refetched", [L_family(), L_tpl()])
        fam("LLM"); page.wait_for_timeout(800); sE = st("g3 exhausted: back to llm"); sbE = save_btn()
        snap(f"{P}_g3b_exhausted_falls_back_complexity", [L_family(), L_appr(), L_judge(), L_freq()])
        aitems = menu_items("Routing approach"); snap(f"{P}_g3c_exhausted_fuse_disabled_menu", [page.get_by_role("menuitemradio", name=re.compile("Fuse"))])
        force_item("Fuse v2"); sF = st("g3 after forced fuse")
        page.wait_for_timeout(1500); sbE2 = save_btn()
        submit(f"{P}_g3_save_payload"); saved = full_cfg(n); persisted(n)
        api_post("/model/delete", {"id": holder_id(f"{P}-g3-holder-fuse")})
        r = page.request.post("http://localhost:4000/auto_router/availability", headers=H, data={"complexity_router_config": saved}); av2 = [e for e in r.json()["allowances"] if e["key"] == "llm_v2"]
        RESULTS["g3"] = {"fuse_draft": s0, "avail_before": av0, "available_restore": sA, "fuse_fields_after_available_restore": fv0, "holder_type": ht, "proxy_exhausted": px, "avail_exhausted": av1,
                         "exhausted_restore": sE, "save_btn_exhausted": sbE, "approach_menu_exhausted": aitems, "after_forced_fuse": sF, "save_btn_before_submit": sbE2,
                         "saved_type": saved and saved["classifier_type"], "saved_llm_config": rubric(saved), "saved_mode": saved and saved.get("classification_mode"),
                         "saved_has_llm_v2_config": bool(saved and saved.get("llm_v2_config")), "avail_after_holder_removed_api": av2}
        print("G3", json.dumps(RESULTS["g3"])[:2500])
    except Exception as e: print("G-ERR g3", str(e).split("\n")[0][:300]); page.screenshot(path=f"{OUT}/err_g3.png")

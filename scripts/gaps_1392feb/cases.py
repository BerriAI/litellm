P = PFX
def c1():
    n=f"{P}-c1-jev-openai"; open_add(n); fam("Decisions Model"); pick("Decision model","jev-decider"); freq("Every new user message")
    st("C1 before"); snap(f"{P}_c1a_before", [L_family(), L_dec(), L_freq()])
    tpl("OpenAI Family"); st("C1 after-openai"); snap(f"{P}_c1b_after_openai", [L_family(), L_dec(), L_freq(), L_tpl()])
    submit(f"{P}_c1_payload"); persisted(n); reopen(n); st("C1 reopened"); snap(f"{P}_c1c_reopened", [L_family(), L_dec(), L_freq()])
def c2():
    n=f"{P}-c2-llm-tpls"; open_add(n); fam("LLM"); j=pick("Judge model","claude-fable-5-1"); freq("Every new user message"); st("C2 llm before")
    for t in ["OpenAI Family","Gemini Family","Anthropic Family"]:
        tpl(t); st(f"C2 llm after {t}")
    snap(f"{P}_c2a_llm_after_o_g_a", [L_family(), L_judge(), L_freq(), L_tpl()]); submit(f"{P}_c2_llm_payload"); persisted(n)
    n=f"{P}-c2-heur-tpls"; open_add(n); menu("Heuristic","Heuristic v2"); freq("Every new user message"); st("C2 heur before")
    for t in ["1M Context","Lite"]:
        tpl(t); st(f"C2 heur after {t}")
    snap(f"{P}_c2b_heurv2_after_1m_lite", [L_family(), L_heur(), L_freq(), L_tpl()]); submit(f"{P}_c2_heur_payload"); persisted(n)
def c3():
    n=f"{P}-c3-roundtrip"; open_add(n)
    fam("Decisions Model"); pick("Decision model","jev-decider"); fam("LLM"); pick("Judge model","claude-fable-5-1"); fam("Heuristics"); menu("Heuristic","Heuristic v2"); st("C3 set all")
    fam("Decisions Model"); s1=st("C3 back jev"); snap(f"{P}_c3a_back_to_jev", [L_family(), L_dec()])
    fam("LLM"); st("C3 back llm"); snap(f"{P}_c3b_back_to_llm", [L_family(), L_judge()])
    fam("Heuristics"); st("C3 back heur"); snap(f"{P}_c3c_back_to_heur", [L_family(), L_heur()])
    tpl("OpenAI Family"); st("C3 heur+openai"); submit(f"{P}_c3_payload"); persisted(n)
def c4():
    n=f"{P}-c4-custom"; open_add(n); tpl("OpenAI Family"); fam("LLM"); pick("Judge model","claude-fable-5-1"); freq("Every new user message")
    D().get_by_role("button", name="Edit tiers").click(); page.wait_for_timeout(600)
    D().get_by_role("button", name="Add tier").click(); page.wait_for_timeout(600)
    D().locator("[aria-label^='Name for tier']").last.fill("LEGAL"); D().locator("[aria-label^='Definition for tier']").last.fill("Legal questions"); page.wait_for_timeout(500)
    s=st("C4 custom"); hint = D().get_by_text(re.compile("template", re.I)).all_inner_texts(); print("C4 template-related text:", [h[:120] for h in hint][:6])
    snap(f"{P}_c4a_custom_template_state", [L_tpl(), D().locator("[aria-label^='Name for tier']").last, L_family()])
    try:
        L_tpl().first.click(timeout=3000); page.wait_for_timeout(500); opts=page.get_by_role("option"); print("C4 opts", opts.evaluate_all("els=>els.map(e=>e.innerText.split('\\n')[0]+':'+(e.getAttribute('aria-disabled')||''))"))
        o=page.get_by_role("option", name="Gemini Family")
        if o.count(): o.first.click(timeout=3000); page.wait_for_timeout(1000)
        else: page.keyboard.press("Escape")
    except Exception as e: print("C4 template click blocked:", str(e).split("\n")[0][:150]); page.keyboard.press("Escape")
    st("C4 after template attempt"); snap(f"{P}_c4b_after_template_attempt", [L_tpl(), L_family()])
    rb=D().get_by_role("button", name="Restore defaults")
    if not rb.count():
        e=D().get_by_role("button", name="Edit tiers")
        if e.count(): e.click(); page.wait_for_timeout(500)
    if rb.count():
        rb.first.click(); page.wait_for_timeout(800); cd=confirm_dialog(); print("C4 restore confirm:", cd)
        if cd: page.locator("[role=alertdialog]").get_by_role("button").last.click(); page.wait_for_timeout(800)
        st("C4 restored"); tpl("Gemini Family"); st("C4 restored+gemini"); snap(f"{P}_c4c_restored_gemini", [L_tpl(), L_family(), L_judge(), L_freq()])
    else: print("C4 no Restore defaults button")
    tpl("Start from scratch"); st("C4 scratch"); snap(f"{P}_c4d_start_from_scratch", [L_tpl(), L_family(), L_freq()])
def c5():
    for appr in ["Capability","Fuse v2"]:
        n=f"{P}-c5-{appr[:3].lower()}"; open_add(n); tpl("OpenAI Family"); fam("LLM"); pick("Judge model","claude-fable-5-1"); freq("Every new user message"); st(f"C5 {appr} before")
        menu("Routing approach", appr); cd=confirm_dialog(); print(f"C5 {appr} confirm:", cd); st(f"C5 {appr} after-select")
        if cd:
            snap(f"{P}_c5_{appr[:3]}_a_confirm", [page.locator("[role=alertdialog]")])
            page.locator("[role=alertdialog]").get_by_role("button", name=re.compile("cancel|keep", re.I)).first.click(); page.wait_for_timeout(800); st(f"C5 {appr} cancelled")
            menu("Routing approach", appr); page.locator("[role=alertdialog]").get_by_role("button").last.click(); page.wait_for_timeout(1000)
        st(f"C5 {appr} applied"); snap(f"{P}_c5_{appr[:3]}_b_applied", [L_appr(), L_family()])
        menu("Routing approach", "Complexity"); cd2=confirm_dialog(); print("C5 back confirm:", cd2)
        if cd2: page.locator("[role=alertdialog]").get_by_role("button").last.click(); page.wait_for_timeout(800)
        st(f"C5 {appr} back-complexity"); snap(f"{P}_c5_{appr[:3]}_c_back", [L_appr(), L_judge(), L_freq(), L_tpl()])

def c2lite():
    n=f"{P}-c2-heur-lite"; open_add(n); menu("Heuristic","Heuristic v2"); freq("Every new user message"); st("C2L heur before")
    L_tpl().first.click(); page.wait_for_timeout(500); print("C2L opts", page.get_by_role("option").evaluate_all("els=>els.map(e=>e.innerText.replace(/\\n/g,' | ').slice(0,90)+':'+(e.getAttribute('aria-disabled')||''))")); page.keyboard.press("Escape"); page.wait_for_timeout(300)
    tpl("1M Context"); st("C2L after 1M"); tpl("Lite"); st("C2L after Lite")
    snap(f"{P}_c2c_heurv2_after_1m_lite", [L_family(), L_heur(), L_freq(), L_tpl()]); submit(f"{P}_c2l_payload"); persisted(n)

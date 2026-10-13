import json
G = {"PFX": "t5927"}
for f in ["boot","lib"]: exec(open(f"/home/ubuntu/qa_ar/{f}.py").read(), G)
G["OUT"] = "/home/ubuntu/qa_ar/shots_5927"
for f in ["cases","tip_extra"]: exec(open(f"/home/ubuntu/qa_ar/{f}.py").read(), G)
exec("""
login(); open_add(f"{P}-c4-open"); tpl("OpenAI Family"); fam("LLM"); pick("Judge model","claude-fable-5-1"); freq("Every new user message")
D().get_by_role("button", name="Edit tiers").click(); page.wait_for_timeout(600)
D().get_by_role("button", name="Add tier").click(); page.wait_for_timeout(600)
D().locator("[aria-label^='Name for tier']").last.fill("LEGAL"); D().locator("[aria-label^='Definition for tier']").last.fill("Legal questions"); page.wait_for_timeout(500)
L_tpl().first.scroll_into_view_if_needed(); L_tpl().first.click(); page.wait_for_timeout(700)
lb = page.get_by_role("listbox")
print("LISTBOX", json.dumps(lb.first.inner_text() if lb.count() else None))
outline([lb] if lb.count() else [page.get_by_role("option")]); page.screenshot(path=f"{OUT}/{P}_c4e_custom_template_menu_open.png"); outline([])
page.keyboard.press("Escape")
""", G)

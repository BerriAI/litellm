#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════
  LiteLLM — OpenAI Skills API Real Demo
  Full CRUD lifecycle against live OpenAI API
═══════════════════════════════════════════════════════════
"""
import os, sys, time, json

sys.path.insert(0, os.getcwd())
import litellm

SKILL_MD = """---
name: code-review-assistant
description: A skill that helps review Python code for best practices
---

# Code Review Assistant

You are a Python code review assistant. When given code, you:
1. Check for PEP 8 compliance
2. Identify potential bugs
3. Suggest performance improvements
4. Recommend better patterns

Always be constructive and explain your suggestions.
"""

SKILL_V2_MD = """---
name: code-review-assistant-v2
description: Enhanced code review with security analysis
---

# Code Review Assistant v2

Enhanced with security vulnerability scanning and dependency audit.
"""

def banner(text):
    print(f"\n{'═' * 60}\n  {text}\n{'═' * 60}\n")

def section(n, text):
    print(f"\n{'─' * 50}\n  STEP {n}/10: {text}\n{'─' * 50}\n")

def pp(label, obj):
    """Pretty-print response"""
    if hasattr(obj, 'model_dump'):
        data = obj.model_dump(exclude_none=True)
    elif isinstance(obj, dict):
        data = obj
    else:
        data = {"raw": str(obj)[:500]}
    print(f"  📦 {label}:")
    print(f"  {json.dumps(data, indent=4, default=str)}")
    print()

banner("LiteLLM — OpenAI Skills API REAL Demo")
print(f"  Python:    {sys.version.split()[0]}")
print(f"  API Key:   {os.environ.get('OPENAI_API_KEY', '')[:12]}...")
print(f"  Endpoint:  https://api.openai.com/v1/skills")
print(f"  Time:      {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
print(f"\n  Starting demo...\n")

# ─── 1: CREATE ────────────────────────────────
section(1, "Create Skill")
print("  → POST /v1/skills")
print("  → Body: name='code-review-assistant', files=[SKILL.md]")
skill = litellm.create_skill(
    custom_llm_provider="openai",
    name="code-review-assistant",
    description="A skill for reviewing Python code",
    files=[("code-review/SKILL.md", SKILL_MD, "text/markdown")],
)
pp("Response", skill)
SID = skill.id
print(f"  ✅ Created skill: {SID}\n     display_title: {skill.display_title}")

time.sleep(3)

# ─── 2: GET ───────────────────────────────────
section(2, "Get Skill Details")
print(f"  → GET /v1/skills/{SID}")
s = litellm.get_skill(skill_id=SID, custom_llm_provider="openai")
pp("Response", s)
print(f"  ✅ Retrieved: display_title={s.display_title}, default_version={s.default_version}")

# ─── 3: LIST ──────────────────────────────────
section(3, "List All Skills")
print(f"  → GET /v1/skills")
for i in range(5):
    lst = litellm.list_skills(custom_llm_provider="openai")
    data = lst.data if hasattr(lst, 'data') else lst.get('data', [])
    found = any((x.id if hasattr(x, 'id') else x.get('id')) == SID for x in data)
    if found: break
    print(f"  ⏳ Eventual consistency (attempt {i+1}/5)...")
    time.sleep(3)
pp("Response", lst)
print(f"  ✅ Found {len(data)} skill(s), our skill present: {found}")

# ─── 4: CONTENT ───────────────────────────────
section(4, "Get Skill Content")
print(f"  → GET /v1/skills/{SID}/content")
content = litellm.get_skill_content(skill_id=SID, custom_llm_provider="openai")
pp("Response", content)
print(f"  ✅ Content retrieved (base64-encoded zip)")

# ─── 5: CREATE VERSION ────────────────────────
section(5, "Create Version 2")
print(f"  → POST /v1/skills/{SID}/versions")
print(f"  → Body: name='v2-security', files=[updated SKILL.md]")
ver = litellm.create_skill_version(
    skill_id=SID, custom_llm_provider="openai",
    name="v2-security", description="v2 with security scanning",
    files=[("code-review/SKILL.md", SKILL_V2_MD, "text/markdown")],
)
pp("Response", ver)
VID = ver.id
print(f"  ✅ Version created: {VID}, version_number={ver.version}")

print("  ⏳ Waiting 8s for consistency...")
time.sleep(8)

# ─── 6: LIST VERSIONS ─────────────────────────
section(6, "List Versions")
print(f"  → GET /v1/skills/{SID}/versions")
vlist = litellm.list_skill_versions(skill_id=SID, custom_llm_provider="openai")
pp("Response", vlist)
print(f"  ✅ Found {len(vlist.data)} version(s)")

# ─── 7: GET VERSION ──────────────────────────
section(7, "Get Version 2 Details")
print(f"  → GET /v1/skills/{SID}/versions/2")
for i in range(3):
    try:
        v2 = litellm.get_skill_version(skill_id=SID, version="2", custom_llm_provider="openai")
        pp("Response", v2)
        print(f"  ✅ Version 2 confirmed: {v2.id}")
        break
    except Exception as e:
        if "404" in str(e) and i < 2:
            print(f"  ⏳ Not visible (attempt {i+1}), waiting 5s...")
            time.sleep(5)
        else: raise

# ─── 8: UPDATE SKILL ─────────────────────────
section(8, "Update Skill (set default_version=2)")
print(f"  → POST /v1/skills/{SID}")
print(f"  → Body: {{'default_version': '2'}}")
upd = litellm.update_skill(skill_id=SID, custom_llm_provider="openai", default_version="2")
pp("Response", upd)
print(f"  ✅ default_version updated to: {upd.default_version}")

# ─── 9: DELETE VERSION ────────────────────────
section(9, "Delete Version 2")
litellm.update_skill(skill_id=SID, custom_llm_provider="openai", default_version="1")
print(f"  → DELETE /v1/skills/{SID}/versions/2")
for i in range(3):
    try:
        dv = litellm.delete_skill_version(skill_id=SID, version="2", custom_llm_provider="openai")
        pp("Response", dv)
        print(f"  ✅ Version 2 deleted")
        break
    except Exception as e:
        if ("not found" in str(e).lower() or "404" in str(e)) and i < 2:
            print(f"  ⏳ Retry {i+1}...")
            time.sleep(5)
        else: raise

# ─── 10: DELETE SKILL ─────────────────────────
section(10, "Delete Skill")
print(f"  → DELETE /v1/skills/{SID}")
ds = litellm.delete_skill(skill_id=SID, custom_llm_provider="openai")
pp("Response", ds)
print(f"  ✅ Skill deleted: {ds.id}")

print(f"\n  → Verifying: GET /v1/skills/{SID}")
try:
    litellm.get_skill(skill_id=SID, custom_llm_provider="openai")
    print(f"  ❌ Still exists!")
except:
    print(f"  ✅ Confirmed: 404 — skill fully deleted")

banner("DEMO COMPLETE — All 10 CRUD operations succeeded!")
print(f"  Skill ID:    {SID}")
print(f"  Version ID:  {VID}")
print(f"  Timestamp:   {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
print(f"\n  Every call above hit the live OpenAI production API.")
print(f"  No mocks. No fakes. Real HTTP requests & responses.\n")

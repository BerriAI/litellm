#!/usr/bin/env python3
"""
OpenAI Skills API — REAL E2E Proof Test
========================================
Performs REAL API calls against the live OpenAI Skills API.
No mocks. No fakes. Every call hits the production endpoint.

Tests the full CRUD lifecycle:
  1. Create Skill
  2. Get Skill
  3. List Skills
  4. Update Skill (set default_version)
  5. Get Skill Content
  6. Create Version
  7. List Versions
  8. Get Version
  9. Delete Version
  10. Delete Skill

Uses the LiteLLM SDK (with bug fixes applied).
"""
import os, sys, time, traceback

sys.path.insert(0, os.getcwd())
import litellm

SKILL_CONTENT = """---
name: e2e-proof-skill
description: E2E proof test skill created by automated test
---

# E2E Proof Skill

This skill was created by an automated E2E test to prove the OpenAI Skills API
integration works correctly with the LiteLLM SDK.
"""

SKILL_V2_CONTENT = """---
name: e2e-proof-skill-v2
description: Version 2 of the E2E proof test skill
---

# E2E Proof Skill — Version 2

Updated content for version 2 of the E2E proof test.
"""

print("=" * 70)
print("  OpenAI Skills API — REAL E2E Proof Test")
print("  Using LiteLLM SDK with bug fixes applied")
print("=" * 70)
print()
print(f"  Branch:    feat/m1-openai-skills-config")
print(f"  Python:    {sys.version.split()[0]}")
print(f"  API Key:   {os.environ.get('OPENAI_API_KEY', 'NOT SET')[:12]}...")
print(f"  Provider:  OpenAI (live production API)")
print(f"  Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
print()
print("=" * 70)
print()

results = []
skill_id = None
version_id = None

def step(num, total, name, fn):
    print(f"\n{'─' * 60}")
    print(f"  Step {num}/{total}: {name}")
    print(f"{'─' * 60}")
    try:
        result = fn()
        results.append((name, "PASS", result))
        print(f"\n  ✅ PASSED: {name}")
        return result
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"\n  ❌ FAILED: {name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return None

# ── Step 1: Create Skill ──
def create_skill():
    global skill_id
    print("  → litellm.create_skill(custom_llm_provider='openai', ...)")
    print("  → Uploading SKILL.md with YAML front matter...")
    skill = litellm.create_skill(
        custom_llm_provider="openai",
        name="e2e-proof-test",
        description="E2E proof test — created by automated test",
        files=[("e2e-proof/SKILL.md", SKILL_CONTENT, "text/markdown")],
    )
    skill_id = skill.id
    print(f"  ← Response type: {type(skill).__name__}")
    print(f"     id:              {skill.id}")
    print(f"     display_title:   {skill.display_title}")
    print(f"     default_version: {skill.default_version}")
    print(f"     source:          {skill.source}")
    assert skill.id.startswith("skill_"), f"Unexpected ID format: {skill.id}"
    return skill

step(1, 10, "Create Skill", create_skill)
if not skill_id:
    print("\n\nFATAL: Cannot continue without a skill ID")
    sys.exit(1)

time.sleep(3)

# ── Step 2: Get Skill ──
def get_skill():
    print(f"  → litellm.get_skill(skill_id='{skill_id}')")
    skill = litellm.get_skill(skill_id=skill_id, custom_llm_provider="openai")
    print(f"  ← Response type: {type(skill).__name__}")
    print(f"     id:              {skill.id}")
    print(f"     display_title:   {skill.display_title}")
    print(f"     default_version: {skill.default_version}")
    assert skill.id == skill_id, f"ID mismatch: {skill.id} != {skill_id}"
    print(f"  ✓ ID matches expected value")
    return skill

step(2, 10, "Get Skill", get_skill)

# ── Step 3: List Skills ──
def list_skills():
    print(f"  → litellm.list_skills(custom_llm_provider='openai')")
    print(f"  → With retry for eventual consistency...")
    for attempt in range(5):
        skills = litellm.list_skills(custom_llm_provider="openai")
        skill_list = skills.data if hasattr(skills, 'data') else skills.get('data', [])
        found = any(s.id == skill_id if hasattr(s, 'id') else s.get('id') == skill_id for s in skill_list)
        print(f"     Attempt {attempt+1}: found={found}, total={len(skill_list)}")
        if found:
            print(f"  ← Our skill found in list!")
            return skills
        time.sleep(3)
    raise Exception(f"Skill {skill_id} not found after 5 attempts")

step(3, 10, "List Skills", list_skills)

# ── Step 4: Update Skill ──
def update_skill():
    print(f"  → litellm.update_skill(skill_id='{skill_id}', default_version='1')")
    updated = litellm.update_skill(
        skill_id=skill_id, custom_llm_provider="openai", default_version="1",
    )
    print(f"  ← Response type: {type(updated).__name__}")
    if hasattr(updated, 'id'):
        print(f"     id:              {updated.id}")
        print(f"     default_version: {updated.default_version}")
    else:
        print(f"     response: {updated}")
    print(f"  ✓ Update succeeded")
    return updated

step(4, 10, "Update Skill", update_skill)

# ── Step 5: Get Skill Content ──
def get_content():
    print(f"  → litellm.get_skill_content(skill_id='{skill_id}')")
    content = litellm.get_skill_content(skill_id=skill_id, custom_llm_provider="openai")
    print(f"  ← Response type: {type(content).__name__}")
    content_str = str(content)
    preview = content_str[:200].replace('\n', '\\n')
    print(f"     preview: {preview}")
    print(f"  ✓ Content retrieved successfully")
    return content

step(5, 10, "Get Skill Content", get_content)

# ── Step 6: Create Version ──
def create_version():
    global version_id
    print(f"  → litellm.create_skill_version(skill_id='{skill_id}')")
    print(f"  → Uploading updated SKILL.md (v2)...")
    version = litellm.create_skill_version(
        skill_id=skill_id, custom_llm_provider="openai",
        name="v2-e2e-proof", description="Version 2 created by E2E proof test",
        files=[("e2e-proof/SKILL.md", SKILL_V2_CONTENT, "text/markdown")],
    )
    version_id = version.id
    print(f"  ← Response type: {type(version).__name__}")
    print(f"     id:       {version.id}")
    print(f"     version:  {version.version}")
    print(f"     skill_id: {version.skill_id}")
    assert version.id.startswith("skillver_"), f"Unexpected ID: {version.id}"
    print(f"  ✓ Version created (version={version.version})")
    return version

step(6, 10, "Create Version", create_version)
if not version_id:
    print("\n  ⚠ Version creation failed, skipping version steps")

print("  → Waiting 8s for eventual consistency...")
time.sleep(8)

# ── Step 7: List Versions ──
def list_versions():
    print(f"  → litellm.list_skill_versions(skill_id='{skill_id}')")
    versions_resp = litellm.list_skill_versions(skill_id=skill_id, custom_llm_provider="openai")
    print(f"  ← Response type: {type(versions_resp).__name__}")
    if isinstance(versions_resp, dict):
        vlist = versions_resp.get('data', [])
    else:
        vlist = versions_resp.data
    print(f"     count: {len(vlist)}")
    for v in vlist:
        vid = v.get('id', '') if isinstance(v, dict) else v.id
        vver = v.get('version', '') if isinstance(v, dict) else v.version
        print(f"     - id={vid}, version={vver}")
    print(f"  ✓ Versions listed successfully")
    return versions_resp

if version_id:
    step(7, 10, "List Versions", list_versions)
else:
    results.append(("List Versions", "SKIP", ""))
    print(f"\n  ⏭ Step 7/10: SKIPPED")

# ── Step 8: Get Version ──
def get_version():
    print(f"  → litellm.get_skill_version(skill_id='{skill_id}', version='2')")
    print(f"  → With retry for eventual consistency...")
    for attempt in range(3):
        try:
            version = litellm.get_skill_version(
                skill_id=skill_id, version="2", custom_llm_provider="openai",
            )
            print(f"  ← Response type: {type(version).__name__}")
            print(f"     id:       {version.id}")
            print(f"     version:  {version.version}")
            print(f"     skill_id: {version.skill_id}")
            assert version.version == "2", f"Expected '2', got '{version.version}'"
            print(f"  ✓ Version details confirmed (attempt {attempt+1})")
            return version
        except Exception as e:
            if "404" in str(e) and attempt < 2:
                print(f"     Attempt {attempt+1}: 404 — version not yet visible, retrying in 5s...")
                time.sleep(5)
            else:
                raise

if version_id:
    step(8, 10, "Get Version", get_version)
else:
    results.append(("Get Version", "SKIP", ""))
    print(f"\n  ⏭ Step 8/10: SKIPPED")

# ── Step 9: Delete Version ──
def delete_version():
    print(f"  → litellm.delete_skill_version(skill_id='{skill_id}', version='2')")
    print(f"  → With retry for eventual consistency...")
    for attempt in range(3):
        try:
            result = litellm.delete_skill_version(
                skill_id=skill_id, version="2", custom_llm_provider="openai",
            )
            print(f"  ← Response type: {type(result).__name__}")
            print(f"     result: {result}")
            print(f"  ✓ Version deleted (attempt {attempt+1})")
            return result
        except Exception as e:
            if ("404" in str(e) or "not found" in str(e).lower()) and attempt < 2:
                print(f"     Attempt {attempt+1}: not found yet, retrying in 5s...")
                time.sleep(5)
            else:
                raise

if version_id:
    step(9, 10, "Delete Version", delete_version)
else:
    results.append(("Delete Version", "SKIP", ""))
    print(f"\n  ⏭ Step 9/10: SKIPPED")

# ── Step 10: Delete Skill ──
def delete_skill():
    print(f"  → litellm.delete_skill(skill_id='{skill_id}')")
    result = litellm.delete_skill(skill_id=skill_id, custom_llm_provider="openai")
    print(f"  ← Response type: {type(result).__name__}")
    print(f"     id:   {result.id}")
    print(f"     type: {result.type}")
    print(f"  ✓ Skill deleted")

    print(f"\n  → Verifying deletion (expect 404)...")
    try:
        litellm.get_skill(skill_id=skill_id, custom_llm_provider="openai")
        raise Exception("Skill still exists after deletion!")
    except Exception as e:
        if "404" in str(e) or "Not Found" in str(e) or "status_code" in str(e):
            print(f"  ← Got expected error: {type(e).__name__}")
            print(f"  ✓ Skill confirmed deleted (404)")
        else:
            raise
    return result

step(10, 10, "Delete Skill", delete_skill)

# ── Summary ──
print()
print("=" * 70)
print("  FINAL RESULTS")
print("=" * 70)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
skipped = sum(1 for _, s, _ in results if s == "SKIP")

for name, status, _ in results:
    icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏭"
    print(f"  {icon} {name}: {status}")

print()
print(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
print()
if failed == 0:
    print("  🎉 ALL TESTS PASSED — E2E proof complete!")
else:
    print(f"  ⚠ {failed} test(s) FAILED")
print()
print(f"  Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)

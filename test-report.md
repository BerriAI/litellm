# Test Report — PR #33449

Reusable `BetaBadge` for Projects sidebar item + shared `disableShowBadges` suppression flag

## How I tested
Ran the LiteLLM proxy locally (:4000, Postgres + `STORE_MODEL_IN_DB=True`) and the dashboard dev server (:3000), logged in as admin. Enabled `enable_projects_ui` and `enable_chat_ui` via `PATCH /update/ui_settings` so both the new Beta badge (Projects) and an existing New badge (Chat) are visible, then exercised the account-menu "Hide Feature Badges" toggle end-to-end in the browser.

## Results

- Test 1 — Projects shows gold "Beta" badge, Chat shows blue "New" (Projects is NOT "New"): PASSED
- Test 2 — Enabling "Hide Feature Badges" hides BOTH the Projects Beta and Chat New badges, labels intact: PASSED
- Test 3 — Disabling the toggle restores BOTH badges: PASSED

No failures or unexpected behavior. The renamed toggle carries the correct aria-label `Toggle hide feature badges` and localStorage key `disableShowBadges`, confirmed in the DOM.

## Evidence

### Baseline: Projects = gold "Beta", Chat = blue "New"
![baseline badges](/home/ubuntu/screenshots/ss_zoom_d3d72f6f.png)

### "Hide Feature Badges" ON — both badges gone, labels remain
![badges hidden](/home/ubuntu/screenshots/ss_zoom_f5a45fc8.png)

### Toggle OFF — both badges restored
![badges restored](/home/ubuntu/screenshots/ss_zoom_c40d80d0.png)

### Account menu with renamed "Hide Feature Badges" toggle
![account menu](/home/ubuntu/screenshots/ss_48d9f27a.png)

## Notes
- The Projects/Chat sidebar items are feature-flag gated (`enable_projects_ui` / `enable_chat_ui`); they only render when those UI settings are on. I enabled them for testing via the intended master-key API path.
- Unit tests for the renamed hook and both badge components pass (72 tests) and `make pre-commit` is clean.

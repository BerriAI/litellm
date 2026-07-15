# Test Plan — PR #33449: reusable BetaBadge + shared `disableShowBadges` flag

## What changed (user-visible)
- Projects sidebar item (ACCESS CONTROL) now shows a gold **Beta** badge (was blue **New**, then removed).
- New reusable `BetaBadge` component at `src/components/BetaBadge.tsx`.
- The badge-suppression preference was renamed `disableShowNewBadge` -> `disableShowBadges`
  (hook `useDisableShowNewBadge` -> `useDisableShowBadges`) and now applies to BOTH `NewBadge` and `BetaBadge`.
- Account menu toggle relabeled "Hide New Feature Indicators" -> **"Hide Feature Badges"**.

## Environment (setup already done)
- Proxy on :4000 (Postgres + STORE_MODEL_IN_DB=True), dashboard dev server on :3000, logged in as admin.
- `enable_projects_ui` and `enable_chat_ui` toggled on via `PATCH /update/ui_settings` so both badges are visible.

## Code grounding
- `leftnav.tsx:211-220` Projects entry renders `<BetaBadge />`; `:446-447` gates projects/chat on flags.
- `BetaBadge.tsx` / `common_components/NewBadge.tsx` call `useDisableShowBadges()`; when true, render children (or null) with no badge.
- Toggle lives in account menu (`SidebarAccountMenu.tsx` / `UserDropdown.tsx`), key `disableShowBadges`, label "Hide Feature Badges".

## Tests

### Test 1 — Beta badge present, not New (baseline)
Steps: Observe the ACCESS CONTROL group in the left sidebar.
Pass/fail:
- PASS: "Projects" has a **gold "Beta"** pill next to it (DOM `<sup title="Beta">Beta</sup>`).
- Also confirm "Chat" (AI GATEWAY) shows a **blue "New"** pill.
- FAIL if Projects shows "New", shows no badge, or badge is not gold.

### Test 2 — "Hide Feature Badges" suppresses BOTH badges (the core change)
Steps:
1. Click the "Account / Admin" button at the bottom of the sidebar to open the account menu.
2. Locate the toggle labeled **"Hide Feature Badges"** and turn it ON.
3. Observe the sidebar.
Pass/fail:
- PASS: Both the Projects **Beta** badge AND the Chat **New** badge disappear, while the "Projects" and "Chat" text labels remain visible.
- FAIL if either badge remains, or if the nav labels/text vanish.
- Adversarial note: a broken impl (BetaBadge not wired to the shared flag) would keep the Beta badge visible while New disappears — visibly different, so this step distinguishes working vs broken.

### Test 3 — Toggling OFF restores both badges
Steps: Turn the "Hide Feature Badges" toggle OFF.
Pass/fail:
- PASS: Both Beta (Projects) and New (Chat) badges reappear.
- FAIL if either stays hidden.

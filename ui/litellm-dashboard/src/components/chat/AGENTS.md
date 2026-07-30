# Chat UI — Agent Instructions

Read `design.md` first. That file is the _what_ (tokens, components, patterns). This file is the _when and how_: decision trees for ambiguous cases, and the checklist to run before you're done.

This directory has drifted from its own spec before — a hand-rolled `<button>` sidebar nav item went through five different ad-hoc color attempts in one session (invisible `bg-accent`, a barely-visible shadow, invisible `bg-primary/10`, a `border-l-2` that clipped at the border-radius corner) before landing on the actual fix: use `bg-sidebar-accent`, a token that already existed and was never checked. Every rule below exists to stop that specific failure mode: **reaching for a plausible-looking Tailwind class instead of checking what the design system already provides.**

---

## Decision tree: "I need a new UI element"

1. **Does `design.md` already show this exact pattern?** Copy it verbatim. Don't restyle it "to match" something else you're looking at — if it needs to look different, that's a `design.md` update, not a one-off deviation.
2. **Is it a clickable thing?** It's a `Button` (some variant/size), full stop. Never a raw `<button>`. If no variant looks right, that's a sign to re-read the Button section in `design.md`, not to hand-roll classes on a `<button>`.
3. **Does it need a color for "selected" / "active" / "hover"?** Before picking a class:
   - Read the actual values in `src/app/globals.css` for the classes you're about to use. Don't assume `accent` ≠ `secondary` ≠ `muted` — in this theme they're identical. Confirm contrast against the _actual container_ background, not the general Tailwind palette in your head.
   - If the element lives inside the sidebar, use the `sidebar-*` token family (`bg-sidebar`, `bg-sidebar-accent`, `text-sidebar-accent-foreground`), not the generic tokens.
4. **Is the shadcn primitive you want to use not in `src/components/ui/`?** Check the "Known gaps" list in `design.md` first — if it's `Card`, `Textarea`, `sonner`, or a `Sidebar` block, use the documented substitute. If it's something else entirely missing, stop and flag it; don't hand-write a parallel implementation inside a chat component.
5. **Still unsure?** Grep this directory and `src/app/(dashboard)/` for an existing instance of the same UI idea (e.g. "how does the rest of the dashboard render a data table") before inventing a new pattern for chat specifically. The chat UI should look like the rest of the app, not like its own product.

## Decision tree: "I'm fixing a bug in an existing component"

1. Reproduce it first — actually load the page, don't infer from reading code.
2. If the bug is visual (wrong color, wrong spacing, misaligned element), check whether the _component_ is even the right primitive before touching classes. A hand-rolled `<button>` with a color bug is often better fixed by replacing it with `Button` than by fixing the color, because the color bug is usually a symptom of not having the component's built-in state handling.
3. Fix the root cause, not the symptom you can see. "The active tab has an invisible highlight" was fixed correctly only on the _second_ attempt — the first fix (a subtle shadow) treated the symptom (not distinct enough) without asking why `bg-accent` had no contrast in the first place.
4. After the fix, look for the same bug pattern elsewhere in the same file and sibling files. If one nav item had a contrast bug, check whether the pattern is reused elsewhere before considering it done.

## When it's OK to deviate from `design.md`

- The pattern truly doesn't exist yet in this codebase and adding the missing shadcn piece (e.g. a real `Sidebar` block, `Textarea`, `sonner`) is out of scope for the current change. Use the documented substitute in "Known gaps," and don't block your task on it — but don't invent a _different_ substitute than the one written down either.
- You're the one updating `design.md` itself because the pattern was wrong (like the `bg-secondary`-sidebar-with-`bg-accent`-active-item combination that shipped with zero contrast). In that case, update `design.md` in the same change, with a note on what was wrong and why the new pattern is correct — don't leave the stale guidance in place for the next agent to copy.

Never deviate silently. If you improvise, the next agent (or you, in six months) will copy the improvisation as if it were the standard.

## Pre-commit checklist

Run through this before considering a chat UI change done:

- [ ] Every clickable element is a `Button` variant, not a raw `<button>` — search your diff for `<button` and justify any that remain (the chat composer's textarea-adjacent icon buttons and similar truly-custom controls are the only accepted exceptions; nav items and dialog actions are not).
- [ ] Every color class you added actually contrasts against its container — checked in `globals.css`, not assumed.
- [ ] Sidebar-scoped elements use `sidebar-*` tokens, not generic `accent`/`secondary`/`muted`.
- [ ] No new `text-foreground/NN` opacity hacks — use `text-muted-foreground`.
- [ ] Loading states are `Skeleton`, not a bare spinner, unless it's a button's own in-flight state.
- [ ] Tabs use `variant="line"` if they're section tabs.
- [ ] Destructive actions go through `AlertDialog`.
- [ ] `npx tsc --noEmit -p .` is clean for touched files.
- [ ] `npx prettier --write <files>` run on anything you hand-edited (don't let formatting drift force a second review pass).
- [ ] Actually opened the page in a browser and looked at the change — a design-system violation that "should" render correctly per the code is still a bug if it doesn't.
- [ ] If you changed or discovered a gap in `design.md`, `design.md` was updated in the same commit, not left for later.

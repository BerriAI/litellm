# Shared knowledge demo

A four-step prototype inside LiteLLM's existing dashboard, available to admins from **Agentic → Memory → Shared knowledge demo**.

1. **Define scope:** enable categories for user systems and operator systems.
2. **Capture a memo:** save an agent's researched finding with a title, summary, guidance, and source evidence.
3. **Review and reuse:** review a proposed correction, publish a new version, load it in another session, and record a helpful outcome. Try an unsupported correction to see rejection preserve the published version.
4. **Learn and curate:** promote frequently useful guidance to a skill, archive a superseded workaround, and retain the rare but useful Bazel fix.

The four numbered PNG files show this sequence in the real dashboard shell. The Memory before/after images highlight the new entry point. The before image uses the static dashboard shipped at the base revision; both use the same empty Memory fixture and URL.

## Run locally

Use the Node and npm versions required by `package.json`. From `ui/litellm-dashboard`:

```sh
npm ci
npm run build -- --webpack
node scripts/shared-knowledge-preview.mjs
```

Open <http://127.0.0.1:3222/preview>. This creates a synthetic localhost demo session and opens `/ui/memory/demo/`. The preview serves the production export and read-only fixtures; it does not require a running proxy or database. The production dashboard continues to use its normal authentication and Memory permissions.

Step links use `?step=2`, `?step=3`, and `?step=4`. Switching steps keeps local demo state; reloading or selecting **Reset Demo** resets it. Jumping to a later step without saving a draft uses a clearly labeled published sample memo.

## Prototype boundaries

All sample data is synthetic. Saving, searching, judging, recording outcomes, and periodic review are simulated in browser state. Postgres persistence, vector/text indexes, service-side model review, real cross-user retrieval, and automatic scheduling are proposed behavior, not implemented services.

Historical counts and review thresholds are illustrative. Unknown outcomes are excluded from usefulness rates. Recorded outcomes retain the memo version used, and a promoted skill retains its source version.

## Message to accompany screenshots

> Hey, I put this into LiteLLM's Memory UI:
>
> 1. Choose the user/operator systems we want agents to remember.
> 2. An agent saves a researched finding as a shared memo.
> 3. A judge reviews corrections; other agents retrieve the published version and report outcomes.
> 4. Periodic review promotes useful patterns to skills, archives weak guidance, and keeps useful rare fixes.
>
> This is an interactive UI demo with simulated data. Is this the flow you had in mind?

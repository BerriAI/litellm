# Architecture decision records

An ADR records one architectural decision: what we chose, why, and what it costs us. It is not a design doc, a tutorial, or a spec. If you can state the decision in a sentence and the reasoning in a page, it belongs here

We keep them because the reasoning behind a design is the part that never survives in code. `ARCHITECTURE.md` tells you where things live, docstrings tell you what a function does, and neither tells you why the usage bridge is generic instead of per-provider. Contributors (and coding agents) that can't find that rationale end up handrolling a second mechanism next to the one that already exists, which is how a one-file fix turns into per-endpoint tech debt

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-provider-usage-extras-and-built-in-tool-cost.md) | Provider usage extras ride on the normalized Usage object | Accepted |

## Writing one

Copy [0000-template.md](0000-template.md) to `NNNN-short-title.md`, taking the next free number, then fill it in and add a row to the index above. Keep it under a page or two: context, the decision, the alternatives you rejected, and the consequences a future contributor will actually hit

Write for someone who has never seen the code, and name the concrete modules, classes, and functions the decision lives in, because those names are what a reader (or an agent's search) uses to find the machinery instead of rebuilding it

## Changing one

ADRs are append-mostly. Correct wording and stale file paths in place, but don't rewrite a decision to match a new one: add a new ADR that supersedes the old one, flip the old status to `Superseded by NNNN`, and link both ways. The history of what we used to believe is the point

## When to write one

Write an ADR when you pick between real alternatives in a way the next person could plausibly get wrong: a new cross-cutting mechanism, a change to how data moves between the endpoints and the SDK, a persistence or concurrency model, an intentional deviation from a provider's own API shape. Skip it for ordinary bug fixes, new providers that follow the existing patterns, and anything the code already says plainly

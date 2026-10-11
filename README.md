# Auto-router UI comparison

Frontend baseline: bea0d5cb331015e629bf01bcbdb5b745c958057b
Frontend candidate: 36b18fa2e07f57a1d72a0bda39cf31b45e96875a

Both captures use /models-and-endpoints with the same local gateway and empty model list. The gateway version badge reports the existing local gateway version, not the frontend source revision. Red outlines mark changed controls. No provider credentials or customer data are present.

## Main populated gate

Baseline: bea0d5cb331015e629bf01bcbdb5b745c958057b
Candidate: 36b18fa2e07f57a1d72a0bda39cf31b45e96875a

These captures use /models-and-endpoints on an isolated local gateway with task-owned fake deployments only (jev-decider, jev-decider-2, jev-catalog, tier-small, tier-large, gpt-6-luna, claude-opus-5-5). Baseline and candidate share the same database and data. Every classifier and solver call went to a controlled local HTTP upstream, so no real provider call is shown. Red outlines mark changed controls

main-populated-before.png and main-populated-after.png show the Add Auto Router modal on the same data

main-edit-deployment-after.png shows a saved deployment-backed router reopened in edit with the Decision model picker listing only decision deployments

main-edit-timeout-after.png shows Advanced settings then Classifier tuning expanded in edit with Classifier Timeout (ms) changed to 4500, which the PATCH and stored config kept alongside the selected deployment alias

main-edit-legacy-before.png and main-edit-legacy-after.png show a legacy direct Jev router in edit, which keeps its provider controls on both sides

main-template-reset-after.png shows Start from scratch clearing the tiers and returning to Heuristics

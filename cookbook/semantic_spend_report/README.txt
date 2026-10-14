Semantic spend POC

Open artifacts/report.html in a browser. The report is self-contained and works without a server

This run uses 24 synthetic prompts across three persistent Claude Code sessions. Tools are disabled. The report describes proposed engineering work and draft material, not measured productivity or realized ROI

Requirements

Python 3.12 or later and the locally installed Claude Code CLI. No Python packages, containers, database services, or browser downloads are required

Credentials

Set LITELLM_API_KEY in the environment using your approved secret mechanism. Never write it into these files. The gateway defaults to https://gateway.litellm-sandbox.ai

Run the stages

    python3 generate.py
    python3 pipeline.py --stage extract
    python3 pipeline.py --stage discover
    python3 pipeline.py
    python3 -m unittest -v test_pipeline.py test_presentation.py

Generation performs paid model calls. The full pipeline performs paid taxonomy, classification, and summary calls. Existing successful generation outputs and analysis fingerprints are reused on rerun. Failed or ambiguous network calls are not automatically retried, because a timeout does not establish that a request was unbilled

To create a separate run, set POC_OUTPUT_DIR to a new empty directory before running both commands. Existing runs are preserved. To change the gateway or models, set GATEWAY_URL, REPORT_MODEL, and JEV_MODEL before a new run. Check the gateway model list for supported current models

Pipeline

1. Generate three persistent Claude sessions, each with eight related prompts and deliberate topic changes. Explicit session headers provide the gateway grouping key. Each child receives stdin from DEVNULL and an isolated task-owned Claude configuration directory

2. Retrieve every page of spend logs using UTC date, user and session filters. Validate those same filters locally. Poll until all expected demo traffic is visible. Fetch request/response details separately. Store only allowlisted fields, excluding raw auth headers and gateway metadata

3. Prepare ordered session streams from the latest user instruction and the resulting response for each request. Earlier history remains available for classification, but is not repeatedly supplied to category discovery. This is intentionally limited to the demonstrated text-only Messages traffic

4. Derive shared work-area categories from all three streams in one LLM call. Definitions contain inclusion and exclusion rules, request examples, unknown and mixed categories

5. Classify each request with Jev against all category definitions. Supply the current instruction, response, and up to seven recent messages. Below 0.6 confidence, retain the proposed category for inspection but assign accounting to unknown. This threshold is illustrative, not calibrated on a representative production dataset

6. Sum the gateway's request-level spend values with Decimal, without recomputing model pricing. Persist session, request, category, classification and summary relationships in SQLite. Every request contributes to one accounting bucket. Missing costs are identified explicitly, and verification refuses to complete this demo with missing costs or content

7. Generate concise presentation records for sessions, work areas, and their intersections. Batch four scopes per model call. Each record has a 3-6 word title, a one-sentence overview capped at 25 words, 1-3 findings capped at 15 words each, and 2-3 verbatim quotes capped at 30 words each. Validate citation scope, speaker attribution, and exact source-substring matches. Regenerate a batch once if validation fails, then stop if it still fails. Preserve all attempts in analysis accounting

8. Render an OpenRouter-inspired rankings layout using current LiteLLM dashboard theme tokens. The default leaderboard ranks sessions by recorded spend. A stacked area chart shows spend per time bucket, and a 100% stacked bar shows category shares. Work-area tabs filter the session ranking and time chart. The share chart remains an overview of all categories for the selected user and time window

9. Clicking a session or work area opens a drawer with concise findings and quotes. A quote opens its original request evidence. All metrics respect the UTC interval and user filter. For partial windows, suppress full-scope overviews and retain only findings whose cited requests are all selected. Quotes outside the selected window are hidden. The reporting overhead card always covers the full report-generation run and is labeled separately

10. Query report-generation logs separately and reconcile every observed analysis call ID. Reuse cached extraction, taxonomy and Jev results when only presentation changes. Do not regenerate demo traffic just to redraw the UI

Artifacts

report.html                 Interactive rankings, charts, filters and detail drawers
semantic-spend-rankings.html Identical standalone HTML with a descriptive filename
presentation.json           Concise findings and exact-source quotes by scope
presentation-data.json      All data embedded in the standalone HTML
presentation-verification.json Word-limit, citation and quote validation results
browser-verification.json   Executed Chrome interaction checks
rankings-overview.png       Desktop overview screenshot
rankings-session-detail.png Session drawer screenshot
rankings-category-detail.png Category drawer screenshot
rankings-mobile.png         Mobile layout screenshot
ui-provenance.json          LiteLLM main commit and referenced theme files
report.json                 Machine-readable report
report.sqlite               Relational dataset, schema in schema.sql
category-spend.csv          Category totals and percentages
request-lineage.csv         User, session, request, cost, label, confidence and evidence text
taxonomy.json               Discovered category definitions and examples
requests.json               Scoped, sanitized request records
classifications.json        Jev outcomes, probabilities and analysis call IDs
summaries.json              Grounded findings per session/category
findings.json               Concise category findings derived from presentation records
verification.json           Accounting, scope, completeness and prompt checks
overhead.json               Gateway-recorded analysis requests and costs
traffic/                    Sanitized Claude outputs and session continuity evidence
analysis/                   Cached model results and input fingerprints
quality-review.json         Manual agent review of classifications and summaries
resource-status.json        Resource ownership, retention and cleanup record

Audit and limitations

The demo's Claude sessions are genuine persistent CLI conversations, but all work is synthetic and advisory. No application files were edited and no engineering tests were executed by Claude. There are no measured time savings, accepted changes or business outcomes from which to compute ROI

The extractor is validated for these text-only, tool-disabled sessions. Tool loops, compaction, branches, images, missing session IDs and other API formats need explicit normalization before a production rollout. It does not claim reliable turn reconstruction for arbitrary client traffic

The reporting window selects billable requests by their start timestamp. Context carried by those requests may predate the reporting window. Cost is attributed only to the selected requests

This POC reports one selected user and a fixed session allowlist. A gateway integration should query authorized users dynamically, enforce tenant access, handle retention/redaction, and run these stages as background jobs. Bounded context selection and taxonomy discovery need evaluation on representative traffic before scaling

Category totals are arithmetic facts from logs. Labels and summaries remain model judgments. Unknown and mixed assignments remain visible rather than being redistributed. Confidence is not a measured accuracy guarantee. Evidence-ID validation checks provenance, not semantic entailment; manual review is still needed

The generated SQLite file and exports are local artifacts. They include the authorized user's ID and synthetic request content, so do not publish them as public repository files without review. No gateway secret is stored in the artifacts

No existing LiteLLM repository files or persistent user Claude settings are changed by this POC

UI verification

The browser check requires Playwright and an installed Google Chrome. It is optional for using the generated report. Run python test_report_browser.py in an existing compatible browser environment. The POC development run reused an existing environment without installing packages or browsers

The report works directly from a local file, without a server or external JavaScript dependencies. Theme tokens are copied from the LiteLLM main commit recorded in ui-provenance.json. Inter is declared with local system font fallbacks. The standalone report does not bundle the entire dashboard React application

The pipeline tests cover quote paraphrases, wrong-speaker attribution, out-of-scope citations, word limits, and grouping, in addition to accounting tests. Browser checks exercise filters, sorting, drawers, quote evidence, empty/invalid windows, keyboard navigation, chart totals and mobile layout

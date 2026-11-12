# Task Insights demo

Task Insights groups coding-agent traffic into named tasks and shows, for each task, how many requests it gets, what it costs and how each model deployment performs on it. It lives in the dashboard under **Usage > Task Insights**. This folder runs it from this checkout with sample data, so you can explore it without any provider keys and without a single model call.

## Run the seeded demo

You need Docker with Compose v2, about 15 GB of free disk for the build, and port 4000 free.

```bash
git fetch origin litellm_task_insights_poc
git checkout litellm_task_insights_poc
printf 'LITELLM_MASTER_KEY=sk-%s\nLITELLM_SALT_KEY=sk-%s\n' "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" > docker/task_insights/.env
docker compose -f docker/task_insights/docker-compose.yml up -d --build --wait
docker compose -f docker/task_insights/docker-compose.yml run --rm task-insights-seed
```

The build compiles the proxy and dashboard from this checkout, which takes about 25 minutes the first time. The seed prints `Task Insights sample data seeded: N sample assignments, no model calls`. Running it again replaces the sample data with fresh dates and leaves every other record alone. It refuses to run when a taxonomy that is not sample data already exists.

Open http://localhost:4000/ui, sign in as `admin` with the `LITELLM_MASTER_KEY` from `docker/task_insights/.env`, then go to **Usage** and pick the **Task Insights** tab.

## Walkthrough

1. The tab opens on the last 7 days with a **Sample data** badge. The cards show requests, ledger deliveries (retries make deliveries outnumber requests), sessions, task coverage and known spend. The figures never count a missing cost as zero, they say how many requests the spend rests on.
2. The **task map** places each task as a bubble sized by volume. Filled dots are discovery sample positions and hollow dots are later requests drawn around their task. Positions only group similar requests, distance says nothing about model quality.
3. **Tasks by volume and spend** is the ranked table equivalent of the map. Switch the ranking to spend to see which task costs most.
4. **Models by task** is the heatmap. Switch between requests, mean cost and p95 latency. Grey striped cells have no evidence for that metric. `claude-fable-5-1` is configured but has no traffic, so its column stays empty. A warning icon marks a deployment whose model metadata says it lacks a capability the task needs.
5. Click a bubble, a row or a cell. The side sheet shows the task definition, what it includes and excludes, examples, per-model metrics and the recent requests behind it, including requests with unknown cost or latency. A cell click filters the requests to that deployment.
6. Change the date range or pick a team (`Sample: Platform`, `Sample: Mobile`, `Sample: Data`). The URL keeps the tab, team, metric and selection, so the link is shareable.
7. Sign in as a non-admin and the tab is not offered, the API answers 403.

## Optional live flow

This sends a small batch of synthetic coding requests through the gateway, discovers tasks from it, and then buckets a second batch with Jev. It calls paid models. In our runs the whole flow costs well under a dollar, and Devin's verification notes report the measured spend.

Add the provider keys to `docker/task_insights/.env`. Only the names are listed here:

- `ANTHROPIC_API_KEY` for the coding traffic (`claude-haiku-5-5`), intent distillation and task synthesis (`claude-sonnet-5-5`)
- `DASHSCOPE_API_KEY` for embeddings with `qwen3.7-text-embedding` through DashScope's international endpoint
- `TYPESAFE_API_KEY` for the Jev decision model behind `task-insights-decision`

Recreate the proxy so it reads them, start the worker and send the discovery batch:

```bash
docker compose -f docker/task_insights/docker-compose.yml up -d --force-recreate --wait litellm
docker compose -f docker/task_insights/docker-compose.yml --profile live up -d --build task-insights-worker
docker compose -f docker/task_insights/docker-compose.yml run --rm task-insights-traffic send --batch discovery
```

For live discovery, start from a database without the sample taxonomy. Either reset the sample first (`run --rm task-insights-seed --reset`) or use a fresh volume. Wait a minute for the ledger to apply the batch, then click **Discover tasks** in the tab and confirm. The worker distills, embeds and clusters the requests and drafts definitions, usually within a couple of minutes. Review and rename the drafts and click **Freeze taxonomy**. The worker then assigns every request with Jev. Send the held-out batch and watch the counts update without rediscovering the taxonomy:

```bash
docker compose -f docker/task_insights/docker-compose.yml run --rm task-insights-traffic send --batch heldout
```

To measure agreement, export the held-out prompts, have a person label each with a frozen task name or `unknown` without looking at the assignments, then score:

```bash
docker compose -f docker/task_insights/docker-compose.yml run --rm task-insights-traffic export /labels/heldout.csv
docker compose -f docker/task_insights/docker-compose.yml run --rm task-insights-traffic score /labels/heldout.csv
```

The prompts were written by us for each category, and that author intent is not a human label.

## Clean up

```bash
docker compose -f docker/task_insights/docker-compose.yml run --rm task-insights-seed --reset
docker compose -f docker/task_insights/docker-compose.yml --profile live down -v
docker image rm litellm-task-insights:local litellm-task-insights-worker:local
```

The first command removes only the sample records. `down -v` deletes the demo database volume.

## How it works

- **Observation.** With `LITELLM_TASK_INSIGHTS_ENABLED=true`, the existing accounting flush copies each applied `LiteLLM_SpendLogDelivery` into `LiteLLM_TaskObservation`, keyed by delivery id, after billing commits. It joins the request log for the deployment id, session and a bounded excerpt of the latest and first user text. The ledger keeps an applied delivery until it is observed, so retention cannot drop analytics. With the flag unset nothing changes. See `docs/adrs/adr_002_task_insights_observes_applied_deliveries.md`.
- **Counting.** Figures count distinct requests. A request's cost is known only when every delivery recorded a charge, failed and retried attempts included. Latency comes from successful attempts. A delivery is tied to a deployment only when its billed model matches the logged request's model.
- **Discovery** follows LLooM's concept induction ([Lam et al., CHI 2024](https://arxiv.org/abs/2404.12259), [workbench](https://github.com/michelle123lam/lloom)): distill each request into a one-line intent with an LLM, embed the intents with Qwen ([Qwen3 Embedding](https://github.com/QwenLM/Qwen3-Embedding)), reduce with UMAP for HDBSCAN clustering and run a separate 2D UMAP for the map ([UMAP clustering notes](https://umap-learn.readthedocs.io/en/latest/clustering.html), [scikit-learn HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)), then synthesize a definition per cluster. Up to 500 distinct request texts, at most 12 tasks, noise and dropped clusters are reported. The UI's concept list, model by task matrix and detail drawer borrow LLooM's concept overview, concept matrix and drill-down ([visualization guide](https://stanfordhci.github.io/lloom/about/vis-guide.html)).
- **Bucketing** asks Jev through the gateway's `/v1/systemone` route for a Choice over the frozen tasks plus a companion Noul fit question ([Jev Choice](https://docs.typesafe.ai/primitives/choice)). A request is unknown when the chosen task's probability or the fit is below 0.5. Requests with the same text share one decision, so a replayed agent history costs one call. Jev's probabilities are its own estimate, not a measured accuracy.
- **Analysis traffic** carries the `litellm-task-insights-analysis` tag, so it is billed like other traffic, shown as analysis spend, and never analyzed itself.
- **Placement.** Usage > Task Insights is a tab because it is another view of the same usage data under the same date filter ([NN/g on tabs](https://www.nngroup.com/articles/tabs-used-right/)).

## Limitations

- The seeded taxonomy, positions and assignments are precomputed sample data. They show the UI and the persistence, not discovery or Jev quality.
- Live quality depends on the labeled check above. A small batch gives a wide margin of error.
- Request text exists only when `store_prompts_in_spend_logs` is on. Requests without it count as not classified.
- Observations keep bounded request excerpts with no retention of their own yet.
- Requests bucketed after discovery have no embedding, so they are drawn at a stable offset around their task.
- Compatibility comes from model metadata. A router must still check each request's actual context size and features.
- The eval fields in the matrix stay empty. There is no eval runner, routing or correction detection.
- Reads need a proxy admin or admin viewer role, writes need a proxy admin. Team admins do not get a team-scoped view yet.

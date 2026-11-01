# .circleci

The `integration` workflow on `main` runs from a scheduled pipeline in the CircleCI project settings, cron `9 0-8,14-23 * * *` (UTC) on `main` with no pipeline parameters. It has no `triggers:` block in this file because a workflow with one runs only on its schedule, so the API pipelines that set `run_migration_tests` or `routing_parity_base` would stop running it. If main stops getting integration runs, check that the schedule still exists and recreate it with these values

Jobs clone with the `checkout_over_https` command, never the built-in `checkout` step, because CircleCI falls back to the project's SSH checkout key without its GitHub App access, GitHub rejects that key, and the public repo needs no key. It fetches the job's commit with full history and no file contents (`--filter=blob:none`), like the built-in step did, so `scripts/path_filter.sh` can find a merge base

A step that checks out files from another commit fetches that commit with `--no-filter`, as `routing_parity` does, because the clone has no file contents and git would otherwise download each missing file in its own request, which took 210 to 896 seconds for the routing parity base

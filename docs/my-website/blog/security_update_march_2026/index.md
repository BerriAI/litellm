---
slug: security-update-march-2026
title: "Security Update: Suspected Supply Chain Incident"
date: 2026-03-24T14:00:00
authors:
  - krrish
  - ishaan-alt
description: "As of 2:00 PM ET on March 24, 2026"
tags: [security, incident-report]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

> **Status:** Active investigation
> **Last updated:** March 26, 2026

> **Update (March 26):** Added `checkmarx[.]zone` to [Indicators of compromise](#indicators-of-compromise-iocs).

> **Update (March 25):** Added community-contributed scripts for scanning GitHub Actions and GitLab CI pipelines for the compromised versions. See [How to check if you are affected](#how-to-check-if-you-are-affected). s/o [@Zach Fury](https://www.linkedin.com/in/fryware/) for these scripts.


## TLDR; 
- The compromised PyPI packages were **litellm==1.82.7** and **litellm==1.82.8**. Those packages have now been removed from PyPI.
- We believe that the compromise originated from the Trivy dependency used in our CI/CD security scanning workflow.
- Customers running the official LiteLLM Proxy Docker image were not impacted. That deployment path pins dependencies in requirements.txt and does not rely on the compromised PyPI packages.
- We are pausing new LiteLLM releases until we complete a broader supply-chain review and confirm the release path is safe.


## Overview

LiteLLM AI Gateway is investigating a suspected supply chain attack involving unauthorized PyPI package publishes. Current evidence suggests a maintainer's PyPI account may have been compromised and used to distribute malicious code.

At this time, we believe this incident may be linked to the broader [Trivy security compromise](https://www.aquasec.com/blog/trivy-supply-chain-attack-what-you-need-to-know/), in which stolen credentials were reportedly used to gain unauthorized access to the LiteLLM publishing pipeline.

This investigation is ongoing. Details below may change as we confirm additional findings.

## Confirmed affected versions

The following LiteLLM versions published to PyPI were impacted:

- **v1.82.7**: contained a malicious payload in the LiteLLM AI Gateway `proxy_server.py`
- **v1.82.8**: contained `litellm_init.pth` and a malicious payload in the LiteLLM AI Gateway `proxy_server.py`

If you installed or ran either of these versions, review the recommendations below immediately.

Note: These versions have already been removed from PyPI.

## What happened

Initial evidence suggests the attacker bypassed official CI/CD workflows and uploaded malicious packages directly to PyPI.

These compromised versions appear to have included a credential stealer designed to:

- Harvest secrets by scanning for:
  - environment variables
  - SSH keys
  - cloud provider credentials (AWS, GCP, Azure)
  - Kubernetes tokens
  - database passwords
- Encrypt and exfiltrate data via a `POST` request to `models.litellm.cloud`, which is **not** an official BerriAI / LiteLLM domain

## Who is affected

You may be affected if **any** of the following are true:

- You installed or upgraded LiteLLM via `pip` on **March 24, 2026**, between **10:39 UTC and 16:00 UTC**
- You ran `pip install litellm` without pinning a version and received **v1.82.7** or **v1.82.8**
- You built a Docker image during this window that included `pip install litellm` without a pinned version
- A dependency in your project pulled in LiteLLM as a transitive, unpinned dependency
  (for example through AI agent frameworks, MCP servers, or LLM orchestration tools)

You are **not** affected if any of the following are true:

**LiteLLM AI Gateway/Proxy users:** Customers running the official LiteLLM Proxy Docker image were not impacted. That deployment path pins dependencies in requirements.txt and does not rely on the compromised PyPI packages.

- You are using **LiteLLM Cloud**
- You are using the official LiteLLM AI Gateway Docker image: `ghcr.io/berriai/litellm`
- You are on **v1.82.6 or earlier** and did not upgrade during the affected window
- You installed LiteLLM from source via the GitHub repository, which was **not** compromised


### How to check if you are affected

<Tabs>
<TabItem value="sdk" label="SDK">

```bash
pip show litellm
```
</TabItem>
<TabItem value="proxy" label="PROXY">

Go to the proxy base url, and check the version of the installed LiteLLM.

![Proxy version check](../../img/security_update_march_2026/proxy_version.png)
</TabItem>
<TabItem value="github" label="GitHub Actions">

Scans all repositories in a GitHub organization for workflow jobs that installed the compromised versions.

**Requirements:** Python 3 and `requests` (`pip install requests`).

**Setup:**

```bash
export GITHUB_TOKEN="your-github-pat"
```

**Run:**

```bash
python find_litellm_github.py
```

Set the `ORG` variable in the script to your GitHub organization name.

Both scripts default to scanning jobs from **today**. Adjust the `WINDOW_START` and `WINDOW_END` constants to cover **March 24, 2026** (the incident date) if running on a different day.

<details>
<summary>View full script (find_litellm_github.py)</summary>

```python
#!/usr/bin/env python3
"""
Scan all GitHub Actions jobs in a GitHub org that ran between
0800-1244 UTC today and identify any that installed litellm 1.82.7 or 1.82.8.

Adjust WINDOW_START / WINDOW_END to cover March 24, 2026 if running later.
"""

import io
import os
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

GITHUB_URL   = "https://api.github.com"
ORG          = "your-org"  # <-- set to your GitHub organization
TOKEN        = os.environ.get("GITHUB_TOKEN", "")

TODAY        = datetime.now(timezone.utc).date()
WINDOW_START = datetime(TODAY.year, TODAY.month, TODAY.day,  8,  0, 0, tzinfo=timezone.utc)
WINDOW_END   = datetime(TODAY.year, TODAY.month, TODAY.day, 12, 44, 0, tzinfo=timezone.utc)

TARGET_VERSIONS = {"1.82.7", "1.82.8"}
VERSION_PATTERN = re.compile(r"litellm[=\-](\d+\.\d+\.\d+)", re.IGNORECASE)

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})


def get_paginated(url, params=None):
    params = dict(params or {})
    params.setdefault("per_page", 100)
    page = 1
    while True:
        params["page"] = page
        resp = SESSION.get(url, params=params, timeout=30)
        if resp.status_code == 404:
            return
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            items = next((v for v in data.values() if isinstance(v, list)), [])
        else:
            items = data
        if not items:
            break
        yield from items
        if len(items) < params["per_page"]:
            break
        page += 1


def parse_ts(ts_str):
    if not ts_str:
        return None
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def get_repos():
    repos = []
    for r in get_paginated(f"{GITHUB_URL}/orgs/{ORG}/repos", {"type": "all"}):
        repos.append({"id": r["id"], "name": r["name"], "full_name": r["full_name"]})
    return repos


def get_runs_in_window(repo_full_name):
    created_filter = (
        f"{WINDOW_START.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"..{WINDOW_END.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    url = f"{GITHUB_URL}/repos/{repo_full_name}/actions/runs"
    runs = []
    for run in get_paginated(url, {"created": created_filter, "per_page": 100}):
        ts = parse_ts(run.get("run_started_at") or run.get("created_at"))
        if ts and WINDOW_START <= ts <= WINDOW_END:
            runs.append(run)
    return runs


def get_jobs_for_run(repo_full_name, run_id):
    url = f"{GITHUB_URL}/repos/{repo_full_name}/actions/runs/{run_id}/jobs"
    jobs = []
    for job in get_paginated(url, {"filter": "all"}):
        ts = parse_ts(job.get("started_at"))
        if ts and WINDOW_START <= ts <= WINDOW_END:
            jobs.append(job)
    return jobs


def fetch_job_log(repo_full_name, job_id):
    url = f"{GITHUB_URL}/repos/{repo_full_name}/actions/jobs/{job_id}/logs"
    resp = SESSION.get(url, timeout=60, allow_redirects=True)
    if resp.status_code in (403, 404, 410):
        return ""
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "zip" in content_type or resp.content[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                parts = []
                for name in sorted(zf.namelist()):
                    with zf.open(name) as f:
                        parts.append(f.read().decode("utf-8", errors="replace"))
                return "\n".join(parts)
        except zipfile.BadZipFile:
            pass
    return resp.text


def check_job(repo_full_name, job):
    job_id   = job["id"]
    job_name = job["name"]
    run_id   = job["run_id"]
    started  = job.get("started_at", "")

    log_text = fetch_job_log(repo_full_name, job_id)
    if not log_text:
        return None

    found_versions = set()
    context_lines  = []
    for line in log_text.splitlines():
        m = VERSION_PATTERN.search(line)
        if m:
            ver = m.group(1)
            if ver in TARGET_VERSIONS:
                found_versions.add(ver)
                context_lines.append(line.strip())

    if not found_versions:
        return None

    return {
        "repo":       repo_full_name,
        "run_id":     run_id,
        "job_id":     job_id,
        "job_name":   job_name,
        "started_at": started,
        "versions":   sorted(found_versions),
        "context":    context_lines[:10],
        "job_url":    job.get("html_url", f"https://github.com/{repo_full_name}/actions/runs/{run_id}"),
    }


def main():
    if not TOKEN:
        print("ERROR: Set GITHUB_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    print(f"Time window : {WINDOW_START.isoformat()} -> {WINDOW_END.isoformat()}")
    print(f"Hunting for : litellm {', '.join(sorted(TARGET_VERSIONS))}")
    print()

    print(f"Fetching repositories for org '{ORG}'...")
    repos = get_repos()
    print(f"  Found {len(repos)} repositories")
    print()

    jobs_to_check = []

    print("Scanning workflow runs for time window...")
    for repo in repos:
        full_name = repo["full_name"]
        try:
            runs = get_runs_in_window(full_name)
        except requests.HTTPError as e:
            print(f"  WARN: {full_name} - {e}", file=sys.stderr)
            continue
        if not runs:
            continue
        print(f"  {full_name}: {len(runs)} run(s) in window")
        for run in runs:
            try:
                jobs = get_jobs_for_run(full_name, run["id"])
            except requests.HTTPError as e:
                print(f"    WARN: run {run['id']} - {e}", file=sys.stderr)
                continue
            for job in jobs:
                jobs_to_check.append((full_name, job))

    total = len(jobs_to_check)
    print(f"\nFetching logs for {total} job(s)...")
    print()

    hits = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(check_job, full_name, job): (full_name, job["id"])
            for full_name, job in jobs_to_check
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            full_name, jid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  ERROR {full_name} job {jid}: {e}", file=sys.stderr)
                continue
            if result:
                hits.append(result)
            print(
                f"  [{done}/{total}] {full_name} job {jid}" +
                (f"  *** HIT: litellm {result['versions']} ***" if result else ""),
                flush=True,
            )

    print()
    print("=" * 72)
    print(f"RESULTS: {len(hits)} job(s) installed litellm {' or '.join(sorted(TARGET_VERSIONS))}")
    print("=" * 72)

    if not hits:
        print("No matches found.")
        return

    for h in sorted(hits, key=lambda x: x["started_at"]):
        print()
        print(f"  Repo      : {h['repo']}")
        print(f"  Job       : {h['job_name']} (#{h['job_id']})")
        print(f"  Run ID    : {h['run_id']}")
        print(f"  Started   : {h['started_at']}")
        print(f"  Versions  : litellm {', '.join(h['versions'])}")
        print(f"  URL       : {h['job_url']}")
        print(f"  Log lines :")
        for line in h["context"]:
            print(f"    {line}")


if __name__ == "__main__":
    main()
```

</details>

</TabItem>
<TabItem value="gitlab" label="GitLab CI">

Scans all projects in a GitLab group (including subgroups) for CI/CD jobs that installed the compromised versions.

**Requirements:** Python 3 and `requests` (`pip install requests`).

**Setup:**

```bash
export GITLAB_TOKEN="your-gitlab-pat"
```

**Run:**

```bash
python find_litellm_jobs.py
```

Set the `GROUP_NAME` variable in the script to your GitLab group name.

Both scripts default to scanning jobs from **today**. Adjust the `WINDOW_START` and `WINDOW_END` constants to cover **March 24, 2026** (the incident date) if running on a different day.

<details>
<summary>View full script (find_litellm_jobs.py)</summary>

```python
#!/usr/bin/env python3
"""
Scan all GitLab CI/CD jobs in a GitLab group that ran between
0800-1244 UTC today and identify any that installed litellm 1.82.7 or 1.82.8.

Adjust WINDOW_START / WINDOW_END to cover March 24, 2026 if running later.
"""

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

GITLAB_URL = "https://gitlab.com"
GROUP_NAME = "YourGroup"  # <-- set to your GitLab group name
TOKEN = os.environ.get("GITLAB_TOKEN", "")

TODAY = datetime.now(timezone.utc).date()
WINDOW_START = datetime(TODAY.year, TODAY.month, TODAY.day, 8, 0, 0, tzinfo=timezone.utc)
WINDOW_END   = datetime(TODAY.year, TODAY.month, TODAY.day, 12, 44, 0, tzinfo=timezone.utc)

TARGET_VERSIONS = {"1.82.7", "1.82.8"}
VERSION_PATTERN = re.compile(r"litellm[=\-](\d+\.\d+\.\d+)", re.IGNORECASE)

HEADERS = {"PRIVATE-TOKEN": TOKEN}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_paginated(url, params=None):
    params = dict(params or {})
    params.setdefault("per_page", 100)
    page = 1
    while True:
        params["page"] = page
        resp = SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        yield from data
        if len(data) < params["per_page"]:
            break
        page += 1


def get_group_id(group_name):
    resp = SESSION.get(f"{GITLAB_URL}/api/v4/groups/{group_name}", timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]


def get_all_projects(group_id):
    projects = []
    for p in get_paginated(
        f"{GITLAB_URL}/api/v4/groups/{group_id}/projects",
        {"include_subgroups": "true", "archived": "false"},
    ):
        projects.append({"id": p["id"], "name": p["path_with_namespace"]})
    return projects


def parse_ts(ts_str):
    if not ts_str:
        return None
    ts_str = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(ts_str)


def jobs_in_window(project_id):
    matching = []
    url = f"{GITLAB_URL}/api/v4/projects/{project_id}/jobs"
    params = {"per_page": 100, "scope[]": ["success", "failed", "canceled", "running"]}

    page = 1
    while True:
        params["page"] = page
        resp = SESSION.get(url, params=params, timeout=30)
        if resp.status_code == 403:
            return matching
        resp.raise_for_status()
        jobs = resp.json()
        if not jobs:
            break

        stop_early = False
        for job in jobs:
            ts = parse_ts(job.get("started_at") or job.get("created_at"))
            if ts is None:
                continue
            if ts > WINDOW_END:
                continue
            if ts < WINDOW_START:
                stop_early = True
                continue
            matching.append(job)

        if stop_early or len(jobs) < 100:
            break
        page += 1

    return matching


def fetch_trace(project_id, job_id):
    url = f"{GITLAB_URL}/api/v4/projects/{project_id}/jobs/{job_id}/trace"
    resp = SESSION.get(url, timeout=60)
    if resp.status_code in (403, 404):
        return ""
    resp.raise_for_status()
    return resp.text


def check_job(project_name, project_id, job):
    job_id   = job["id"]
    job_name = job["name"]
    ref      = job.get("ref", "")
    started  = job.get("started_at", job.get("created_at", ""))

    trace = fetch_trace(project_id, job_id)
    if not trace:
        return None

    found_versions = set()
    for match in VERSION_PATTERN.finditer(trace):
        ver = match.group(1)
        if ver in TARGET_VERSIONS:
            found_versions.add(ver)

    if not found_versions:
        return None

    context_lines = []
    for line in trace.splitlines():
        if VERSION_PATTERN.search(line):
            ver_match = VERSION_PATTERN.search(line)
            if ver_match and ver_match.group(1) in TARGET_VERSIONS:
                context_lines.append(line.strip())

    return {
        "project":    project_name,
        "project_id": project_id,
        "job_id":     job_id,
        "job_name":   job_name,
        "ref":        ref,
        "started_at": started,
        "versions":   sorted(found_versions),
        "context":    context_lines[:10],
        "job_url":    f"{GITLAB_URL}/{project_name}/-/jobs/{job_id}",
    }


def main():
    if not TOKEN:
        print("ERROR: Set GITLAB_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    print(f"Time window : {WINDOW_START.isoformat()} -> {WINDOW_END.isoformat()}")
    print(f"Hunting for : litellm {', '.join(sorted(TARGET_VERSIONS))}")
    print()

    print(f"Resolving group '{GROUP_NAME}'...")
    group_id = get_group_id(GROUP_NAME)

    print("Fetching projects...")
    projects = get_all_projects(group_id)
    print(f"  Found {len(projects)} projects")
    print()

    all_jobs_to_check = []

    print("Scanning job listings for time window...")
    for proj in projects:
        try:
            jobs = jobs_in_window(proj["id"])
        except requests.HTTPError as e:
            print(f"  WARN: {proj['name']} - {e}", file=sys.stderr)
            continue
        if jobs:
            print(f"  {proj['name']}: {len(jobs)} job(s) in window")
        for j in jobs:
            all_jobs_to_check.append((proj["name"], proj["id"], j))

    total = len(all_jobs_to_check)
    print(f"\nFetching traces for {total} job(s)...")
    print()

    hits = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(check_job, pname, pid, job): (pname, job["id"])
            for pname, pid, job in all_jobs_to_check
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            pname, jid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  ERROR checking {pname} job {jid}: {e}", file=sys.stderr)
                continue
            if result:
                hits.append(result)
            print(f"  [{done}/{total}] checked {pname} job {jid}" +
                  (f"  *** HIT: litellm {result['versions']} ***" if result else ""),
                  flush=True)

    print()
    print("=" * 72)
    print(f"RESULTS: {len(hits)} job(s) installed litellm {' or '.join(sorted(TARGET_VERSIONS))}")
    print("=" * 72)

    if not hits:
        print("No matches found.")
        return

    for h in sorted(hits, key=lambda x: x["started_at"]):
        print()
        print(f"  Project   : {h['project']}")
        print(f"  Job       : {h['job_name']} (#{h['job_id']})")
        print(f"  Branch/tag: {h['ref']}")
        print(f"  Started   : {h['started_at']}")
        print(f"  Versions  : litellm {', '.join(h['versions'])}")
        print(f"  URL       : {h['job_url']}")
        print(f"  Log lines :")
        for line in h["context"]:
            print(f"    {line}")


if __name__ == "__main__":
    main()
```

</details>

</TabItem>
</Tabs>

*CI/CD scripts contributed by the community ([original gist](https://gist.github.com/fryz/93ec8d4898ffe5b5ac5706a208823ef3)). Review before running.*


## Indicators of compromise (IoCs)

Review affected systems for the following indicators:

- `litellm_init.pth` present in your `site-packages`
- Outbound traffic or requests to `models.litellm[.]cloud`
  This domain is **not** affiliated with LiteLLM
- Outbound traffic or requests to `checkmarx[.]zone`
  This domain is **not** affiliated with LiteLLM


## Immediate actions for affected users

If you installed or ran **v1.82.7** or **v1.82.8**, take the following actions immediately.

### 1. Rotate all secrets

Treat any credentials present on the affected systems as compromised, including:

- API keys
- Cloud access keys
- Database passwords
- SSH keys
- Kubernetes tokens
- Any secrets stored in environment variables or configuration files

### 2. Inspect your filesystem

Check your `site-packages` directory for a file named `litellm_init.pth`:

```bash
find /usr/lib/python3.13/site-packages/ -name "litellm_init.pth"
```

If present:

- remove it immediately
- investigate the host for further compromise
- preserve relevant artifacts if your security team is performing forensics

### 3. Audit version history

Review your:

- Local environments
- CI/CD pipelines
- Docker builds
- Deployment logs

Confirm whether **v1.82.7** or **v1.82.8** was installed anywhere.

Pin LiteLLM to a known safe version such as **v1.82.6 or earlier**, or to a later verified release once announced.


## Response and remediation

The LiteLLM AI Gateway team has already taken the following steps:

- Removed compromised packages from PyPI
- Rotated maintainer credentials and established new authorized maintainers
- Engaged Google's Mandiant security team to assist with forensic analysis of the build and publishing chain


## Questions and support

If you believe your systems may be affected, contact us immediately:

- **Security:** `security@berri.ai`
- **Support:** `support@berri.ai`
- **Slack:** Reach out to the LiteLLM team directly

For real-time updates, follow [LiteLLM (YC W23) on X](https://x.com/LiteLLM).


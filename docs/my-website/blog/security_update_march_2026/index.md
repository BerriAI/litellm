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
import VersionVerificationTable from '@site/src/components/VersionVerificationTable';

> **Status:** Active investigation
> **Last updated:** March 27, 2026

> **Update (March 30):** A new **clean** version of LiteLLM is now available (v1.83.0). This was released by our new [CI/CD v2](https://docs.litellm.ai/blog/ci-cd-v2-improvements) pipeline which added isolated environments, stronger security gates, and safer release separation for LiteLLM.

> **Update (March 27):** Review Townhall updates, including explanation of the incident, what we've done, and what comes next. [Learn more](https://docs.litellm.ai/blog/security-townhall-updates)

> **Update (March 27):** Added [Verified safe versions](#verified-safe-versions) section with SHA-256 checksums for all audited PyPI and Docker releases.

> **Update (March 26):** Added `checkmarx[.]zone` to [Indicators of compromise](#indicators-of-compromise-iocs)

> **Update (March 25):** Added community-contributed scripts for scanning GitHub Actions and GitLab CI pipelines for the compromised versions. See [How to check if you are affected](#how-to-check-if-you-are-affected). s/o [@Zach Fury](https://www.linkedin.com/in/fryware/) for these scripts.


## TLDR; 
- The compromised PyPI packages were **litellm==1.82.7** and **litellm==1.82.8**. Those packages were live on March 24, 2026 from 10:39 UTC for about 40 minutes before being quarantined by PyPI.
- We believe that the compromise originated from the [Trivy dependency](https://www.aquasec.com/blog/trivy-supply-chain-attack-what-you-need-to-know/) used in our CI/CD security scanning workflow.
- Customers running the official LiteLLM Proxy Docker image were not impacted. That deployment path pins dependencies in requirements.txt and does not rely on the compromised PyPI packages.
- ~~We have paused all new LiteLLM releases until we complete a broader supply-chain review and confirm the release path is safe.~~ **Updated:** We have now released a new **safe** version of LiteLLM (v1.83.0) by our new [CI/CD v2](https://docs.litellm.ai/blog/ci-cd-v2-improvements) pipeline which added isolated environments, stronger security gates, and safer release separation for LiteLLM. We have also verified the codebase is safe and no malicious code was pushed to `main`.


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


## Verify Docker image signatures

Starting from `v1.83.0-nightly`, all LiteLLM Docker images published to GHCR are signed with [cosign](https://docs.sigstore.dev/cosign/overview/). Every release is signed with the same key introduced in [commit `0112e53`](https://github.com/BerriAI/litellm/commit/0112e53046018d726492c814b3644b7d376029d0).

**Verify using the pinned commit hash (recommended):**

A commit hash is cryptographically immutable, so this is the strongest way to ensure you are using the original signing key:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub \
  ghcr.io/berriai/litellm:<release-tag>
```

**Verify using a release tag (convenience):**

Tags are protected in this repository and resolve to the same key. This option is easier to read but relies on tag protection rules:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/<release-tag>/cosign.pub \
  ghcr.io/berriai/litellm:<release-tag>
```

Replace `<release-tag>` with the version you are deploying (e.g. `v1.83.0-stable`).

Expected output:

```
The following checks were performed on each of these signatures:
  - The cosign claims were validated
  - The signatures were verified against the specified public key
```

## Verified safe versions

We have audited every LiteLLM release published between v1.78.0 and v1.82.6 across both PyPI and Docker. Each artifact was verified by:

1. Downloading the published artifact and computing its SHA-256 digest
2. Scanning for the known [indicators of compromise](#indicators-of-compromise-iocs) (IOCs)
3. Comparing the artifact contents against the corresponding Git commit in the BerriAI/litellm repository

**All versions listed below are confirmed clean.**

<Tabs>
<TabItem value="pypi" label="PyPI Releases">

<VersionVerificationTable entries={[
  { version: "1.82.6", sha256: "164a3ef3e19f309e3cabc199bef3d2045212712fefdfa25fc7f75884a5b5b205", gitCommit: "38d477507dad" },
  { version: "1.82.5", sha256: "e1012ab816352215c4e00776dd48b0c68058b537888a8ff82cca62af19e6fb11", gitCommit: "1998c4f3703f" },
  { version: "1.82.4", sha256: "d37c34a847e7952a146ed0e2888a24d3edec7787955c6826337395e755ad5c4b", gitCommit: "cfeafbe38811" },
  { version: "1.82.3", sha256: "609901f6c5a5cf8c24386e4e3f50738bb8a9db719709fd76b208c8ee6d00f7a7", gitCommit: "61409275c8d8" },
  { version: "1.82.2", sha256: "641ed024774fa3d5b4dd9347f0efb1e31fa422fba2a6500aabedee085d1194cb", gitCommit: "f351bbdb3683" },
  { version: "1.82.1", sha256: "a9ec3fe42eccb1611883caaf8b1bf33c9f4e12163f94c7d1004095b14c379eb2", gitCommit: "94b002066e3a" },
  { version: "1.82.0", sha256: "5496b5d4532cccdc7a095c21cbac4042f7662021c57bc1d17be4e39838929e80", gitCommit: "6c6585af568e" },
  { version: "1.81.16", sha256: "d6bcc13acbd26719e07bfa6b9923740e88409cbf1f9d626d85fc9ae0e0eec88c", gitCommit: "678200ee4887" },
  { version: "1.81.15", sha256: "2fa253658702509ce09fe0e172e5a47baaadf697fb0f784c7fd4ff665ae76ae1", gitCommit: "2e819656cee9" },
  { version: "1.81.14", sha256: "6394e61bbdef7121e5e3800349f6b01e9369e7cf611e034f1832750c481abfed", gitCommit: "96bcee0b0af7" },
  { version: "1.81.13", sha256: "ae4aea2a55e85993f5f6dd36d036519422d24812a1a3e8540d9e987f2d7a4304", gitCommit: "cc957a19a560" },
  { version: "1.81.12", sha256: "219cf9729e5ea30c6d3f75aa43fef3c56a717369939a6d717cbad0fd78e3c146", gitCommit: "ba0d541b1982" },
  { version: "1.81.11", sha256: "06a66c24742e082ddd2813c87f40f5c12fe7baa73ce1f9457eaf453dc44a0f65", gitCommit: "231aedeeff7e" },
  { version: "1.81.10", sha256: "9efa1cbe61ac051f6500c267b173d988ff2d511c2eecf1c8f2ee546c0870747c", gitCommit: "7488abece8e7" },
  { version: "1.81.9", sha256: "24ee273bc8a62299fbb754035f83fb7d8d44329c383701a2bd034f4fd1c19084", gitCommit: "a09d3e9162eb" },
  { version: "1.81.8", sha256: "78cca92f36bc6c267c191d1fe1e2630c812bff6daec32c58cade75748c2692f6", gitCommit: "4fea649f519b" },
  { version: "1.81.7", sha256: "58466c88c3289c6a3830d88768cf8f307581d9e6c87861de874d1128bb2de90d", gitCommit: "3f6a281d0f7a" },
  { version: "1.81.6", sha256: "573206ba194d49a1691370ba33f781671609ac77c35347f8a0411d852cf6341a", gitCommit: "8da3a93e6e63" },
  { version: "1.81.5", sha256: "206505c5a0c6503e465154b9c979772be3ede3f5bf746d15b37dca5ae54d239f", gitCommit: "2cc3778761d4" },
  { version: "1.81.3", sha256: "3f60fd8b727587952ad3dd18b68f5fed538d6f43d15bb0356f4c3a11bccb2b92", gitCommit: "f30742fe6e8e" },
]} />

</TabItem>
<TabItem value="docker" label="Docker Images">

<VersionVerificationTable entries={[
  { version: "1.82.3", sha256: "0a571da849db5f9c3cf3fead2ffbf1df982eebff7e7b38b46dbec3f640dafdbb", gitCommit: "61409275c8d8" },
  { version: "1.82.3-stable", sha256: "0c2b2a0ad3e50af1702fc493ecd07f22a5180b6d1cfb169440b429b40e340e29", gitCommit: "61409275c8d8" },
  { version: "1.82.0-stable", sha256: "71bf7283767ca436edcfa9f1f26c1743487b5fa29736c61c3eb6977776007c42", gitCommit: "97947c254252" },
  { version: "1.81.15", sha256: "303c31af87e7915e7b34d6c4d55a6ac753ef947a5deaa899e9ccfd3d1d58f7c2", gitCommit: "20bf3aa8070a" },
  { version: "1.81.14-stable", sha256: "a34f9758048231817d799b703fb998e40e2a5cbabb89ab95039fc30798f01b3c", gitCommit: "0435375b1271" },
  { version: "1.81.13", sha256: "a876f3f22f9b6fd481c9091c44a8a893d81c172d66dc2749298dcd3dc4a3d6f0", gitCommit: "cc957a19a560" },
  { version: "1.81.12-stable", sha256: "e24022878ccc87f57d808ac9304f18b87b8359e6556746d81cc20a5dc85f423a", gitCommit: "ba0d541b1982" },
  { version: "1.81.9-stable", sha256: "262e53d7702ed82579717faff0b08f7c0b7e9973a6406cfcc0e4af7826327627", gitCommit: "a09d3e9162eb" },
  { version: "1.81.3-stable", sha256: "dff82ccc32fb648927c090607887401c7e8ec814fe7c951beb95fe51073ca02b", gitCommit: "61ed8f9e0355" },
  { version: "1.81.0-stable", sha256: "f4913297d1bb3dc373eb8911a5ac816b597be9b5e08a91636b6c2786dd572aa8", gitCommit: "790a5ce0b323" },
  { version: "1.80.15-stable", sha256: "0b4ec3861e978b4aa254f4070f292cd345496a5fb59c72e1ee21cd6db94b670b", gitCommit: "17c8d8d109b5" },
  { version: "1.80.11-stable", sha256: "4068108d9101cd2affba3924310fd7f34f23d14e36dd4853733898b9e04d81ca", gitCommit: "57e07bddd341" },
  { version: "1.80.8-stable", sha256: "0304c2eb1f3cf54262d1b4e0629487232bab459e95b99a21e5810231d2b27021", gitCommit: "3381d63152f8" },
  { version: "1.80.5-stable", sha256: "a89e173135fff96af4b5b91ea31845164eadcf6497c82adeb64c36a23c8a3d11", gitCommit: "6c49b95a4ab7" },
  { version: "1.80.0-stable", sha256: "a3416f4cd0c896c94a1f526d872ff6c19bee22ff4afcdcc6f9ff690707900176", gitCommit: "98365205acd0" },
  { version: "1.79.3-stable", sha256: "27aae83d6ab6cb0b63bf8179e375ce0e11f5cfef51f2675b0c1e60c6f546dbc1", gitCommit: "c0548542d4a9" },
  { version: "1.79.1-stable", sha256: "7780d29a9543c4ce762430db7dfb0640105f7357fc38e35bf3fb7bbb1e6ba63f", gitCommit: "c217bddb59ba" },
  { version: "1.79.0-stable", sha256: "32bf6ac059a56641e11e4712f63b8467c295f988b6c160dc7229660417ee44bd", gitCommit: "8d495f56a9cc" },
  { version: "1.78.5-stable", sha256: "d5e607648eafa15edc63b0b1a5ed01f8b31a1fa0c80f7d25b252ae18a593ee29", gitCommit: "c471bf1f16c2" },
  { version: "1.78.0-stable", sha256: "7a56b32dc7153763d31c0a056123dc878a598959935d8c7daacb1fca5272c205", gitCommit: "5fde83d9f154" },
]} />

</TabItem>
</Tabs>


## Questions and support

If you believe your systems may be affected, contact us immediately:

- **Security:** `security@berri.ai`
- **Support:** `support@berri.ai`
- **Slack:** Reach out to the LiteLLM team directly

For real-time updates, follow [LiteLLM (YC W23) on X](https://x.com/LiteLLM).


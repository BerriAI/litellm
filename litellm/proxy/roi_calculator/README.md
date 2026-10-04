# ROI Calculator

The dashboard compares merged pull or merge requests, elapsed time from opening to merge, new bug and regression issues, and recorded gateway spend over 7, 28, or 90 complete UTC days. Compare against the immediately preceding period of the same length or the same-length period last year

The calculator combines repository activity with spend recorded by the gateway. Spend per merged change is a person's recorded gateway spend during the period divided by their matched merged changes. To track a branch's AI cost, send repository and branch tags with each request

## Connect repositories

Use **Preview sample report** beside the title to explore the dashboard before connecting repositories. Sample periods, engineer details, quality signals, and branch spend work without changing your connections or live report. **Exit demo** returns to your report or setup

Open `/ui/roi-calculator/`, choose GitHub or GitLab, then connect with an app or access token. Select several repositories and start the sync. Use **Add connection** to keep both providers connected. Each provider and API host retains its credentials, repositories, and identity mappings, and the report combines their activity while counting each person’s gateway spend once. Public repositories also accept an empty token, subject to the provider's anonymous API limits

For GitHub tokens, grant read access to metadata, pull requests and issues. GitLab tokens require `read_api`. Self-hosted instances use their API URL, for example `https://git.example.com/api/v4`

## Configure app authorization

Register a GitHub App with read-only repository permissions for metadata, pull requests and issues. Enable expiring user access tokens and leave authorization during installation disabled, since the gateway starts authorization after installation. Generate a private key in the app settings to allow installation and store it securely. The gateway uses a generated client secret for authorization and does not need the private key

Register a confidential GitLab OAuth application with `read_api` and `read_user` scopes

Set `PROXY_BASE_URL` to the gateway's public URL. The callback URLs are `<PROXY_BASE_URL>/roi-calculator/observed/oauth/github/callback` and `<PROXY_BASE_URL>/roi-calculator/observed/oauth/gitlab/callback`

Set the GitHub App setup URL to `<PROXY_BASE_URL>/roi-calculator/observed/oauth/github/installed`, enable **Redirect on update**, and set `LITELLM_ROI_GITHUB_APP_SLUG` to its URL slug. The first connection then starts with repository installation and continues to user authorization

Set `LITELLM_ROI_GITHUB_CLIENT_ID` and `LITELLM_ROI_GITHUB_CLIENT_SECRET` for GitHub, or `LITELLM_ROI_GITLAB_CLIENT_ID` and `LITELLM_ROI_GITLAB_CLIENT_SECRET` for GitLab. For a self-hosted provider, set `LITELLM_ROI_GITHUB_URL` or `LITELLM_ROI_GITLAB_URL` to its base URL without the API suffix

The gateway encrypts access and refresh tokens using its configured encryption key. Authorization uses PKCE and an expiring, single-use state tied to an HTTP-only browser cookie. Refreshes are coordinated across gateway workers

## Link people

Use **Link accounts** to associate several current or historical usernames with one internal email. Each connection has a separate username field, so a GitHub username never matches a GitLab user implicitly. Saving immediately recalculates the report without fetching repositories again. Public profile emails match automatically when they resolve unambiguously to an internal user

**Matched people only** is on by default for people, merged changes, and branch lists. Turn it off to include outside contributors and their branches. Matching depends on the linked internal account, even when no spend was recorded. This switch filters the lists; summary metrics and quality signals still cover all selected repositories

Agent-authored changes count for a person only when the supported agent metadata explicitly names a requester. Repository issue counts and revert titles are quality signals, not an individual defect score

Bug and regression counts combine repositories with issue tracking enabled. They remain unavailable when none of the selected repositories has issue tracking enabled

## Sync behavior

The default refresh interval is daily and applies to every connection in the workspace. Adding or editing a connection preserves it unless `update_interval_minutes` is supplied. The observed settings API accepts `update_interval_minutes: 0` for manual updates. A cancelled or failed sync preserves the last complete report

Existing settings retain `report_mode: legacy` and their scheduled reports until an administrator saves a connection, authorizes an app, or starts an observed sync. Reading the new dashboard alone does not change the mode. The legacy settings API can explicitly select `report_mode: legacy` again

GitHub collection splits large searches into smaller date ranges to avoid its search-result limit. Both providers validate pagination and reject incomplete responses instead of publishing partial counts


## Branch request tags

Send `repo:github.com/owner/repo` or `repo:gitlab.com/group/project` together with `branch:feature/name` in `metadata.tags`, top-level `tags`, or the comma-separated `x-litellm-tags` header. The tags must identify the source repository and branch, including forks

The report sums recorded requests inside its UTC dates. A branch cost is assigned to a merged change only when that source branch matches one change in the period. Reused branches stay visible in Branch spend without duplicating costs across changes. No retained tagged requests means unknown cost; a recorded zero remains zero

An empty repository produces a successful report with zero merged changes and no merge duration or spend-per-change ratio

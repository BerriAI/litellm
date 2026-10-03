# ROI Calculator

The dashboard compares merged pull or merge requests, elapsed time from opening to merge, new bug and regression issues, and recorded gateway spend over 28 complete UTC days. Compare against the previous 28 days or the same period last year

No estimator model is required. Spend per person is their recorded gateway spend during the period divided by their matched merged changes. Branch spend remains separate and uses repository and branch tags on spend logs

## Connect repositories

Open `/ui/roi-calculator/`, choose GitHub or GitLab, then connect with an app or access token. Select repositories and start the sync. Public repositories also accept an empty token, subject to the provider's anonymous API limits

For GitHub tokens, grant read access to metadata, pull requests and issues. GitLab tokens require `read_api`. Self-hosted instances use their API URL, for example `https://git.example.com/api/v4`

## Configure app authorization

Register a GitHub App with read-only repository permissions for metadata, pull requests and issues. Enable expiring user access tokens and leave authorization during installation disabled, since the gateway starts authorization after installation

Register a confidential GitLab OAuth application with `read_api` and `read_user` scopes

Set `PROXY_BASE_URL` to the gateway's public URL. The callback URLs are `<PROXY_BASE_URL>/roi-calculator/observed/oauth/github/callback` and `<PROXY_BASE_URL>/roi-calculator/observed/oauth/gitlab/callback`

Set the GitHub App setup URL to `<PROXY_BASE_URL>/roi-calculator/observed/oauth/github/installed`, enable **Redirect on update**, and set `LITELLM_ROI_GITHUB_APP_SLUG` to its URL slug. The first connection then starts with repository installation and continues to user authorization

Set `LITELLM_ROI_GITHUB_CLIENT_ID` and `LITELLM_ROI_GITHUB_CLIENT_SECRET` for GitHub, or `LITELLM_ROI_GITLAB_CLIENT_ID` and `LITELLM_ROI_GITLAB_CLIENT_SECRET` for GitLab. For a self-hosted provider, set `LITELLM_ROI_GITHUB_URL` or `LITELLM_ROI_GITLAB_URL` to its base URL without the API suffix

The gateway encrypts access and refresh tokens using its configured encryption key. Authorization uses PKCE and an expiring, single-use state tied to an HTTP-only browser cookie. Refreshes are coordinated across gateway workers

## Link people

Use **Link accounts** to associate several current or historical usernames with one internal email. Saving immediately recalculates the report without fetching repositories again. Public profile emails match automatically when they resolve unambiguously to an internal user

Agent-authored changes count for a person only when the supported agent metadata explicitly names a requester. Repository issue counts and revert titles are quality signals, not an individual defect score

## Sync behavior

The default refresh interval is daily. The observed settings API accepts `update_interval_minutes: 0` for manual updates. A cancelled or failed sync preserves the last complete report

GitHub collection splits large searches into smaller date ranges to avoid its search-result limit. Both providers validate pagination and reject incomplete responses instead of publishing partial counts

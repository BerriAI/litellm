# Troubleshooting

## Common issues and solutions

### "OAuth credentials not found"
```
OAuth credentials not found. Please provide credentials using one of these methods:
1. Config directory (recommended):
   Place your gcp-oauth.keys.json file in: ~/.config/google-drive-mcp/
2. Environment variable:
   export GOOGLE_DRIVE_OAUTH_CREDENTIALS="/path/to/gcp-oauth.keys.json"
```

**Solution:**
- Download credentials from Google Cloud Console
- Place the file in `~/.config/google-drive-mcp/gcp-oauth.keys.json` (recommended), or set the environment variable
- Ensure the file has proper read permissions

### "Authentication failed" or Browser doesn't open
**Possible causes:**
1. **Wrong credential type**: Must be "Desktop app", not "Web application"
2. **Port blocked**: Ports 3000-3004 must be available (or custom range if `GOOGLE_DRIVE_MCP_AUTH_PORT` is set)
3. **Test user not added**: Add your email in OAuth consent screen
4. **`redirect_uri_mismatch` (Web application clients only)**: The callback redirect URI uses the loopback IP `http://127.0.0.1:<port>/oauth2callback`. Switch to a "Desktop app" client (recommended), or add `http://127.0.0.1:3000/oauth2callback` … `http://127.0.0.1:3004/oauth2callback` (plus any custom `GOOGLE_DRIVE_MCP_AUTH_PORT` range) as authorized redirect URIs

**Solution:**
```bash
# Check if ports are in use
lsof -i :3000-3004

# Option 1: Kill processes if needed
kill -9 <PID>

# Option 2: Use a different port range
export GOOGLE_DRIVE_MCP_AUTH_PORT=3100

# Re-run authentication
npx -y @piotr-agier/google-drive-mcp auth
```

### "Tokens expired" or "Invalid grant"
**For Google OAuth apps in "Testing" status:**
- Google automatically expires refresh tokens after 7 days
- You'll need to re-authenticate weekly until you publish your app

**Solution:**
```bash
# Clear old tokens and re-authenticate
rm ~/.config/google-drive-mcp/tokens.json
npx -y @piotr-agier/google-drive-mcp auth
```

**For production:**
- Move app to "Published" status in Google Cloud Console
- Complete OAuth verification process

### "Login Required" error even with valid tokens
**If you updated the OAuth scopes but still get errors:**
- Google caches app authorizations even after removing local tokens
- The app might be using old/limited scopes

**Solution:**
1. Go to [Google Account Permissions](https://myaccount.google.com/permissions)
2. Find and remove access for "Google Drive MCP"
3. Clear local tokens: `rm ~/.config/google-drive-mcp/tokens.json`
4. Re-authenticate to grant all required scopes
5. Verify the consent screen shows ALL scopes including full Drive access

### `search` returns 0 results and Shared Drives are invisible, despite a valid token
**Symptom:** `search` returns `Found 0 files (ordered by modifiedTime desc):` (even for My Drive), `listSharedDrives` shows none, and `authTestFileAccess` reports "File not found" — yet `authGetStatus` shows a valid token with full Drive scope, and the same account works via the Drive REST API directly.

**Most common cause:** an environment variable is silently overriding your interactive OAuth `tokens.json`. Service-account mode (`GOOGLE_APPLICATION_CREDENTIALS`) and external-token mode (`GOOGLE_DRIVE_MCP_ACCESS_TOKEN`) take **priority** over `tokens.json` whenever they are present in the server's environment. If the process inherits one of these (common when `gcloud`, CI runners, or other Google tooling set `GOOGLE_APPLICATION_CREDENTIALS` globally), every call runs as that other identity — often an empty service account with no files and no Shared Drive membership — which returns empty results with no error.

**Diagnose:**
```bash
# Run authGetStatus — it reports the ACTIVE auth mode and the EFFECTIVE identity.
# If authMode is "service_account"/"external_token" (not "oauth"), or the reported
# identity email is not your account, that env var is the culprit.
```
Check your environment for `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_DRIVE_MCP_ACCESS_TOKEN` (Windows: `echo %GOOGLE_APPLICATION_CREDENTIALS%` / `$env:GOOGLE_APPLICATION_CREDENTIALS`). The server also logs a warning on startup when a present `tokens.json` is being bypassed.

**Solution:** unset the overriding variable for the MCP server's environment (or, if you intend to use a service account, grant its email address access to the files/Shared Drives you need — and set `GOOGLE_DRIVE_MCP_SUBJECT` for domain-wide delegation if you need to act as a real Workspace user).

### "API not enabled" errors
```
Error: Google Sheets API has not been used in project...
```

**Solution:**
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Select your project
3. Navigate to "APIs & Services" > "Library"
4. Search and enable the missing API
5. Wait 1-2 minutes for propagation

### "Insufficient permissions"
**Check scopes in your credentials:**
- Need drive.file or drive scope
- Need docs, sheets, slides scopes for respective services

**Solution:**
- Re-create OAuth credentials with correct scopes
- Re-authenticate after updating credentials

### Rate Limiting (429 errors)
**Google API Quotas:**
- Drive API: 12,000 requests per minute
- Docs/Sheets/Slides: 300 requests per minute

**Solution:**
- Implement exponential backoff
- Batch operations where possible
- Check quota usage in Google Cloud Console

## Docker-Specific Issues

### "Authentication required" in Docker
**Problem:** The MCP server in Docker shows authentication errors even though you have valid tokens.

**Cause:** OAuth flow requires browser access, which isn't available in Docker containers.

**Solution:**
```bash
# 1. Authenticate outside Docker first
npx -y @piotr-agier/google-drive-mcp auth

# 2. Verify tokens exist
ls -la ~/.config/google-drive-mcp/tokens.json

# 3. Rebuild the image and restart the client
docker build -t google-drive-mcp .
# The client will invoke scripts/docker-mcp.sh, which auto-replaces the stale container
```

### "npm ci failed" during Docker build
**Problem:** Docker build fails with `tsc: not found` or similar errors.

**Solution:**
```bash
# Build the project locally first
npm install
npm run build

# Then build Docker image
docker build -t google-drive-mcp .
```

The Dockerfile expects the `dist/` directory to exist from your local build.

### "Google's token endpoint did not respond within 15000ms"
**Problem:** An access-token refresh stalled. The server gives each refresh attempt 15 seconds and retries once, then fails the call with this message rather than leaving every call waiting on the stalled request. It is a connectivity problem between the server and `oauth2.googleapis.com` (a proxy, a firewall, or a dropped connection), not a revoked grant.

**Solution:** Retry the call; a transient stall clears on its own. If it keeps happening behind a slow proxy, raise the limit with `--token-refresh-timeout=<ms>` (or `GOOGLE_DRIVE_MCP_TOKEN_REFRESH_TIMEOUT`), and check that the server can reach `https://oauth2.googleapis.com/token` directly.

### "Token refresh failed" in Docker
**Problem:** Tokens can't refresh inside the container.

**Solution:** Ensure the token file is mounted with write permissions:
```bash
# Correct: tokens can be updated
-v "$HOME/.config/google-drive-mcp/tokens.json":/config/tokens.json

# Wrong: read-only mount prevents token refresh
-v "$HOME/.config/google-drive-mcp/tokens.json":/config/tokens.json:ro
```

## Getting Help

1. **Check logs**: Server logs errors to stderr
2. **Verify setup**: Run `npx -y @piotr-agier/google-drive-mcp help`
3. **Test auth**: Run `npx -y @piotr-agier/google-drive-mcp auth`
4. **Report issues**: [GitHub Issues](https://github.com/piotr-agier/google-drive-mcp/issues)

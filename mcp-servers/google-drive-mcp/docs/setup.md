# Setup

This guide covers the Google Cloud project, OAuth credentials, and package installation required before connecting an MCP client. For a condensed path, start with the [README quick start](../README.md#quick-start).

## Requirements

- **Node.js**: Version 18 or higher (LTS recommended)
- **Google Cloud Project**: With the following APIs enabled:
  - Google Drive API
  - Google Docs API
  - Google Sheets API
  - Google Slides API
  - Google Calendar API
- **OAuth 2.0 Credentials**: Desktop application type (Client ID only - no client secret required)

## Google Cloud Setup

### 1. Create a Google Cloud Project
- Go to the [Google Cloud Console](https://console.cloud.google.com)
- Click "Select a project" > "New Project"
- Name your project (e.g., "Google Drive MCP")
- Note the Project ID for later

### 2. Enable Required APIs
- In your project, go to "APIs & Services" > "Library"
- Search for and enable each of these APIs:
  - **Google Drive API**
  - **Google Docs API**
  - **Google Sheets API**
  - **Google Slides API**
  - **Google Calendar API**
- Wait for each API to be enabled before proceeding

### 3. Configure OAuth Consent Screen
- Go to "APIs & Services" > "OAuth consent screen"
- Under 'Branding' fill in the required fields:
  - App name: "My Personal Google Drive MCP"
  - User support email: Your email
  - Developer contact: Your email
- Under 'Audience':
  - Choose "External" (default choice) or "Internal" for Google Workspace accounts
  - Add your email as a test user
- Under 'Data Access' add scopes. The recommended set of scopes for best user experience is the following:
  - `./auth/drive.file`
  - `.../auth/documents`
  - `.../auth/spreadsheets`
  - `.../auth/presentations`
  - `.../auth/drive`
  - `.../auth/drive.readonly`
  - `.../auth/calendar`
  - `.../auth/calendar.events`

### 4. Create OAuth 2.0 Credentials
- Go to "APIs & Services" > "Credentials"
- Click "+ CREATE CREDENTIALS" > "OAuth client ID"
- Application type: **Desktop app** (Important!)
- Name: "Google Drive MCP Client"
- Click "Create"
- Download the JSON file
- Rename it to `gcp-oauth.keys.json`

### 5. Place the Credentials File

The server looks for `gcp-oauth.keys.json` in this order and uses the first location that exists:

1. The path in `GOOGLE_DRIVE_OAUTH_CREDENTIALS`, if that variable is set:
   ```bash
   export GOOGLE_DRIVE_OAUTH_CREDENTIALS="/path/to/gcp-oauth.keys.json"
   ```
2. The config directory (recommended — it works with `npx`, global installs, and local checkouts alike):
   ```text
   ~/.config/google-drive-mcp/gcp-oauth.keys.json
   ```
   `XDG_CONFIG_HOME` replaces `~/.config` when it is set.
3. The project root, as a legacy fallback. This works for local development but is unreliable with `npx` or global installs.

If none of these contain the file, the server exits with "OAuth credentials not found". See [Authentication](authentication.md#oauth-credentials-configuration) for the full credential and token model.

## Installation

### Option 1: Use with npx (Recommended)

You can run the server directly without installation:

```bash
# Run the server (authentication happens automatically on first run)
npx -y @piotr-agier/google-drive-mcp

# Optional: Run authentication manually if needed
npx -y @piotr-agier/google-drive-mcp auth
```

### Option 2: Local Installation

1. Clone and install:
   ```bash
   git clone https://github.com/piotr-agier/google-drive-mcp.git
   cd google-drive-mcp
   npm install
   ```

2. Set up credentials:
   ```bash
   # Copy the example file
   cp gcp-oauth.keys.example.json gcp-oauth.keys.json

   # Edit gcp-oauth.keys.json with your OAuth client ID
   ```

3. Authenticate (optional):
   ```bash
   npm run auth
   ```

   Note: Authentication happens automatically on first run of an MCP client if you skip this step.

Next, add the server to an MCP client using [Client configuration](clients.md#client-configuration).

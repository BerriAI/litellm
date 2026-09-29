import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { fileURLToPath } from 'url';

// Walk up from `startDir` to the nearest ancestor containing a package.json and
// return it. This is robust to the build layout — bundled output is
// `dist/index.js`, an unbundled build would be `dist/auth/utils.js`, and an
// installed package lives at `node_modules/<pkg>/dist/index.js`; all resolve to
// the package root. A previous hard-coded "up two levels" assumed the unbundled
// `dist/auth/utils.js` path, so with the bundled `dist/index.js` it resolved to
// the package's PARENT directory instead and the project-root fallback for
// `gcp-oauth.keys.json` never matched. Exported for testing.
export function findPackageRoot(startDir: string): string {
  let dir = startDir;
  const fsRoot = path.parse(dir).root;
  while (true) {
    if (fs.existsSync(path.join(dir, 'package.json'))) return dir;
    if (dir === fsRoot) break;
    dir = path.dirname(dir);
  }
  // Fallback to the legacy behavior if no package.json is found on the path.
  return path.resolve(startDir, '..', '..');
}

// Helper to get the project root directory reliably.
function getProjectRoot(): string {
  return findPackageRoot(path.dirname(fileURLToPath(import.meta.url)));
}

// Returns the config directory for google-drive-mcp, following XDG Base Directory spec.
function getConfigDir(): string {
  const configHome = process.env.XDG_CONFIG_HOME ||
    path.join(os.homedir(), '.config');
  return path.join(configHome, 'google-drive-mcp');
}

// Returns the absolute path for the saved token file.
// Uses XDG Base Directory spec with fallback to home directory
export function getSecureTokenPath(): string {
  // Check for custom token path environment variable first
  const customTokenPath = process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH;
  if (customTokenPath) {
    return path.resolve(customTokenPath);
  }

  return path.join(getConfigDir(), 'tokens.json');
}

// Returns the legacy token path for backward compatibility
export function getLegacyTokenPath(): string {
  const projectRoot = getProjectRoot();
  return path.join(projectRoot, ".gcp-saved-tokens.json");
}

// Additional legacy paths to check
export function getAdditionalLegacyPaths(): string[] {
  return [
    process.env.GOOGLE_TOKEN_PATH,
    path.join(process.cwd(), 'google-tokens.json'),
    path.join(process.cwd(), '.gcp-saved-tokens.json')
  ].filter(Boolean) as string[];
}

// Returns all candidate paths for the credentials file, in priority order:
// 1. Environment variable GOOGLE_DRIVE_OAUTH_CREDENTIALS (highest priority)
// 2. Config directory ~/.config/google-drive-mcp/gcp-oauth.keys.json
// 3. Project root directory (legacy fallback)
export function getKeysFilePaths(): string[] {
  const paths: string[] = [];

  const envCredentialsPath = process.env.GOOGLE_DRIVE_OAUTH_CREDENTIALS;
  if (envCredentialsPath) {
    paths.push(path.resolve(envCredentialsPath));
  }

  paths.push(path.join(getConfigDir(), 'gcp-oauth.keys.json'));

  const projectRoot = getProjectRoot();
  paths.push(path.join(projectRoot, "gcp-oauth.keys.json"));

  return paths;
}

/**
 * Render an error for logging without leaking credential material.
 *
 * Passing raw errors to console.error is unsafe in the auth layer: gaxios
 * errors carry the full request config (a token-refresh POST body embeds the
 * refresh token and client secret), and JSON.parse SyntaxErrors echo fragments
 * of the unparseable source — which for a token or credentials file is secret
 * material. Extract only known-safe fields.
 */
export function describeErrorForLog(err: unknown): string {
  if (err instanceof SyntaxError) return 'SyntaxError: invalid JSON';
  if (err === null || typeof err !== 'object') return String(err);
  const e = err as {
    name?: unknown;
    message?: unknown;
    code?: unknown;
    response?: { status?: unknown; data?: { error?: unknown; error_description?: unknown } };
  };
  const parts: string[] = [];
  if (typeof e.message === 'string' && e.message.length > 0) parts.push(e.message);
  else if (typeof e.name === 'string') parts.push(e.name);
  if (e.code !== undefined) parts.push(`code=${String(e.code)}`);
  if (e.response?.status !== undefined) parts.push(`status=${String(e.response.status)}`);
  if (typeof e.response?.data?.error === 'string') parts.push(`error=${e.response.data.error}`);
  if (typeof e.response?.data?.error_description === 'string') {
    parts.push(`error_description=${e.response.data.error_description}`);
  }
  return parts.length > 0 ? parts.join(' ') : 'unknown error';
}

// Interface for OAuth credentials
export interface OAuthCredentials {
  client_id: string;
  client_secret?: string;
  redirect_uris?: string[];
}

// Generate helpful error message for missing credentials
export function generateCredentialsErrorMessage(): string {
  const configDir = getConfigDir();

  return `
OAuth credentials not found. Please provide credentials using one of these methods:

1. Config directory (recommended):
   Place your gcp-oauth.keys.json file in: ${configDir}/

2. Environment variable:
   Set GOOGLE_DRIVE_OAUTH_CREDENTIALS to the path of your credentials file:
   export GOOGLE_DRIVE_OAUTH_CREDENTIALS="/path/to/gcp-oauth.keys.json"

Token storage:
- Tokens are saved to: ${getSecureTokenPath()}
- To use a custom token location, set GOOGLE_DRIVE_MCP_TOKEN_PATH environment variable

To get OAuth credentials:
1. Go to the Google Cloud Console (https://console.cloud.google.com/)
2. Create or select a project
3. Enable the Google Drive, Docs, Sheets, and Slides APIs
4. Create OAuth 2.0 credentials (Desktop app type)
5. Download the credentials file as gcp-oauth.keys.json
`.trim();
}
// ---------------------------------------------------------------------------
// Shared OAuth scope constants & resolution
// ---------------------------------------------------------------------------

export const SCOPE_ALIASES: Record<string, string> = {
  drive: 'https://www.googleapis.com/auth/drive',
  'drive.file': 'https://www.googleapis.com/auth/drive.file',
  'drive.readonly': 'https://www.googleapis.com/auth/drive.readonly',
  documents: 'https://www.googleapis.com/auth/documents',
  spreadsheets: 'https://www.googleapis.com/auth/spreadsheets',
  presentations: 'https://www.googleapis.com/auth/presentations',
  calendar: 'https://www.googleapis.com/auth/calendar',
  'calendar.events': 'https://www.googleapis.com/auth/calendar.events',
};

export const SCOPE_PRESETS: Record<string, string[]> = {
  readonly: ['drive.readonly'],
  'content-editor': ['drive.file', 'documents', 'spreadsheets', 'presentations'],
  full: ['drive', 'documents', 'spreadsheets', 'presentations', 'calendar', 'calendar.events'],
};

export const DEFAULT_SCOPES: readonly string[] = [
  'drive', 'drive.file', 'drive.readonly',
  'documents', 'spreadsheets', 'presentations',
  'calendar', 'calendar.events',
].map((s) => SCOPE_ALIASES[s]);

/**
 * Scopes requested in addition to the user's DEFAULT_SCOPES when adding a new
 * account via `manage_accounts add`, so we can discover the account's email
 * and Google-stable subject id immediately after consent. Never added to
 * DEFAULT_SCOPES to avoid forcing re-consent on existing users.
 */
export const USERINFO_SCOPES: readonly string[] = [
  'openid',
  'https://www.googleapis.com/auth/userinfo.email',
];

/**
 * Resolve OAuth scopes from `GOOGLE_DRIVE_MCP_SCOPES` env var.
 * Accepts comma-separated aliases (e.g. "drive,documents") or full URLs.
 * Throws on unknown aliases so mis-configurations surface immediately.
 */
export function resolveOAuthScopes(): string[] {
  const raw = process.env.GOOGLE_DRIVE_MCP_SCOPES?.trim();
  if (!raw) return [...DEFAULT_SCOPES];

  const scopes = raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      if (SCOPE_ALIASES[s]) return SCOPE_ALIASES[s];
      if (s.startsWith('https://')) return s;
      const known = Object.keys(SCOPE_ALIASES).join(', ');
      throw new Error(
        `Unknown OAuth scope alias "${s}". Use a full URL (https://...) or one of: ${known}`
      );
    });

  if (scopes.length === 0) return [...DEFAULT_SCOPES];
  return [...new Set(scopes)];
}

/**
 * Scopes requested by `manage_accounts add`: the operator-configured OAuth scopes
 * (`GOOGLE_DRIVE_MCP_SCOPES` via `resolveOAuthScopes`) plus `USERINFO_SCOPES` so the
 * new account's email and stable subject id can be discovered right after consent.
 * Mirrors the boot flow, which also honors `GOOGLE_DRIVE_MCP_SCOPES`.
 */
export function resolveAddAccountScopes(): string[] {
  return [...resolveOAuthScopes(), ...USERINFO_SCOPES];
}

/**
 * Split an OAuth scope string (space/whitespace-separated per RFC 6749) into a
 * deduped list of individual scopes. The single source of truth for scope-string
 * tokenization across the resolver, store, and diagnostics.
 */
export function splitScopes(scope: string | null | undefined): string[] {
  if (!scope) return [];
  return [...new Set(scope.split(/\s+/).filter(Boolean))];
}

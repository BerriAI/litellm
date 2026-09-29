import type { drive_v3, calendar_v3 } from 'googleapis';
import type { google as GoogleApisType } from 'googleapis';
import type { RuntimeConfig } from './utils/cliArgs.js';
import type {
  AccountRecord,
  AccountTargeting,
  AuthMode,
  RedactedAccountView,
  ToolOpKind,
} from './auth/types.js';

export interface ToolResult {
  [key: string]: unknown;
  // `text` blocks carry a string; `image` blocks carry base64 `data` + `mimeType`
  // (an MCP image content block). Kept as a single widened shape for backward
  // compatibility with the many `{ type: "text", text }` construction sites.
  content: Array<{ type: string; text?: string; data?: string; mimeType?: string }>;
  isError?: boolean;
}

export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

export interface ToolContext {
  // Back-compat: resolves to the active (default) account's client.
  // New handlers should prefer `resolveAccount` + `getDriveFor` / `googleFor`.
  authClient: any;
  google: typeof GoogleApisType;
  getDrive: () => drive_v3.Drive;
  getCalendar: () => calendar_v3.Calendar;
  log: (message: string, data?: any) => void;
  resolvePath: (pathStr: string) => Promise<string>;
  resolveFolderId: (input: string | undefined) => Promise<string>;
  checkFileExists: (name: string, parentFolderId?: string) => Promise<string | null>;
  validateTextFileExtension: (name: string) => void;
  runtimeConfig: RuntimeConfig;

  /** Stable user identity for the currently acting principal. Team mode sets
   * this to the bearer's Google `sub`. Single-user mode leaves it undefined
   * (one local account, no per-user ID). Intended for log attribution; the
   * human-readable email is stamped onto payloads by `ctx.log` itself rather
   * than stored here. */
  userId?: string;

  // Multi-account surface (Phase 1: present on context but not yet consumed by tool
  // handlers — resolver always targets the sole default account).
  sessionId: string;
  resolveAccount: (
    input: string | string[] | undefined,
    kind: ToolOpKind,
    acceptableScopes: string[],
  ) => Promise<AccountTargeting>;
  getDriveFor: (account: AccountRecord) => Promise<drive_v3.Drive>;
  getCalendarFor: (account: AccountRecord) => Promise<calendar_v3.Calendar>;
  getAuthClientFor: (account: AccountRecord) => Promise<any>;

  /** Lifecycle API for `manage_accounts` — lives on ctx to avoid circular deps. */
  accountOps: AccountOps;
}

export interface AddAccountResult {
  /** URL the user visits to complete consent. Shown in tool output. */
  authUrl: string;
  /** Resolves when the OAuth callback lands and the record is persisted. */
  completion: Promise<AccountRecord>;
  /** Stops the embedded auth server (idempotent). */
  cancel: () => Promise<void>;
}

export interface AccountOps {
  mode: AuthMode;
  list(): RedactedAccountView[];
  getDefault(): string | undefined;
  /** Kick off an OAuth flow for a new account. Caller usually awaits completion. */
  add(alias: string, opts?: { openBrowser?: boolean }): Promise<AddAccountResult>;
  remove(alias: string): Promise<void>;
  setDefault(alias: string | null): Promise<void>;
}

export function errorResponse(message: string): ToolResult {
  return { content: [{ type: "text", text: `Error: ${message}` }], isError: true };
}

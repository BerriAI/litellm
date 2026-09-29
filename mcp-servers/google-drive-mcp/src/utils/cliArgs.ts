// cliArgs.ts
export interface RuntimeConfig {
  apiTimeout: number;
  /**
   * Per-attempt timeout for an OAuth token refresh (ms). Kept separate from
   * `apiTimeout`: a token POST is tiny and should fail fast, while API calls
   * may legitimately stream large uploads/downloads for minutes. 0 disables.
   */
  tokenRefreshTimeout: number;
  retryMax: number;
  retryBaseDelay: number;
  disableResources: boolean;
}

export const RUNTIME_DEFAULTS: Readonly<RuntimeConfig> = {
  apiTimeout: 120_000,
  tokenRefreshTimeout: 15_000,
  retryMax: 3,
  retryBaseDelay: 1_000,
  disableResources: false,
};

const DEFAULTS = RUNTIME_DEFAULTS;

function parseIntOr(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const n = parseInt(value, 10);
  // Allow 0: retryMax=0 disables retries, apiTimeout=0 disables the timeout.
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

// Deliberately omits enable/disable: on a negated flag (`--no-resources`) and a
// negated env var (`..._DISABLE_RESOURCES`) those words form a double negative
// (`--no-resources=disabled` reading as "resources enabled"). on/off covers the
// same intent unambiguously under negation.
const TRUTHY = new Set(['1', 'true', 'yes', 'on']);
const FALSY = new Set(['0', 'false', 'no', 'off']);

export function parseBoolEnv(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  const v = value.trim().toLowerCase();
  if (TRUTHY.has(v)) return true;
  if (FALSY.has(v)) return false;
  return fallback;
}

export function loadRuntimeConfig(argv: string[] = process.argv.slice(2)): RuntimeConfig {
  const cfg: RuntimeConfig = { ...DEFAULTS };

  // Env vars fallback
  cfg.apiTimeout = parseIntOr(process.env.GOOGLE_DRIVE_MCP_API_TIMEOUT, cfg.apiTimeout);
  cfg.tokenRefreshTimeout = parseIntOr(
    process.env.GOOGLE_DRIVE_MCP_TOKEN_REFRESH_TIMEOUT,
    cfg.tokenRefreshTimeout,
  );
  cfg.retryMax = parseIntOr(process.env.GOOGLE_DRIVE_MCP_RETRY_MAX, cfg.retryMax);
  cfg.retryBaseDelay = parseIntOr(process.env.GOOGLE_DRIVE_MCP_RETRY_BASE_DELAY, cfg.retryBaseDelay);
  cfg.disableResources = parseBoolEnv(process.env.GOOGLE_DRIVE_MCP_DISABLE_RESOURCES, cfg.disableResources);

  // CLI args override (numeric flags: --key=value; boolean flags: presence)
  for (const arg of argv) {
    if (arg.startsWith('--api-timeout=')) {
      cfg.apiTimeout = parseIntOr(arg.split('=')[1], cfg.apiTimeout);
    } else if (arg.startsWith('--token-refresh-timeout=')) {
      cfg.tokenRefreshTimeout = parseIntOr(arg.split('=')[1], cfg.tokenRefreshTimeout);
    } else if (arg.startsWith('--retry-max=')) {
      cfg.retryMax = parseIntOr(arg.split('=')[1], cfg.retryMax);
    } else if (arg.startsWith('--retry-base-delay=')) {
      cfg.retryBaseDelay = parseIntOr(arg.split('=')[1], cfg.retryBaseDelay);
    } else if (arg === '--no-resources') {
      cfg.disableResources = true;
    } else if (arg.startsWith('--no-resources=')) {
      // Bare --no-resources disables; --no-resources=false re-enables (e.g. to
      // override a truthy GOOGLE_DRIVE_MCP_DISABLE_RESOURCES). An unrecognized
      // value falls back to the bare-flag intent (disable).
      cfg.disableResources = parseBoolEnv(arg.split('=')[1], true);
    }
  }

  return cfg;
}

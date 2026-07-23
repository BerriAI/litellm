/**
 * Thin fetch wrapper for the LiteLLM proxy.
 *
 * Responsibilities:
 *  - Resolve apiKey + baseUrl from explicit args or env
 *  - Inject `Authorization: Bearer <apiKey>`
 *  - Retry retryable errors (5xx, 429) with exponential backoff
 *  - Normalize errors to `LiteLLMAgentError`
 *  - Translate between the SDK's camelCase public API and the backend's
 *    snake_case wire format. Request bodies are converted with
 *    `camelToSnake` before serialization; response JSON is converted with
 *    `snakeToCamel` before being returned to callers. Only object keys are
 *    rewritten — values pass through unchanged.
 */

import { ClientOptions, LiteLLMAgentError } from "../types.js";

/**
 * Recursively convert object keys from snake_case to camelCase.
 * Walks plain objects and arrays; leaves Date/Buffer/etc. untouched.
 * Single-word keys (no underscores) pass through unchanged.
 */
export function snakeToCamel(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(snakeToCamel);
  }
  if (isPlainObject(value)) {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      out[snakeToCamelKey(k)] = snakeToCamel(v);
    }
    return out;
  }
  return value;
}

/**
 * Recursively convert object keys from camelCase to snake_case.
 * Walks plain objects and arrays; leaves Date/Buffer/etc. untouched.
 * Single-word keys (no uppercase) pass through unchanged.
 */
export function camelToSnake(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(camelToSnake);
  }
  if (isPlainObject(value)) {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      out[camelToSnakeKey(k)] = camelToSnake(v);
    }
    return out;
  }
  return value;
}

function snakeToCamelKey(key: string): string {
  if (!key.includes("_")) return key;
  return key.replace(/_([a-zA-Z0-9])/g, (_, c: string) => c.toUpperCase());
}

function camelToSnakeKey(key: string): string {
  // Only rewrite if there's at least one uppercase letter to convert.
  if (!/[A-Z]/.test(key)) return key;
  return key.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  if (v === null || typeof v !== "object") return false;
  const proto = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null;
}

const DEFAULT_BASE_URL = "https://api.litellm.ai";
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_RETRIES = 3;
/** Upper bound for honoring `Retry-After` headers, to defend against a
 * misbehaving or adversarial server returning extreme values. */
const MAX_RETRY_AFTER_MS = 60_000;

export interface ResolvedClient {
  apiKey: string;
  baseUrl: string;
  fetch: typeof fetch;
  timeoutMs: number;
  maxRetries: number;
}

export function resolveClient(options: ClientOptions = {}): ResolvedClient {
  const apiKey =
    options.apiKey ??
    (typeof process !== "undefined" ? process.env?.LITELLM_API_KEY : undefined);

  if (!apiKey) {
    throw new LiteLLMAgentError(
      "Missing apiKey. Pass `apiKey` explicitly or set LITELLM_API_KEY in the environment.",
      { code: "missing_api_key" },
    );
  }

  const baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
  return {
    apiKey,
    baseUrl,
    fetch: options.fetch ?? globalThis.fetch.bind(globalThis),
    timeoutMs: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    maxRetries: options.maxRetries ?? DEFAULT_MAX_RETRIES,
  };
}

export interface RequestOpts {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  path: string;
  body?: unknown;
  query?: Record<string, string | number | undefined>;
  headers?: Record<string, string>;
  /** If true, the response is returned raw (used for SSE). */
  stream?: boolean;
  /** Abort signal forwarded to fetch. */
  signal?: AbortSignal;
}

export async function request(
  client: ResolvedClient,
  opts: RequestOpts,
): Promise<Response> {
  const url = buildUrl(client.baseUrl, opts.path, opts.query);
  const headers: Record<string, string> = {
    Authorization: `Bearer ${client.apiKey}`,
    Accept: opts.stream ? "text/event-stream" : "application/json",
    ...(opts.headers ?? {}),
  };
  if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  const init: RequestInit = {
    method: opts.method ?? "GET",
    headers,
    body:
      opts.body !== undefined
        ? JSON.stringify(camelToSnake(opts.body))
        : undefined,
    signal: opts.signal,
  };

  let lastError: unknown;
  for (let attempt = 0; attempt <= client.maxRetries; attempt++) {
    try {
      const res = await withTimeout(client.fetch(url, init), client.timeoutMs);
      if (!res.ok) {
        const err = await toAgentError(res);
        if (err.retryable && attempt < client.maxRetries) {
          await sleep(backoffMs(attempt, res));
          continue;
        }
        throw err;
      }
      return res;
    } catch (e) {
      lastError = e;
      if (e instanceof LiteLLMAgentError) {
        if (!e.retryable || attempt === client.maxRetries) throw e;
        await sleep(backoffMs(attempt));
        continue;
      }
      // Network errors are retryable.
      if (attempt === client.maxRetries) throw e;
      await sleep(backoffMs(attempt));
    }
  }
  throw lastError;
}

export async function requestJson<T>(
  client: ResolvedClient,
  opts: RequestOpts,
): Promise<T> {
  const res = await request(client, opts);
  if (res.status === 204) return undefined as unknown as T;
  const parsed = await res.json();
  return snakeToCamel(parsed) as T;
}

function buildUrl(
  baseUrl: string,
  path: string,
  query?: Record<string, string | number | undefined>,
): string {
  const url = new URL(path.startsWith("/") ? path : `/${path}`, baseUrl + "/");
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined) continue;
      url.searchParams.set(k, String(v));
    }
  }
  return url.toString();
}

async function toAgentError(res: Response): Promise<LiteLLMAgentError> {
  const status = res.status;
  let code = `http_${status}`;
  let message = `HTTP ${status}`;
  try {
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      const body = (await res.json()) as {
        error?: { code?: string; message?: string };
        detail?: unknown;
      };
      if (body?.error?.code) code = body.error.code;
      if (body?.error?.message) message = body.error.message;
      else if (typeof body?.detail === "string") message = body.detail;
    } else {
      const text = await res.text();
      if (text) message = text.slice(0, 500);
    }
  } catch {
    // ignore body-parse failures
  }
  const retryable = status >= 500 || status === 429;
  return new LiteLLMAgentError(message, { code, status, retryable });
}

function backoffMs(attempt: number, res?: Response): number {
  if (res) {
    const ra = res.headers.get("retry-after");
    if (ra) {
      const n = Number(ra);
      if (!Number.isNaN(n)) {
        return Math.min(MAX_RETRY_AFTER_MS, Math.max(0, n * 1000));
      }
    }
  }
  // 250ms, 500ms, 1s, 2s, ...
  return 250 * Math.pow(2, attempt);
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  if (!ms || ms <= 0) return p;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () =>
        reject(
          new LiteLLMAgentError(`Request timed out after ${ms}ms`, {
            code: "timeout",
            retryable: true,
          }),
        ),
      ms,
    );
  });
  try {
    return (await Promise.race([p, timeout])) as T;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

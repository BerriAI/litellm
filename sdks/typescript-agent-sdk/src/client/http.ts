/**
 * Thin fetch wrapper for the LiteLLM proxy.
 *
 * Responsibilities:
 *  - Resolve apiKey + baseUrl from explicit args or env
 *  - Inject `Authorization: Bearer <apiKey>`
 *  - Retry retryable errors (5xx, 429) with exponential backoff
 *  - Normalize errors to `LiteLLMAgentError`
 */

import { ClientOptions, LiteLLMAgentError } from "../types.js";

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
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
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
  return (await res.json()) as T;
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

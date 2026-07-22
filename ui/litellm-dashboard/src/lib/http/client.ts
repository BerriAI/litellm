/**
 * The single HTTP client for the dashboard. This is the only file allowed to
 * call fetch() directly (enforced by the no-restricted-syntax lint rule and its
 * src/lib/http/** override in eslint.config.mjs).
 *
 * It is framework-agnostic on purpose (no React, no module-level singletons from
 * the component tree) so the same client can run in client components today and
 * in server components later. Everything environment-specific (base URL, auth
 * header name, the logout side effect) is injected through createApiClient.
 */

export type HttpMethod = "GET" | "POST" | "PUT" | "DELETE" | "PATCH";

export type QueryValue = string | number | boolean | null | undefined;

export type QueryParams = Record<string, QueryValue | QueryValue[]>;

export interface RequestOptions {
  /** Bearer token. When present, the auth header is set automatically. */
  accessToken?: string | null;
  /** Serialized to JSON unless `rawBody` is provided. */
  body?: unknown;
  /** Sent verbatim (FormData, Blob, pre-stringified text); disables JSON handling. */
  rawBody?: BodyInit;
  query?: QueryParams;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/**
 * Best-effort extraction of a human-readable message from a proxy error body.
 * Lives here because error parsing is the client's job; networking.tsx re-exports
 * it so existing `@/components/networking` import paths keep working.
 */
export const deriveErrorMessage = (errorData: any): string => {
  const detail = errorData?.detail;
  const detailStr = Array.isArray(detail)
    ? detail.map((d: any) => d?.msg || JSON.stringify(d)).join("; ")
    : typeof detail === "string"
      ? detail
      : detail && typeof detail === "object"
        ? (typeof detail.error === "string" ? detail.error : detail.error?.message) || detail.message
        : undefined;
  return (
    (errorData?.error &&
      (errorData.error.message || (typeof errorData.error === "string" ? errorData.error : undefined))) ||
    errorData?.message ||
    detailStr ||
    JSON.stringify(errorData)
  );
};

export interface ApiClientConfig {
  /** Resolves the API origin at call time (it can change at runtime). */
  getBaseUrl: () => string;
  /** Resolves the auth header name at call time. Defaults to "Authorization". */
  getAuthHeaderName?: () => string;
  /** Invoked with the derived message right before a non-2xx response throws. Fire-and-forget. */
  onError?: (message: string) => void | Promise<void>;
  /** Injectable fetch implementation; defaults to the global. */
  fetchImpl?: typeof fetch;
}

export interface ApiClient {
  request<T = any>(method: HttpMethod, path: string, options?: RequestOptions): Promise<T>;
  get<T = any>(path: string, options?: RequestOptions): Promise<T>;
  post<T = any>(path: string, options?: RequestOptions): Promise<T>;
  put<T = any>(path: string, options?: RequestOptions): Promise<T>;
  delete<T = any>(path: string, options?: RequestOptions): Promise<T>;
  patch<T = any>(path: string, options?: RequestOptions): Promise<T>;
}

const appendQuery = (url: string, query: QueryParams | undefined): string => {
  if (!query) return url;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) {
      value.forEach((v) => v !== undefined && v !== null && search.append(key, String(v)));
    } else {
      search.append(key, String(value));
    }
  }
  const qs = search.toString();
  if (!qs) return url;
  return url.includes("?") ? `${url}&${qs}` : `${url}?${qs}`;
};

export function createApiClient(config: ApiClientConfig): ApiClient {
  const { getBaseUrl, getAuthHeaderName, onError, fetchImpl } = config;
  const doFetch: typeof fetch = (input, init) => (fetchImpl ?? fetch)(input, init);

  async function request<T = any>(method: HttpMethod, path: string, options: RequestOptions = {}): Promise<T> {
    const { accessToken, body, rawBody, query, headers: extraHeaders, signal } = options;

    const url = appendQuery(`${getBaseUrl()}${path}`, query);

    const headers: Record<string, string> = {};
    if (rawBody === undefined) {
      headers["Content-Type"] = "application/json";
    }
    if (accessToken) {
      const headerName = getAuthHeaderName ? getAuthHeaderName() : "Authorization";
      headers[headerName] = `Bearer ${accessToken}`;
    }
    if (extraHeaders) {
      Object.assign(headers, extraHeaders);
    }

    const init: RequestInit = { method, headers, signal };
    if (rawBody !== undefined) {
      init.body = rawBody;
    } else if (body !== undefined) {
      init.body = JSON.stringify(body);
    }

    const response = await doFetch(url, init);

    if (!response.ok) {
      const raw = await response.text();
      let errorBody: unknown = raw;
      let message: string;
      try {
        errorBody = JSON.parse(raw);
        message = deriveErrorMessage(errorBody);
      } catch {
        message = raw || `HTTP ${response.status}`;
      }
      onError?.(message);
      throw new ApiError(message, response.status, errorBody);
    }

    const text = await response.text();
    return (text ? JSON.parse(text) : undefined) as T;
  }

  return {
    request,
    get: (path, options) => request("GET", path, options),
    post: (path, options) => request("POST", path, options),
    put: (path, options) => request("PUT", path, options),
    delete: (path, options) => request("DELETE", path, options),
    patch: (path, options) => request("PATCH", path, options),
  };
}

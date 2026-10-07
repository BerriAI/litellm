import createFetchClient, { type Middleware } from "openapi-fetch";
import createQueryClient from "openapi-react-query";
import type { paths } from "./schema";
import { ApiError, deriveErrorMessage, retryAfterMs } from "./client";
import { getAuthHeaderName, getAuthToken, getRequestBaseUrl, reportError } from "./runtime";
import { resolveRequestUrl } from "./resolveApiBase";

const BaseAwareRequest = function (url: string, init?: RequestInit): Request {
  const target = resolveRequestUrl(url, {
    registeredBase: getRequestBaseUrl(),
    pageOrigin: globalThis.location?.origin,
  });
  return new globalThis.Request(target, init);
} as unknown as typeof Request;

const isJsonMediaType = (contentType: string): boolean => /[/+]json\b/i.test(contentType);

const carriesJson = async (response: Response): Promise<boolean> => {
  const contentType = response.headers.get("content-type");
  if (contentType !== null && isJsonMediaType(contentType)) return true;
  const text = await response.clone().text();
  if (!text) return true;
  try {
    JSON.parse(text);
    return true;
  } catch {
    return false;
  }
};

const middleware: Middleware = {
  onRequest({ request }) {
    if (!request.headers.has("Accept")) {
      request.headers.set("Accept", "application/json");
    }
    const token = getAuthToken();
    if (token && !request.headers.has(getAuthHeaderName())) {
      request.headers.set(getAuthHeaderName(), `Bearer ${token}`);
    }
  },
  async onResponse({ request, response }) {
    if (response.ok) {
      if (await carriesJson(response)) return response;
      const contentType = response.headers.get("content-type") ?? "an unknown content type";
      const message = `Expected JSON from ${new URL(request.url).pathname} but the server returned ${contentType}`;
      reportError(message);
      throw new ApiError(message, response.status, await response.clone().text());
    }
    const raw = await response.clone().text();
    let body: unknown = raw;
    let message: string;
    try {
      body = JSON.parse(raw);
      message = deriveErrorMessage(body);
    } catch {
      message = raw || `HTTP ${response.status}`;
    }
    reportError(message);
    throw new ApiError(message, response.status, body, retryAfterMs(response.headers));
  },
};

/**
 * The typed, schema-bound HTTP client. Use it inside TanStack Query hooks
 * (`fetchClient.GET("/path", { params })`) and for imperative calls; path
 * params, query params, and request bodies are inferred from schema.d.ts.
 *
 * The base URL is injected, not fixed at import: every request is built against
 * whatever registerBaseUrlGetter supplies at call time (a split-origin proxy or
 * worker URL), falling back to the current origin. The middleware sends
 * `Accept: application/json` (the next dev rewrite routes on it), injects the
 * auth header, and maps non-2xx responses and non-JSON success bodies to
 * ApiError so query functions can just read `.data`.
 */
export const fetchClient = createFetchClient<paths>({
  Request: BaseAwareRequest,
  fetch: (request) => globalThis.fetch(request),
});
fetchClient.use(middleware);

/**
 * TanStack Query bound to the typed client. Callers write
 * `$api.useQuery("get", "/path", init, options)`; the query key is derived from
 * method + path + init (no hand-maintained key), the request signal is
 * forwarded for cancellation, and the response type comes from schema.d.ts.
 */
export const $api = createQueryClient(fetchClient);

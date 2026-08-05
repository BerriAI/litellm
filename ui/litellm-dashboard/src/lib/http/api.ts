import createFetchClient, { type Middleware } from "openapi-fetch";
import createQueryClient from "openapi-react-query";
import type { paths } from "./schema";
import { ApiError, deriveErrorMessage } from "./client";
import { getAuthHeaderName, getAuthToken, getRequestBaseUrl, reportError } from "./runtime";
import { resolveRequestUrl } from "./resolveApiBase";

const BaseAwareRequest = function (url: string, init?: RequestInit): Request {
  const target = resolveRequestUrl(url, {
    registeredBase: getRequestBaseUrl(),
    pageOrigin: globalThis.location?.origin,
  });
  return new globalThis.Request(target, init);
} as unknown as typeof Request;

const middleware: Middleware = {
  onRequest({ request }) {
    const token = getAuthToken();
    if (token) {
      request.headers.set(getAuthHeaderName(), `Bearer ${token}`);
    }
  },
  async onResponse({ response }) {
    if (response.ok) return response;
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
    throw new ApiError(message, response.status, body);
  },
};

/**
 * The typed, schema-bound HTTP client. Use it inside TanStack Query hooks
 * (`fetchClient.GET("/path", { params })`) and for imperative calls; path
 * params, query params, and request bodies are inferred from schema.d.ts.
 *
 * The base URL is injected, not fixed at import: every request is built against
 * whatever registerBaseUrlGetter supplies at call time (a split-origin proxy or
 * worker URL), falling back to the current origin. The middleware injects the
 * auth header and maps non-2xx responses to ApiError so query functions can just
 * read `.data`.
 */
export const fetchClient = createFetchClient<paths>({ Request: BaseAwareRequest });
fetchClient.use(middleware);

/**
 * TanStack Query bound to the typed client. Callers write
 * `$api.useQuery("get", "/path", init, options)`; the query key is derived from
 * method + path + init (no hand-maintained key), the request signal is
 * forwarded for cancellation, and the response type comes from schema.d.ts.
 */
export const $api = createQueryClient(fetchClient);

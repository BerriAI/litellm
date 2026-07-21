import createFetchClient, { type Middleware } from "openapi-fetch";
import createQueryClient from "openapi-react-query";
import type { paths } from "./schema";
import { ApiError, deriveErrorMessage } from "./client";
import { getAuthHeaderName, getAuthToken, getRequestBaseUrl, reportError } from "./runtime";

const rebaseUrl = (requestUrl: string, base: string): string => {
  const { pathname, search } = new URL(requestUrl);
  return `${base.replace(/\/+$/, "")}${pathname}${search}`;
};

const rebaseRequest = async (request: Request, url: string): Promise<Request> => {
  const init: RequestInit = {
    method: request.method,
    headers: request.headers,
    body: request.body ? await request.arrayBuffer() : undefined,
    mode: request.mode,
    credentials: request.credentials,
    cache: request.cache,
    redirect: request.redirect,
    referrer: request.referrer,
    integrity: request.integrity,
    keepalive: request.keepalive,
    signal: request.signal,
  };
  return new Request(url, init);
};

const middleware: Middleware = {
  async onRequest({ request }) {
    const base = getRequestBaseUrl();
    const next = base ? await rebaseRequest(request, rebaseUrl(request.url, base)) : request;
    const token = getAuthToken();
    if (token) {
      next.headers.set(getAuthHeaderName(), `Bearer ${token}`);
    }
    return next;
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
 * The creation-time base is the current origin so request URLs are absolute; the
 * middleware rebases each call onto the runtime base when one is registered (a
 * split-origin proxy or worker URL), injects the auth header, and maps non-2xx
 * responses to ApiError so query functions can just read `.data`.
 */
export const fetchClient = createFetchClient<paths>({ baseUrl: globalThis.location?.origin ?? "" });
fetchClient.use(middleware);

/**
 * TanStack Query bound to the typed client. Callers write
 * `$api.useQuery("get", "/path", init, options)`; the query key is derived from
 * method + path + init (no hand-maintained key), the request signal is
 * forwarded for cancellation, and the response type comes from schema.d.ts.
 */
export const $api = createQueryClient(fetchClient);

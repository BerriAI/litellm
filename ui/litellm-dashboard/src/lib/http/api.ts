import createFetchClient, { type Middleware } from "openapi-fetch";
import type { paths } from "./schema";
import { ApiError, deriveErrorMessage } from "./client";
import { getAuthHeaderName, getAuthToken, getRequestBaseUrl, reportError } from "./runtime";

const rebaseUrl = (requestUrl: string, base: string): string => {
  const { pathname, search } = new URL(requestUrl);
  return `${base.replace(/\/+$/, "")}${pathname}${search}`;
};

const middleware: Middleware = {
  onRequest({ request }) {
    const base = getRequestBaseUrl();
    const next = new Request(base ? rebaseUrl(request.url, base) : request.url, request);
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

/**
 * Typed API surface generated from schema.d.ts via openapi-fetch + openapi-react-query.
 * Coexists with the legacy hand-rolled client/networking during the migration: new
 * data access should use `$api`; call sites move over endpoint by endpoint.
 */
import createFetchClient, { type Middleware } from "openapi-fetch";
import createQueryClient from "openapi-react-query";

import type { paths } from "@/lib/http/schema";
import { ApiError, deriveErrorMessage } from "@/lib/http/client";
import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { getCookie } from "@/utils/cookieUtils";

const authAndErrors: Middleware = {
  onRequest({ request }) {
    const token = typeof document !== "undefined" ? getCookie("token") : null;
    if (token) {
      request.headers.set(getGlobalLitellmHeaderName(), `Bearer ${token}`);
    }
    return request;
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
    throw new ApiError(message, response.status, body);
  },
};

const fetchClient = createFetchClient<paths>({
  baseUrl: getProxyBaseUrl(),
  fetch: (request) => globalThis.fetch(request),
});
fetchClient.use(authAndErrors);

export const $api = createQueryClient(fetchClient);

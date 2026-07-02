import createFetchClient from "openapi-fetch";
import createClient from "openapi-react-query";
import type { paths } from "./schema";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";

// Placeholder origin so openapi-fetch always builds a parseable absolute URL (relative
// Request construction throws under Node/SSR). The onRequest middleware replaces it with
// the real, call-time base from getProxyBaseUrl on every request, so it never hits the network.
const PLACEHOLDER_ORIGIN = "http://litellm.local";

const fetchClient = createFetchClient<paths>({ baseUrl: PLACEHOLDER_ORIGIN });

fetchClient.use({
  onRequest({ request }) {
    const tail = new URL(request.url);
    return new Request(getProxyBaseUrl() + tail.pathname + tail.search, request);
  },
});

export const $api = createClient(fetchClient);

export const authHeader = (accessToken: string): Record<string, string> => ({
  [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
});

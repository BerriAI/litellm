import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { withRequiredHeaders } from "@/components/llm_calls/request_headers";
import { createApiClient } from "@/lib/http/client";
import type { SystemOneRequest, SystemOneResponse } from "../components/systemOneUI/system_one_types";

export async function makeSystemOneRequest(
  payload: SystemOneRequest,
  accessToken: string,
  customBaseUrl?: string,
  signal?: AbortSignal,
): Promise<{ response: SystemOneResponse; latencyMs: number }> {
  const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();
  const normalizedBaseUrl = proxyBaseUrl.endsWith("/") ? proxyBaseUrl.slice(0, -1) : proxyBaseUrl;
  const startedAt = performance.now();
  const headers = withRequiredHeaders(
    {},
    {
      "Content-Type": "application/json",
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
    },
  );
  const client = createApiClient({ getBaseUrl: () => normalizedBaseUrl });
  const response = await client.post<SystemOneResponse>("/typesafe/v1/systemone", {
    body: payload,
    headers,
    signal,
  });

  return { response, latencyMs: performance.now() - startedAt };
}

import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { withRequiredHeaders } from "@/components/llm_calls/request_headers";
import { createApiClient } from "@/lib/http/client";
import {
  systemOneResponseSchema,
  type SystemOneRequest,
  type SystemOneResponse,
} from "../components/systemOneUI/lib/schemas";

export interface SystemOneResult {
  response: SystemOneResponse;
  latencyMs: number;
}

export async function makeSystemOneRequest(
  payload: SystemOneRequest,
  accessToken: string,
  customBaseUrl?: string,
  signal?: AbortSignal,
): Promise<SystemOneResult> {
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
  const body = await client.post<unknown>("/typesafe/v1/systemone", { body: payload, headers, signal });
  const parsed = systemOneResponseSchema.safeParse(body);
  if (!parsed.success) {
    throw new Error("System One response has an invalid shape.");
  }
  return { response: parsed.data, latencyMs: performance.now() - startedAt };
}

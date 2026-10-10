import { z } from "zod";
import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { withRequiredHeaders } from "@/components/llm_calls/request_headers";
import { createApiClient } from "@/lib/http/client";
import { isDecisionMode } from "@/lib/decisionModels";
import {
  systemOneResponseSchema,
  type DecisionEndpoint,
  type PlaygroundRequest,
  type SystemOneResponse,
} from "../components/systemOneUI/lib/schemas";

export interface SystemOneResult {
  response: SystemOneResponse;
  latencyMs: number;
}

const modelGroupInfoSchema = z.object({
  data: z.array(z.object({ model_group: z.string(), mode: z.string().nullish() })),
});

function proxyClient(customBaseUrl?: string) {
  const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();
  const normalizedBaseUrl = proxyBaseUrl.endsWith("/") ? proxyBaseUrl.slice(0, -1) : proxyBaseUrl;
  return createApiClient({ getBaseUrl: () => normalizedBaseUrl });
}

function authHeaders(accessToken: string): Record<string, string> {
  return withRequiredHeaders(
    {},
    {
      "Content-Type": "application/json",
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
    },
  );
}

export async function fetchDecisionModels(
  accessToken: string,
  customBaseUrl?: string,
  signal?: AbortSignal,
): Promise<string[]> {
  const body = await proxyClient(customBaseUrl).get<unknown>("/model_group/info", {
    headers: authHeaders(accessToken),
    signal,
  });
  return modelGroupInfoSchema
    .parse(body)
    .data.filter((group) => isDecisionMode(group.mode))
    .map((group) => group.model_group)
    .sort((a, b) => a.localeCompare(b));
}

export async function makeSystemOneRequest(
  payload: PlaygroundRequest,
  accessToken: string,
  customBaseUrl?: string,
  { signal, endpoint = "/typesafe/v1/systemone" }: { signal?: AbortSignal; endpoint?: DecisionEndpoint } = {},
): Promise<SystemOneResult> {
  const startedAt = performance.now();
  const body = await proxyClient(customBaseUrl).post<unknown>(endpoint, {
    body: payload,
    headers: authHeaders(accessToken),
    signal,
  });
  const parsed = systemOneResponseSchema.safeParse(body);
  if (!parsed.success) {
    throw new Error("System One response has an invalid shape.");
  }
  return { response: parsed.data, latencyMs: performance.now() - startedAt };
}

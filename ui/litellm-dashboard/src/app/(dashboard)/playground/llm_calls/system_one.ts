import { z } from "zod";
import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { withRequiredHeaders } from "@/components/llm_calls/request_headers";
import { createApiClient } from "@/lib/http/client";
import { isDecisionMode } from "@/lib/decisionModels";
import {
  systemOneResponseSchema,
  type PlaygroundRequest,
  type SystemOneResponse,
} from "../components/systemOneUI/lib/schemas";
import {
  openAIDecisionsResponseSchema,
  type OpenAIDecisionsRequest,
  type OpenAIDecisionsResponse,
} from "../components/systemOneUI/lib/openAIDecisions";

export interface DecisionResult<T> {
  response: T;
  latencyMs: number;
}

export type SystemOneResult = DecisionResult<SystemOneResponse>;
export type OpenAIDecisionsResult = DecisionResult<OpenAIDecisionsResponse>;
export type PlaygroundDecisionRequest =
  | { endpoint: "/v1/systemone"; payload: PlaygroundRequest }
  | { endpoint: "/v1/decisions"; payload: OpenAIDecisionsRequest };

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

async function sendDecisionRequest<T>(
  request: { endpoint: PlaygroundDecisionRequest["endpoint"]; payload: object; schema: z.ZodType<T> },
  accessToken: string,
  customBaseUrl?: string,
  signal?: AbortSignal,
): Promise<DecisionResult<T>> {
  const startedAt = performance.now();
  const body = await proxyClient(customBaseUrl).post<unknown>(request.endpoint, {
    body: request.payload,
    headers: authHeaders(accessToken),
    signal,
  });
  const parsed = request.schema.safeParse(body);
  if (!parsed.success) {
    throw new Error(`${request.endpoint} response has an invalid shape.`);
  }
  return { response: parsed.data, latencyMs: performance.now() - startedAt };
}

export function makeSystemOneRequest(
  payload: PlaygroundRequest,
  accessToken: string,
  customBaseUrl?: string,
  { signal }: { signal?: AbortSignal } = {},
): Promise<SystemOneResult> {
  return sendDecisionRequest(
    { endpoint: "/v1/systemone", payload, schema: systemOneResponseSchema },
    accessToken,
    customBaseUrl,
    signal,
  );
}

export function makePlaygroundDecisionRequest(
  request: PlaygroundDecisionRequest,
  accessToken: string,
  customBaseUrl?: string,
  { signal }: { signal?: AbortSignal } = {},
): Promise<SystemOneResult | OpenAIDecisionsResult> {
  if (request.endpoint === "/v1/systemone") {
    return makeSystemOneRequest(request.payload, accessToken, customBaseUrl, { signal });
  }
  return sendDecisionRequest({ ...request, schema: openAIDecisionsResponseSchema }, accessToken, customBaseUrl, signal);
}

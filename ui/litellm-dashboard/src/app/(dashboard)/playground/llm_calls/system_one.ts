import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { withRequiredHeaders } from "@/components/llm_calls/request_headers";
import { createApiClient } from "@/lib/http/client";
import type { SystemOneAnswer, SystemOneRequest, SystemOneResponse } from "../components/systemOneUI/system_one_types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isProbability(value: unknown): value is number {
  if (!isFiniteNumber(value)) {
    return false;
  }
  return value >= 0 && value <= 1;
}

function isNonNegativeNumber(value: unknown): value is number {
  if (!isFiniteNumber(value)) {
    return false;
  }
  return value >= 0;
}

function hasProbabilityValues(value: unknown): value is Record<string, number> {
  if (!isRecord(value)) {
    return false;
  }
  return Object.values(value).every(isProbability);
}

function hasStringValues(value: unknown): value is Record<string, string> {
  if (!isRecord(value)) {
    return false;
  }
  return Object.values(value).every((entry) => typeof entry === "string");
}

function hasOptionalProbability(record: Record<string, unknown>, key: string): boolean {
  if (!(key in record)) {
    return true;
  }
  return isProbability(record[key]);
}

function isSystemOneAnswer(value: unknown): value is SystemOneAnswer {
  if (!isRecord(value)) {
    return false;
  }
  if (value.type === "noul") {
    return isProbability(value.noul);
  }
  if (value.type === "choice") {
    if (typeof value.choice !== "string") {
      return false;
    }
    if (!hasOptionalProbability(value, "confidence")) {
      return false;
    }
    return hasProbabilityValues(value.probabilities);
  }
  if (value.type === "score") {
    if (!isFiniteNumber(value.score)) {
      return false;
    }
    if (!hasOptionalProbability(value, "confidence")) {
      return false;
    }
    if (!hasProbabilityValues(value.probabilities)) {
      return false;
    }
    if ("legend" in value && !hasStringValues(value.legend)) {
      return false;
    }
    return true;
  }
  return false;
}

function isSystemOneUsage(value: unknown): value is NonNullable<SystemOneResponse["usage"]> {
  if (!isRecord(value)) {
    return false;
  }
  if (!isNonNegativeNumber(value.input_tokens)) {
    return false;
  }
  return isNonNegativeNumber(value.output_tokens);
}

function isOptionalModel(value: unknown): value is string | null | undefined {
  if (value === undefined || value === null) {
    return true;
  }
  return typeof value === "string" && value.trim().length > 0;
}

function isSystemOneResponse(value: unknown): value is SystemOneResponse {
  if (!isRecord(value)) {
    return false;
  }
  if (!isOptionalModel(value.model)) {
    return false;
  }
  if (!isRecord(value.answers)) {
    return false;
  }
  if (!Object.values(value.answers).every(isSystemOneAnswer)) {
    return false;
  }
  if ("usage" in value && !isSystemOneUsage(value.usage)) {
    return false;
  }
  return true;
}

export type SystemOneRequestResult =
  | { type: "success"; response: SystemOneResponse; latencyMs: number }
  | { type: "error"; message: string };

export async function makeSystemOneRequest(
  payload: SystemOneRequest,
  accessToken: string,
  customBaseUrl?: string,
  signal?: AbortSignal,
): Promise<SystemOneRequestResult> {
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
  const response = await client.post<unknown>("/typesafe/v1/systemone", {
    body: payload,
    headers,
    signal,
  });
  if (!isSystemOneResponse(response)) {
    return { type: "error", message: "System One response has an invalid shape." };
  }

  return { type: "success", response, latencyMs: performance.now() - startedAt };
}

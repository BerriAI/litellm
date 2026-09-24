import OpenAI from "openai";
import type { ChatCompletionChunk, ChatCompletionCreateParamsStreaming } from "openai/resources/chat/completions";
import packageJson from "../package.json";
import { parseModelGroups, type ConfigurationValues, type ModelGroupInfo } from "./models";

export interface GatewayConfig {
  readonly baseUrl: string;
  readonly apiKey: string;
}

export type GatewayConfigResult =
  | { readonly kind: "ok"; readonly config: GatewayConfig }
  | { readonly kind: "unconfigured" }
  | { readonly kind: "missing_fields"; readonly fields: readonly string[] }
  | { readonly kind: "invalid_url"; readonly baseUrl: string };

export type ModelGroupsResult =
  | { readonly kind: "ok"; readonly groups: readonly ModelGroupInfo[] }
  | { readonly kind: "http_error"; readonly status: number; readonly body: string }
  | { readonly kind: "invalid_response"; readonly reason: string };

export interface GatewayClient {
  listModelGroups(config: GatewayConfig, signal: AbortSignal): Promise<ModelGroupsResult>;
  streamChatCompletion(
    config: GatewayConfig,
    params: ChatCompletionCreateParamsStreaming,
    signal: AbortSignal,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
}

export const USER_AGENT = `litellm-vscode/${packageJson.version}`;
export const ERROR_SUMMARY_LIMIT = 200;

const GATEWAY_PROTOCOLS: ReadonlySet<string> = new Set(["http:", "https:"]);

const parsesAsHttpUrl = (value: string): boolean => {
  try {
    return GATEWAY_PROTOCOLS.has(new URL(value).protocol);
  } catch {
    return false;
  }
};

export const gatewayRoot = (baseUrl: string): string | undefined => {
  const root = baseUrl.trim().replace(/\/+$/, "").replace(/\/v1$/, "");
  return parsesAsHttpUrl(root) ? root : undefined;
};

const nonEmptyString = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() !== "" ? value.trim() : undefined;

export const gatewayConfigFrom = (configuration: ConfigurationValues | undefined): GatewayConfigResult => {
  if (configuration === undefined) {
    return { kind: "unconfigured" };
  }
  const baseUrl = nonEmptyString(configuration.baseUrl);
  const apiKey = nonEmptyString(configuration.apiKey);
  if (baseUrl === undefined || apiKey === undefined) {
    const fields = [...(baseUrl === undefined ? ["Gateway URL"] : []), ...(apiKey === undefined ? ["API key"] : [])];
    return { kind: "missing_fields", fields };
  }
  const root = gatewayRoot(baseUrl);
  return root === undefined ? { kind: "invalid_url", baseUrl } : { kind: "ok", config: { baseUrl: root, apiKey } };
};

export const modelGroupInfoUrl = (root: string): string => `${root}/model_group/info`;

export const openAiBaseUrl = (root: string): string => `${root}/v1`;

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;

const errorMessageIn = (body: string): string | undefined => {
  try {
    const parsed: unknown = JSON.parse(body);
    if (!isRecord(parsed)) {
      return undefined;
    }
    if (isRecord(parsed.error) && typeof parsed.error.message === "string") {
      return parsed.error.message;
    }
    return typeof parsed.detail === "string" ? parsed.detail : undefined;
  } catch {
    return undefined;
  }
};

export const summarizeErrorBody = (body: string): string => {
  const message = (errorMessageIn(body) ?? body).replace(/\s+/g, " ").trim();
  return message.length > ERROR_SUMMARY_LIMIT ? `${message.slice(0, ERROR_SUMMARY_LIMIT)}...` : message;
};

export const createGatewayClient = (fetchImpl: typeof fetch = fetch): GatewayClient => ({
  async listModelGroups(config, signal) {
    const response = await fetchImpl(modelGroupInfoUrl(config.baseUrl), {
      headers: { Authorization: `Bearer ${config.apiKey}`, "User-Agent": USER_AGENT },
      signal,
    });
    if (!response.ok) {
      return { kind: "http_error", status: response.status, body: await response.text() };
    }
    const parsed = parseModelGroups(await response.json());
    return parsed.kind === "ok" ? parsed : { kind: "invalid_response", reason: parsed.reason };
  },
  streamChatCompletion(config, params, signal) {
    const client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: openAiBaseUrl(config.baseUrl),
      defaultHeaders: { "User-Agent": USER_AGENT },
      fetch: fetchImpl,
      maxRetries: 0,
    });
    return client.chat.completions.create(params, { signal });
  },
});

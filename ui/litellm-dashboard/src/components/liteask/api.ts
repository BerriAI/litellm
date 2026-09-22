import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { createApiClient } from "@/lib/http/client";
import type { components } from "@/lib/http/schema";

export type JsonValue = components["schemas"]["JsonValue"];
export type LiteAskConfig = components["schemas"]["LiteAskConfig"];
export type LiteAskMessage = components["schemas"]["LiteAskMessage"];
export type LiteAskProposal = components["schemas"]["LiteAskProposal"];
export type LiteAskResponse = components["schemas"]["LiteAskResponse"];

const client = createApiClient({
  getBaseUrl: () => getProxyBaseUrl() ?? "",
  getAuthHeaderName: () => getGlobalLitellmHeaderName(),
});
const path = "/management/v1/liteask";

export const liteAskApi = {
  config: (accessToken: string, signal: AbortSignal) =>
    client.get<LiteAskConfig>(`${path}/config`, { accessToken, signal }),
  chat: (accessToken: string, body: components["schemas"]["LiteAskChatRequest"], signal: AbortSignal) =>
    client.post<LiteAskResponse>(`${path}/chat`, { accessToken, body, signal }),
  approve: (accessToken: string, body: components["schemas"]["LiteAskApprovalRequest"], signal: AbortSignal) =>
    client.post<LiteAskResponse>(`${path}/approve`, { accessToken, body, signal }),
};

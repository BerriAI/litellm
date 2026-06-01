/**
 * Client for the cloud-agents (Cursor SDK) API.
 *
 * Routes through `proxyBaseUrl` so all calls hit the LiteLLM proxy on the same
 * origin. When `NEXT_PUBLIC_USE_MOCK_AGENTS=true`, every method short-circuits
 * to `mock-agents.ts` so the dashboard can develop without Epic A merged.
 *
 * Backend namespace is `/v2/` — `/v1/agents` is reserved for the existing A2A
 * registry. When Epic A lands and exposes `/v2/agents`, `/v2/sessions`, etc.,
 * flip the env var off and the real fetches kick in. No component changes
 * required.
 */
import { getProxyBaseUrl } from "@/components/networking";
import {
  isMockEnabled,
  mockCreateAgent,
  mockCreateSession,
  mockGetAgent,
  mockGetConversation,
  mockGetRun,
  mockGetSession,
  mockListAgents,
  mockListSessions,
  mockSendFollowup,
} from "@/lib/mock-agents";
import type { CloudAgent, CloudAgentConversationMessage, CloudAgentRun, CloudAgentSession } from "@/types/cloud-agents";

function authHeaders(accessToken: string | null): HeadersInit {
  const headers: HeadersInit = { "Content-Type": "application/json" };
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }
  return headers;
}

async function jsonOrThrow<T>(res: Response, label: string): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${label} failed: ${res.status} ${text}`);
  }
  return (await res.json()) as T;
}

/* ---------- agents ---------- */

export async function listCloudAgents(accessToken: string | null): Promise<CloudAgent[]> {
  if (isMockEnabled()) return mockListAgents();
  const res = await fetch(`${getProxyBaseUrl()}/v2/agents`, { headers: authHeaders(accessToken) });
  const body = await jsonOrThrow<{ agents: CloudAgent[] }>(res, "listCloudAgents");
  return body.agents;
}

export async function getCloudAgent(accessToken: string | null, agentId: string): Promise<CloudAgent | null> {
  if (isMockEnabled()) return mockGetAgent(agentId);
  const res = await fetch(`${getProxyBaseUrl()}/v2/agents/${agentId}`, {
    headers: authHeaders(accessToken),
  });
  if (res.status === 404) return null;
  return jsonOrThrow<CloudAgent>(res, "getCloudAgent");
}

export async function createCloudAgent(
  accessToken: string | null,
  input: { name: string; model: string; system_prompt?: string },
): Promise<CloudAgent> {
  if (isMockEnabled()) return mockCreateAgent(input);
  const res = await fetch(`${getProxyBaseUrl()}/v2/agents`, {
    method: "POST",
    headers: authHeaders(accessToken),
    body: JSON.stringify(input),
  });
  return jsonOrThrow<CloudAgent>(res, "createCloudAgent");
}

/* ---------- sessions ---------- */

export async function listCloudSessions(accessToken: string | null, agentId: string): Promise<CloudAgentSession[]> {
  if (isMockEnabled()) return mockListSessions(agentId);
  const res = await fetch(`${getProxyBaseUrl()}/v2/sessions?agent_id=${encodeURIComponent(agentId)}`, {
    headers: authHeaders(accessToken),
  });
  const body = await jsonOrThrow<{ sessions: CloudAgentSession[] }>(res, "listCloudSessions");
  return body.sessions;
}

export async function getCloudSession(
  accessToken: string | null,
  sessionId: string,
): Promise<CloudAgentSession | null> {
  if (isMockEnabled()) return mockGetSession(sessionId);
  const res = await fetch(`${getProxyBaseUrl()}/v2/sessions/${sessionId}`, {
    headers: authHeaders(accessToken),
  });
  if (res.status === 404) return null;
  return jsonOrThrow<CloudAgentSession>(res, "getCloudSession");
}

export async function createCloudSession(
  accessToken: string | null,
  input: { agent_id: string; repo_url: string },
): Promise<CloudAgentSession> {
  if (isMockEnabled()) return mockCreateSession(input);
  const res = await fetch(`${getProxyBaseUrl()}/v2/sessions`, {
    method: "POST",
    headers: authHeaders(accessToken),
    body: JSON.stringify(input),
  });
  return jsonOrThrow<CloudAgentSession>(res, "createCloudSession");
}

export async function getSessionConversation(
  accessToken: string | null,
  sessionId: string,
): Promise<CloudAgentConversationMessage[]> {
  if (isMockEnabled()) return mockGetConversation(sessionId);
  const res = await fetch(`${getProxyBaseUrl()}/v2/sessions/${sessionId}/conversation`, {
    headers: authHeaders(accessToken),
  });
  const body = await jsonOrThrow<{ messages: CloudAgentConversationMessage[] }>(res, "getSessionConversation");
  return body.messages;
}

export async function sendSessionFollowup(
  accessToken: string | null,
  sessionId: string,
  content: string,
): Promise<CloudAgentConversationMessage> {
  if (isMockEnabled()) return mockSendFollowup(sessionId, content);
  const res = await fetch(`${getProxyBaseUrl()}/v2/sessions/${sessionId}/followup`, {
    method: "POST",
    headers: authHeaders(accessToken),
    body: JSON.stringify({ content }),
  });
  return jsonOrThrow<CloudAgentConversationMessage>(res, "sendSessionFollowup");
}

/* ---------- runs ---------- */

export async function getCloudRun(accessToken: string | null, runId: string): Promise<CloudAgentRun | null> {
  if (isMockEnabled()) return mockGetRun(runId);
  const res = await fetch(`${getProxyBaseUrl()}/v2/runs/${runId}`, {
    headers: authHeaders(accessToken),
  });
  if (res.status === 404) return null;
  return jsonOrThrow<CloudAgentRun>(res, "getCloudRun");
}

/**
 * Build the SSE URL for a run's event stream. The hook (`useSessionEventStream`)
 * is what actually opens the EventSource — this just centralizes the URL shape
 * so the route can change in one place.
 */
export function buildRunEventStreamUrl(sessionId: string, runId: string, sinceSeq?: number): string {
  const base = `${getProxyBaseUrl()}/v2/sessions/${sessionId}/runs/${runId}/events`;
  return sinceSeq !== undefined ? `${base}?since_seq=${sinceSeq}` : base;
}

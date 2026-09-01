import { APIRequestContext, expect } from "@playwright/test";

/** Model names served by fixtures/config.yml, both backed by the mock LLM server. */
export const CHAT_MODEL_A = "fake-openai-gpt-4";
export const CHAT_MODEL_B = "fake-anthropic-claude";

/** The deployment each of those models routes to, as spend logs and usage breakdowns name it. */
export const DEPLOYMENT_MODEL_A = "openai/fake-gpt-4";
export const DEPLOYMENT_MODEL_B = "openai/fake-claude";

/** The only completion text fixtures/mock_llm_server/server.py ever returns. */
export const MOCK_RESPONSE_TEXT = "This is a mock response.";

export const masterKey = (): string => process.env.LITELLM_MASTER_KEY || "sk-1234";

export const rootPath = (): string => process.env.SERVER_ROOT_PATH ?? "";

interface ChatOptions {
  model: string;
  prompt: string;
  apiKey?: string;
  /** Sent as `user`, which lands in the spend log's end_user column. */
  endUser?: string;
}

/** POST /v1/chat/completions and return the completion id (the Logs Request ID). */
export async function sendChatCompletion(request: APIRequestContext, opts: ChatOptions): Promise<string> {
  const res = await request.post(`${rootPath()}/v1/chat/completions`, {
    headers: {
      Authorization: `Bearer ${opts.apiKey ?? masterKey()}`,
      "Content-Type": "application/json",
    },
    data: {
      model: opts.model,
      messages: [{ role: "user", content: opts.prompt }],
      ...(opts.endUser ? { user: opts.endUser } : {}),
    },
  });
  expect(res.ok(), `chat completion for ${opts.model} failed (${res.status()}): ${await res.text()}`).toBe(true);
  const body = await res.json();
  expect(body.choices?.[0]?.message?.content).toContain(MOCK_RESPONSE_TEXT);
  return body.id as string;
}

/** `key` is the sk- value to authenticate with; `token` is its hash, which spend aggregates are keyed by. */
export async function createVirtualKey(
  request: APIRequestContext,
  data: Record<string, unknown> = {},
): Promise<{ key: string; token: string; alias?: string }> {
  const res = await request.post(`${rootPath()}/key/generate`, {
    headers: {
      Authorization: `Bearer ${masterKey()}`,
      "Content-Type": "application/json",
    },
    data,
  });
  expect(res.ok(), `key generate failed (${res.status()}): ${await res.text()}`).toBe(true);
  const body = await res.json();
  return {
    key: body.key as string,
    token: (body.token ?? body.token_id) as string,
    alias: body.key_alias as string | undefined,
  };
}

/** Spend logs are flushed on a timer, so an assertion straight after a completion races the writer. */
export async function waitForSpendLog(
  request: APIRequestContext,
  requestId: string,
  timeoutMs = 60_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastStatus = 0;
  while (Date.now() < deadline) {
    const res = await request.get(`${rootPath()}/spend/logs?request_id=${encodeURIComponent(requestId)}`, {
      headers: { Authorization: `Bearer ${masterKey()}` },
    });
    lastStatus = res.status();
    if (res.ok()) {
      const body = await res.json();
      const rows = Array.isArray(body) ? body : (body?.data ?? []);
      if (rows.length > 0) {
        return;
      }
    }
    await new Promise((r) => setTimeout(r, 2_000));
  }
  throw new Error(`spend log for request ${requestId} never appeared (last /spend/logs status ${lastStatus})`);
}

export async function waitForSpendLogByPrompt(
  request: APIRequestContext,
  prompt: string,
  timeoutMs = 60_000,
): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  let lastStatus = 0;
  while (Date.now() < deadline) {
    const res = await request.get(`${rootPath()}/spend/logs`, {
      headers: { Authorization: `Bearer ${masterKey()}` },
    });
    lastStatus = res.status();
    if (res.ok()) {
      const rows: { request_id?: string; messages?: unknown; proxy_server_request?: unknown }[] = await res.json();
      const row = (Array.isArray(rows) ? rows : []).find(
        (candidate) =>
          JSON.stringify(candidate.messages ?? "").includes(prompt) ||
          JSON.stringify(candidate.proxy_server_request ?? "").includes(prompt),
      );
      if (row?.request_id) {
        return row.request_id;
      }
    }
    await new Promise((r) => setTimeout(r, 2_000));
  }
  throw new Error(`no spend log row carrying prompt ${prompt} appeared (last /spend/logs status ${lastStatus})`);
}

const isoDay = (d: Date): string => d.toISOString().slice(0, 10);

interface DailyActivityKey {
  metrics?: { api_requests?: number };
}

interface DailyActivityPage {
  results?: { breakdown?: { api_keys?: Record<string, DailyActivityKey> } }[];
  metadata?: { total_pages?: number };
}

const requestsOnPage = (body: DailyActivityPage, keyToken: string): number =>
  (body.results ?? []).reduce((sum, day) => sum + (day.breakdown?.api_keys?.[keyToken]?.metrics?.api_requests ?? 0), 0);

/**
 * The route paginates its per-key breakdown. Reading only the first page finds a key while the
 * database is small and stops finding it once a run has generated more keys than one page holds,
 * which reads as "the rollup is not running" when the rollup is fine.
 */
async function keyRequestsInDailyActivity(
  request: APIRequestContext,
  query: string,
  keyToken: string,
  page = 1,
  seen = 0,
): Promise<number> {
  const res = await request.get(`${rootPath()}/user/daily/activity?${query}&page=${page}`, {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  if (!res.ok()) {
    return seen;
  }
  const body = (await res.json()) as DailyActivityPage;
  const total = seen + requestsOnPage(body, keyToken);
  return page >= (body.metadata?.total_pages ?? 1)
    ? total
    : keyRequestsInDailyActivity(request, query, keyToken, page + 1, total);
}

/**
 * The Usage page reads /user/daily/activity, a rollup written by a background job, and fetches it once
 * on mount. Navigating before the rollup lands leaves a stale render that never refreshes.
 *
 * The rollup lands request by request, so waiting only for the key to appear leaves a caller that
 * sent several requests reading a partial count. Pass `minRequests` to wait for all of them.
 */
export async function waitForKeyInDailyActivity(
  request: APIRequestContext,
  keyToken: string,
  minRequests = 1,
  timeoutMs = 120_000,
): Promise<void> {
  const now = new Date();
  const start = new Date(now);
  start.setDate(start.getDate() - 7);
  const query = `start_date=${isoDay(start)}&end_date=${isoDay(now)}`;

  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const seen = await keyRequestsInDailyActivity(request, query, keyToken);
    if (seen >= minRequests) {
      return;
    }
    if (Date.now() >= deadline) {
      throw new Error(
        `key ${keyToken} reached ${seen} of ${minRequests} requests in /user/daily/activity across every page; ` +
          "the daily spend rollup may not be running",
      );
    }
    await new Promise((r) => setTimeout(r, 3_000));
  }
}

import { APIRequestContext, expect } from "@playwright/test";

/** Model names served by fixtures/config.yml, both backed by the mock LLM server. */
export const CHAT_MODEL_A = "fake-openai-gpt-4";
export const CHAT_MODEL_B = "fake-anthropic-claude";

/** The only completion text fixtures/mock_llm_server/server.py ever returns. */
export const MOCK_RESPONSE_TEXT = "This is a mock response.";

export const masterKey = (): string => process.env.LITELLM_MASTER_KEY || "sk-1234";

const rootPath = (): string => process.env.SERVER_ROOT_PATH ?? "";

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

const isoDay = (d: Date): string => d.toISOString().slice(0, 10);

/**
 * The Usage page reads /user/daily/activity, a rollup written by a background job, and fetches it once
 * on mount. Navigating before the rollup lands leaves a stale render that never refreshes.
 */
export async function waitForKeyInDailyActivity(
  request: APIRequestContext,
  keyToken: string,
  timeoutMs = 120_000,
): Promise<void> {
  const now = new Date();
  const start = new Date(now);
  start.setDate(start.getDate() - 7);
  const query = `start_date=${isoDay(start)}&end_date=${isoDay(now)}`;

  const deadline = Date.now() + timeoutMs;
  let lastStatus = 0;
  while (Date.now() < deadline) {
    const res = await request.get(`${rootPath()}/user/daily/activity?${query}`, {
      headers: { Authorization: `Bearer ${masterKey()}` },
    });
    lastStatus = res.status();
    if (res.ok()) {
      const body = await res.json();
      const seen = (body?.results ?? []).some(
        (day: { breakdown?: { api_keys?: Record<string, unknown> } }) => keyToken in (day.breakdown?.api_keys ?? {}),
      );
      if (seen) {
        return;
      }
    }
    await new Promise((r) => setTimeout(r, 3_000));
  }
  throw new Error(
    `key ${keyToken} never appeared in /user/daily/activity (last status ${lastStatus}); ` +
      "the daily spend rollup may not be running",
  );
}

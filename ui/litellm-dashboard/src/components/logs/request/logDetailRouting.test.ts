import { describe, expect, it } from "vitest";

import type { LogEntry } from "../types";
import { isMultiRequestSession, requestLogRowLabel } from "./logDetailRouting";

const logEntry = (overrides: Partial<LogEntry> = {}): LogEntry => ({
  request_id: "r1",
  api_key: "key-1",
  team_id: "team-1",
  model: "gpt-4o",
  model_id: "model-1",
  call_type: "acompletion",
  spend: 0,
  total_tokens: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  startTime: "2026-07-07T09:50:13Z",
  endTime: "2026-07-07T09:50:14Z",
  cache_hit: "false",
  messages: [],
  response: {},
  ...overrides,
});

describe("Request Logs row labels", () => {
  it("labels a multi-request session by its session id", () => {
    const log = logEntry({ session_id: "s1", session_total_count: 3 });

    expect(isMultiRequestSession(log)).toBe(true);
    expect(requestLogRowLabel(log)).toBe("Open session s1");
  });

  it("labels a single-request session by its request id", () => {
    const log = logEntry({ session_id: "s1", session_total_count: 1 });

    expect(isMultiRequestSession(log)).toBe(false);
    expect(requestLogRowLabel(log)).toBe("Open request r1");
  });

  it("labels a request without a session by its request id", () => {
    const log = logEntry({ session_total_count: 3 });

    expect(isMultiRequestSession(log)).toBe(false);
    expect(requestLogRowLabel(log)).toBe("Open request r1");
  });

  it("defaults an undefined session count to a request label", () => {
    const log = logEntry({ request_id: "", session_id: "s1", session_total_count: undefined });

    expect(isMultiRequestSession(log)).toBe(false);
    expect(requestLogRowLabel(log)).toBe("Open request");
  });
});

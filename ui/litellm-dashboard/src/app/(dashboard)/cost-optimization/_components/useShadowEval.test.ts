import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/http/api", () => ({ $api: { useQuery: vi.fn() }, fetchClient: { POST: vi.fn() } }));

import { shadowEvalListPollMs, shadowEvalPollMs } from "./useShadowEval";

describe("shadowEvalPollMs", () => {
  it("keeps polling while the job is active or its status is not yet known", () => {
    expect(shadowEvalPollMs("running")).toBe(15_000);
    expect(shadowEvalPollMs(undefined)).toBe(15_000);
    expect(shadowEvalPollMs("completed")).toBe(false);
    expect(shadowEvalPollMs("stopped")).toBe(false);
  });
});

describe("shadowEvalListPollMs", () => {
  it("polls the list while any job is running, so finished jobs migrate to previous", () => {
    expect(shadowEvalListPollMs([{ status: "running" } as never, { status: "stopped" } as never])).toBe(15_000);
    expect(shadowEvalListPollMs([{ status: "completed" } as never])).toBe(false);
    expect(shadowEvalListPollMs([])).toBe(false);
    expect(shadowEvalListPollMs(undefined)).toBe(false);
  });
});

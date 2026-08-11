import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/http/api", () => ({ $api: { useQuery: vi.fn() }, fetchClient: { POST: vi.fn() } }));
vi.mock("@/components/molecules/notifications_manager", () => ({ default: { fromBackend: vi.fn() } }));

import { shadowEvalPollMs } from "./useShadowEval";

describe("shadowEvalPollMs", () => {
  it("keeps polling while the job is active or its status is not yet known", () => {
    expect(shadowEvalPollMs("pending")).toBe(15_000);
    expect(shadowEvalPollMs("running")).toBe(15_000);
    expect(shadowEvalPollMs(undefined)).toBe(15_000);
    expect(shadowEvalPollMs("completed")).toBe(false);
  });
});

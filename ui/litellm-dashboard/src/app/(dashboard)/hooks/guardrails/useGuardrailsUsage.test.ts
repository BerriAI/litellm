import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useGuardrailsUsageDetail, useGuardrailsUsageOverview } from "./useGuardrailsUsage";

const useQueryMock = vi.fn();
vi.mock("@/lib/http/api", () => ({
  $api: { useQuery: (...args: unknown[]) => useQueryMock(...args) },
}));

const lastCall = () => {
  const calls = useQueryMock.mock.calls;
  return calls[calls.length - 1] as [string, string, unknown, { enabled: boolean }];
};

describe("useGuardrailsUsageOverview", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useQueryMock.mockReturnValue({ data: undefined });
  });

  it("queries GET /guardrails/usage/overview with the window as query params", () => {
    renderHook(() => useGuardrailsUsageOverview({ accessToken: "sk", startDate: "2026-09-01", endDate: "2026-09-04" }));

    expect(lastCall().slice(0, 3)).toEqual([
      "get",
      "/guardrails/usage/overview",
      { params: { query: { start_date: "2026-09-01", end_date: "2026-09-04" } } },
    ]);
    expect(lastCall()[3].enabled).toBe(true);
  });

  it("omits blank dates so the proxy applies its default window", () => {
    renderHook(() => useGuardrailsUsageOverview({ accessToken: "sk", startDate: "", endDate: "" }));

    expect(lastCall()[2]).toEqual({ params: { query: { start_date: undefined, end_date: undefined } } });
  });

  it("stays disabled without an access token", () => {
    renderHook(() => useGuardrailsUsageOverview({ accessToken: null, startDate: "2026-09-01", endDate: "2026-09-04" }));

    expect(lastCall()[3].enabled).toBe(false);
  });
});

describe("useGuardrailsUsageDetail", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useQueryMock.mockReturnValue({ data: undefined });
  });

  it("queries GET /guardrails/usage/detail/{guardrail_id} with the id as a path param", () => {
    renderHook(() =>
      useGuardrailsUsageDetail("bedrock-pii-mask", {
        accessToken: "sk",
        startDate: "2026-09-01",
        endDate: "2026-09-04",
      }),
    );

    expect(lastCall().slice(0, 3)).toEqual([
      "get",
      "/guardrails/usage/detail/{guardrail_id}",
      {
        params: {
          path: { guardrail_id: "bedrock-pii-mask" },
          query: { start_date: "2026-09-01", end_date: "2026-09-04" },
        },
      },
    ]);
    expect(lastCall()[3].enabled).toBe(true);
  });

  it("stays disabled without a guardrail id", () => {
    renderHook(() =>
      useGuardrailsUsageDetail("", { accessToken: "sk", startDate: "2026-09-01", endDate: "2026-09-04" }),
    );

    expect(lastCall()[3].enabled).toBe(false);
  });
});

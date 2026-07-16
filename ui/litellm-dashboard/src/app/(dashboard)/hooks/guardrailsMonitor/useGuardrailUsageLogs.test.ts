import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useGuardrailUsageLogs } from "./useGuardrailUsageLogs";

const useQueryMock = vi.fn();
vi.mock("@/lib/http/api", () => ({
  $api: { useQuery: (...args: unknown[]) => useQueryMock(...args) },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const params = {
  guardrailId: "guardrail-1",
  page: 1,
  pageSize: 50,
  startDate: "2026-07-01",
  endDate: "2026-07-08",
};

const lastCall = () => useQueryMock.mock.calls[useQueryMock.mock.calls.length - 1];
const lastInit = () => lastCall()[2] as { params: { query: Record<string, unknown> } };
const lastOptions = () => lastCall()[3] as { enabled: boolean };

describe("useGuardrailUsageLogs", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useQueryMock.mockReturnValue({ data: undefined });
    mockUseAuthorized.mockReturnValue({ accessToken: "test-access-token", userRole: "Admin" });
  });

  it("queries GET /guardrails/usage/logs", () => {
    renderHook(() => useGuardrailUsageLogs(params));
    expect(lastCall()[0]).toBe("get");
    expect(lastCall()[1]).toBe("/guardrails/usage/logs");
  });

  it("sends the date range so the derived query key refetches when it changes", () => {
    const expectedQuery = {
      guardrail_id: "guardrail-1",
      page: 1,
      page_size: 50,
      start_date: "2026-07-01",
      end_date: "2026-07-08",
    };

    renderHook(() => useGuardrailUsageLogs(params));

    expect(lastInit().params.query).toEqual(expectedQuery);
  });

  it("passes a changed date range through to the request", () => {
    const nextParams = { ...params, startDate: "2026-06-01", endDate: "2026-06-30" };
    const { rerender } = renderHook((p: typeof params) => useGuardrailUsageLogs(p), { initialProps: params });

    rerender(nextParams);

    expect(lastInit().params.query.start_date).toBe("2026-06-01");
    expect(lastInit().params.query.end_date).toBe("2026-06-30");
  });

  it("is enabled for an authorized user with a guardrail selected", () => {
    renderHook(() => useGuardrailUsageLogs(params));
    expect(lastOptions().enabled).toBe(true);
  });

  it("is disabled without an access token", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    renderHook(() => useGuardrailUsageLogs(params));
    expect(lastOptions().enabled).toBe(false);
  });

  it("is disabled without a guardrail id", () => {
    renderHook(() => useGuardrailUsageLogs({ ...params, guardrailId: "" }));
    expect(lastOptions().enabled).toBe(false);
  });
});

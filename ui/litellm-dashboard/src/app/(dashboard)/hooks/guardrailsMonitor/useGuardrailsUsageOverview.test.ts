import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useGuardrailsUsageOverview } from "./useGuardrailsUsageOverview";

const useQueryMock = vi.fn();
vi.mock("@/lib/http/api", () => ({
  $api: { useQuery: (...args: unknown[]) => useQueryMock(...args) },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const lastCall = () => useQueryMock.mock.calls[useQueryMock.mock.calls.length - 1];
const lastInit = () => lastCall()[2] as { params: { query: Record<string, unknown> } };
const lastOptions = () => lastCall()[3] as { enabled: boolean };

describe("useGuardrailsUsageOverview", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useQueryMock.mockReturnValue({ data: undefined });
    mockUseAuthorized.mockReturnValue({ accessToken: "test-access-token", userRole: "Admin" });
  });

  it("queries GET /guardrails/usage/overview with the selected date range", () => {
    renderHook(() => useGuardrailsUsageOverview("2026-07-01", "2026-07-08"));

    expect(lastCall()[0]).toBe("get");
    expect(lastCall()[1]).toBe("/guardrails/usage/overview");
    expect(lastInit().params.query).toEqual({ start_date: "2026-07-01", end_date: "2026-07-08" });
  });

  it("is enabled once an access token is present", () => {
    renderHook(() => useGuardrailsUsageOverview("2026-07-01", "2026-07-08"));
    expect(lastOptions().enabled).toBe(true);
  });

  it("is disabled without an access token", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    renderHook(() => useGuardrailsUsageOverview("2026-07-01", "2026-07-08"));
    expect(lastOptions().enabled).toBe(false);
  });
});

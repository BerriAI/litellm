import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useGuardrailUsageDetail } from "./useGuardrailUsageDetail";

const useQueryMock = vi.fn();
vi.mock("@/lib/http/api", () => ({
  $api: { useQuery: (...args: unknown[]) => useQueryMock(...args) },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const lastCall = () => useQueryMock.mock.calls[useQueryMock.mock.calls.length - 1];
const lastInit = () => lastCall()[2] as { params: { path: Record<string, unknown>; query: Record<string, unknown> } };
const lastOptions = () => lastCall()[3] as { enabled: boolean };

describe("useGuardrailUsageDetail", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useQueryMock.mockReturnValue({ data: undefined });
    mockUseAuthorized.mockReturnValue({ accessToken: "test-access-token", userRole: "Admin" });
  });

  it("queries the guardrail detail route with the id as a path param", () => {
    renderHook(() => useGuardrailUsageDetail("guardrail-1", "2026-07-01", "2026-07-08"));

    expect(lastCall()[0]).toBe("get");
    expect(lastCall()[1]).toBe("/guardrails/usage/detail/{guardrail_id}");
    expect(lastInit().params.path).toEqual({ guardrail_id: "guardrail-1" });
    expect(lastInit().params.query).toEqual({ start_date: "2026-07-01", end_date: "2026-07-08" });
  });

  it("is enabled for an authorized user with a guardrail selected", () => {
    renderHook(() => useGuardrailUsageDetail("guardrail-1", "2026-07-01", "2026-07-08"));
    expect(lastOptions().enabled).toBe(true);
  });

  it("is disabled without an access token", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    renderHook(() => useGuardrailUsageDetail("guardrail-1", "2026-07-01", "2026-07-08"));
    expect(lastOptions().enabled).toBe(false);
  });

  it("is disabled without a guardrail id", () => {
    renderHook(() => useGuardrailUsageDetail("", "2026-07-01", "2026-07-08"));
    expect(lastOptions().enabled).toBe(false);
  });
});

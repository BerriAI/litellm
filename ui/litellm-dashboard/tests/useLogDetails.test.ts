import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// ── Mocks ─────────────────────────────────────────────────────────────────────

vi.mock("@/components/networking", () => ({
  uiSpendLogDetailsCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

import { useLogDetails } from "../src/app/(dashboard)/hooks/logDetails/useLogDetails";
import { uiSpendLogDetailsCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

// ── Helpers ────────────────────────────────────────────────────────────────────

const DEFAULT_AUTH = {
  token: "mock-token",
  accessToken: "mock-access-token",
  userId: "user-1",
  userEmail: "user@example.com",
  userRole: "Admin",
  premiumUser: false,
  disabledPersonalKeyCreation: null,
  showSSOBanner: false,
};

function makeWrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);

  Wrapper.displayName = "QueryClientWrapper";
  return Wrapper;
}

// ── Tests ──────────────────────────────────────────────────────────────────────

describe("useLogDetails — conditional lazy loading", () => {
  const mockUseAuthorized = vi.mocked(useAuthorized);
  const mockApiCall = vi.mocked(uiSpendLogDetailsCall);

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue(DEFAULT_AUTH);
    mockApiCall.mockResolvedValue({ messages: [], response: {} });
  });

  it("should not call the API when enabled is false", () => {
    renderHook(() => useLogDetails("req-123", "2025-01-01 00:00:00", false), { wrapper: makeWrapper() });
    expect(mockApiCall).not.toHaveBeenCalled();
  });

  it("should not call the API when requestId is undefined", () => {
    renderHook(() => useLogDetails(undefined, "2025-01-01 00:00:00", true), { wrapper: makeWrapper() });
    expect(mockApiCall).not.toHaveBeenCalled();
  });

  it("should not call the API when startTime is undefined", () => {
    renderHook(() => useLogDetails("req-123", undefined, true), { wrapper: makeWrapper() });
    expect(mockApiCall).not.toHaveBeenCalled();
  });

  it("should not call the API when accessToken is null", () => {
    mockUseAuthorized.mockReturnValue({ ...DEFAULT_AUTH, accessToken: null });

    renderHook(() => useLogDetails("req-123", "2025-01-01 00:00:00", true), { wrapper: makeWrapper() });
    expect(mockApiCall).not.toHaveBeenCalled();
  });

  it("should call the API with accessToken, requestId and startTime when all conditions met", async () => {
    const { result } = renderHook(() => useLogDetails("req-123", "2025-01-01 00:00:00", true), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(mockApiCall).toHaveBeenCalledWith("mock-access-token", "req-123", "2025-01-01 00:00:00");
  });

  it("should return the data from the API response", async () => {
    const mockData = { messages: [{ role: "user", content: "hello" }], response: { id: "resp-1" } };
    mockApiCall.mockResolvedValue(mockData);

    const { result } = renderHook(() => useLogDetails("req-456", "2025-01-02 12:00:00", true), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockData);
  });

  it("should transition from disabled to enabled and trigger the API call", async () => {
    const { result, rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) => useLogDetails("req-789", "2025-01-03 00:00:00", enabled),
      { wrapper: makeWrapper(), initialProps: { enabled: false } },
    );

    expect(mockApiCall).not.toHaveBeenCalled();

    rerender({ enabled: true });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(mockApiCall).toHaveBeenCalledTimes(1);
    expect(mockApiCall).toHaveBeenCalledWith("mock-access-token", "req-789", "2025-01-03 00:00:00");
  });

  it("should expose isLoading=true while the API call is in progress", async () => {
    // Make the API call never resolve during this check
    let resolveCall!: (v: { messages: unknown[]; response: Record<string, unknown> }) => void;
    mockApiCall.mockReturnValue(
      new Promise((res) => {
        resolveCall = res;
      }),
    );

    const { result } = renderHook(() => useLogDetails("req-loading", "2025-01-04 00:00:00", true), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(true));

    // Clean up — resolve the pending promise to avoid open handles
    resolveCall({ messages: [], response: {} });
  });
});

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getLicenseInfo } from "@/components/networking";
import { useLicenseInfo } from "./useLicenseInfo";

vi.mock("@/components/networking", () => ({
  getLicenseInfo: vi.fn(),
}));

const mockGetLicenseInfo = vi.mocked(getLicenseInfo);

describe("useLicenseInfo", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
    mockGetLicenseInfo.mockResolvedValue({ has_license: true, expiration_date: "2030-01-01T00:00:00Z" } as Awaited<
      ReturnType<typeof getLicenseInfo>
    >);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])(
    "fetches /health/license for %s",
    async (role) => {
      const { result } = renderHook(() => useLicenseInfo("sk-test", role), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockGetLicenseInfo).toHaveBeenCalledTimes(1);
      expect(mockGetLicenseInfo).toHaveBeenCalledWith("sk-test");
      expect(result.current.data?.has_license).toBe(true);
    },
  );

  it.each(["Internal User", "Internal Viewer", "Org Admin", "App User", "Unknown Role", ""])(
    "never requests the admin-only license info for %s",
    async (role) => {
      const { result } = renderHook(() => useLicenseInfo("sk-test", role), { wrapper });

      await Promise.resolve();

      expect(mockGetLicenseInfo).not.toHaveBeenCalled();
      expect(result.current.isFetched).toBe(false);
      expect(result.current.data).toBeUndefined();
    },
  );

  it("reports missing license info as absent rather than as an error", async () => {
    const { result } = renderHook(() => useLicenseInfo("sk-test", "Internal User"), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.isError).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("does not fetch for an admin whose token has not resolved yet", async () => {
    renderHook(() => useLicenseInfo(null, "Admin"), { wrapper });

    await Promise.resolve();

    expect(mockGetLicenseInfo).not.toHaveBeenCalled();
  });
});

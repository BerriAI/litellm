import { getOnboardingCredentials, claimOnboardingToken } from "@/components/networking";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor, act } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useOnboardingCredentials, useClaimOnboardingToken } from "./useOnboarding";

vi.mock("@/components/networking", () => ({
  getOnboardingCredentials: vi.fn(),
  claimOnboardingToken: vi.fn(),
}));

const mockUseUIConfig = vi.fn();
vi.mock("@/app/(dashboard)/hooks/uiConfig/useUIConfig", () => ({
  useUIConfig: () => mockUseUIConfig(),
}));

const mockCredentialsResponse = { token: "mock.jwt.token", login_url: "http://example.com/login" };

describe("useOnboardingCredentials", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.clearAllMocks();
    mockUseUIConfig.mockReturnValue({ isLoading: false });
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("fetches credentials when inviteId is provided and UIConfig is loaded", async () => {
    (getOnboardingCredentials as any).mockResolvedValue(mockCredentialsResponse);

    const { result } = renderHook(() => useOnboardingCredentials("invite-123"), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockCredentialsResponse);
    expect(getOnboardingCredentials).toHaveBeenCalledWith("invite-123");
    expect(getOnboardingCredentials).toHaveBeenCalledTimes(1);
  });

  it("does not fetch when inviteId is null", async () => {
    const { result } = renderHook(() => useOnboardingCredentials(null), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.isFetched).toBe(false);
    expect(getOnboardingCredentials).not.toHaveBeenCalled();
  });

  it("does not fetch while UIConfig is loading", async () => {
    mockUseUIConfig.mockReturnValue({ isLoading: true });

    const { result } = renderHook(() => useOnboardingCredentials("invite-123"), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.isFetched).toBe(false);
    expect(getOnboardingCredentials).not.toHaveBeenCalled();
  });

  it("exposes error state when fetch fails", async () => {
    const error = new Error("Invalid invite");
    (getOnboardingCredentials as any).mockRejectedValue(error);

    const { result } = renderHook(() => useOnboardingCredentials("bad-invite"), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toEqual(error);
  });
});

describe("useClaimOnboardingToken", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.clearAllMocks();
    mockUseUIConfig.mockReturnValue({ isLoading: false });
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("calls claimOnboardingToken with correct params", async () => {
    (claimOnboardingToken as any).mockResolvedValue({ success: true });

    const { result } = renderHook(() => useClaimOnboardingToken(), { wrapper });

    act(() => {
      result.current.mutate({
        accessToken: "acc-token",
        inviteId: "invite-123",
        userId: "user-456",
        password: "secret",
      });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(claimOnboardingToken).toHaveBeenCalledWith("acc-token", "invite-123", "user-456", "secret");
  });

  it("exposes error state when mutation fails", async () => {
    const error = new Error("Claim failed");
    (claimOnboardingToken as any).mockRejectedValue(error);

    const { result } = renderHook(() => useClaimOnboardingToken(), { wrapper });

    act(() => {
      result.current.mutate({
        accessToken: "acc-token",
        inviteId: "invite-123",
        userId: "user-456",
        password: "secret",
      });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toEqual(error);
  });
});

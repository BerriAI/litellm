import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useUserInfo } from "./useUserInfo";
import { userGetInfoV2 } from "@/components/networking";
import type { UserInfoV2Response } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  userGetInfoV2: vi.fn(),
}));

vi.mock("../common/queryKeysFactory", () => ({
  createQueryKeys: vi.fn((resource: string) => ({
    all: [resource],
    lists: () => [resource, "list"],
    list: (params?: any) => [resource, "list", { params }],
    details: () => [resource, "detail"],
    detail: (uid: string) => [resource, "detail", uid],
  })),
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const mockSelectedUser: UserInfoV2Response = {
  user_id: "selected-user-id",
  user_email: "selected@example.com",
  user_alias: "Selected User",
  user_role: "internal_user",
  spend: 42.0,
  max_budget: 600.0,
  models: [],
  budget_duration: "30d",
  budget_reset_at: null,
  metadata: null,
  created_at: null,
  updated_at: null,
  sso_user_id: null,
  teams: [],
};

describe("useUserInfo", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userId: "admin-user-id",
      userRole: "Admin",
    });
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("fetches the specified user's info by user_id (not the caller's)", async () => {
    (userGetInfoV2 as any).mockResolvedValue(mockSelectedUser);

    const { result } = renderHook(() => useUserInfo("selected-user-id"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSelectedUser);
    // The whole point of the fix: the target user_id is passed through.
    expect(userGetInfoV2).toHaveBeenCalledWith("test-access-token", "selected-user-id");
    expect(userGetInfoV2).toHaveBeenCalledTimes(1);
  });

  it("does not execute when userId is null (global view has no single user)", async () => {
    const { result } = renderHook(() => useUserInfo(null), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(userGetInfoV2).not.toHaveBeenCalled();
  });

  it("does not execute when accessToken is missing", async () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null, userId: "admin-user-id", userRole: "Admin" });

    const { result } = renderHook(() => useUserInfo("selected-user-id"), { wrapper });

    expect(result.current.isFetched).toBe(false);
    expect(userGetInfoV2).not.toHaveBeenCalled();
  });

  it("surfaces the error when the lookup fails", async () => {
    const err = new Error("forbidden");
    (userGetInfoV2 as any).mockRejectedValue(err);

    const { result } = renderHook(() => useUserInfo("selected-user-id"), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toEqual(err);
  });
});

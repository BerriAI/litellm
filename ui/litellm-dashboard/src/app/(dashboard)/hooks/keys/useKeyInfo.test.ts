import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { keyInfoV1Call, userGetInfoV2 } from "@/components/networking";

import { useKeyInfo } from "./useKeyInfo";

vi.mock("@/components/networking", () => ({
  keyInfoV1Call: vi.fn(),
  userGetInfoV2: vi.fn(),
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const mockKeyInfoV1Call = vi.mocked(keyInfoV1Call);
const mockUserGetInfoV2 = vi.mocked(userGetInfoV2);

const createWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
  return { queryClient, wrapper };
};

const KEY_ID = "sk-key-1";
const ACCESS_TOKEN = "sk-access";

const OWNER = {
  user_id: "user-1",
  user_email: "owner@example.com",
  user_alias: "Budget Owner",
  user_role: "user",
  spend: 0,
  max_budget: 1500,
  models: [],
  budget_duration: "1mo",
  budget_reset_at: null,
  metadata: null,
  created_at: null,
  updated_at: null,
  sso_user_id: null,
  teams: [],
};

const keyInfoResponse = (info: Record<string, unknown>) => ({ info });

describe("useKeyInfo", () => {
  beforeEach(() => {
    mockKeyInfoV1Call.mockReset();
    mockUserGetInfoV2.mockReset();
    mockUseAuthorized.mockReturnValue({ accessToken: ACCESS_TOKEN });
  });

  it("attaches the owner's budget fields when the key has a user_id", async () => {
    mockKeyInfoV1Call.mockResolvedValue(keyInfoResponse({ user_id: "user-1", key_alias: "team-key" }));
    mockUserGetInfoV2.mockResolvedValue(OWNER);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKeyInfo(KEY_ID), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockUserGetInfoV2).toHaveBeenCalledWith(ACCESS_TOKEN, "user-1");
    const expectedUser = {
      user_id: "user-1",
      user_email: "owner@example.com",
      user_alias: "Budget Owner",
      max_budget: 1500,
      budget_duration: "1mo",
    };
    expect(result.current.data?.user).toEqual(expectedUser);
  });

  it("does not fetch an owner when the key has no user_id", async () => {
    mockKeyInfoV1Call.mockResolvedValue(keyInfoResponse({ user_id: null, key_alias: "service-key" }));
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKeyInfo(KEY_ID), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockUserGetInfoV2).not.toHaveBeenCalled();
    expect(result.current.data?.user).toBeUndefined();
  });

  it("still resolves the key data when the owner lookup fails", async () => {
    mockKeyInfoV1Call.mockResolvedValue(keyInfoResponse({ user_id: "user-1" }));
    mockUserGetInfoV2.mockRejectedValue(new Error("403"));
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKeyInfo(KEY_ID), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.token).toBe(KEY_ID);
    expect(result.current.data?.api_key).toBe(KEY_ID);
    expect(result.current.data?.user).toBeUndefined();
  });
});

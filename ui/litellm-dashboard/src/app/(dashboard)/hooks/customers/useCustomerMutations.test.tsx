import React, { type ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "@/components/networking";
import { $api } from "@/lib/http/api";
import {
  useCreateCustomer,
  useCustomer,
  useDeleteCustomers,
  useSetCustomerBlockedState,
  useSetCustomersBlockedState,
  useUpdateCustomer,
} from "./useCustomerMutations";

const authorized = vi.hoisted(() => vi.fn());
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: authorized,
}));
vi.mock("@/components/networking", () => ({ apiClient: { post: vi.fn() } }));
vi.mock("@/lib/http/api", () => ({ $api: { useQuery: vi.fn() } }));

const post = vi.mocked(apiClient.post);
const query = vi.mocked($api.useQuery);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return { queryClient, wrapper };
};

beforeEach(() => {
  vi.clearAllMocks();
  authorized.mockReturnValue({
    accessToken: "access-token",
    userRole: "Admin",
  });
});

describe("useCustomer", () => {
  it("fetches the requested customer with an admin access token", () => {
    renderHook(() => useCustomer("customer-id"));
    expect(query).toHaveBeenCalledWith(
      "get",
      "/customer/info",
      { params: { query: { end_user_id: "customer-id" } } },
      { enabled: true },
    );
  });

  it.each([
    {
      role: "Admin Viewer",
      accessToken: "access-token",
      id: "customer-id",
      enabled: true,
    },
    {
      role: "Internal User",
      accessToken: "access-token",
      id: "customer-id",
      enabled: false,
    },
    { role: "Admin", accessToken: null, id: "customer-id", enabled: false },
    { role: "Admin", accessToken: "access-token", id: null, enabled: false },
  ])("gates the query for $role with customer $id", ({ role, accessToken, id, enabled }) => {
    authorized.mockReturnValue({ accessToken, userRole: role });
    renderHook(() => useCustomer(id));
    expect(query).toHaveBeenCalledWith(
      "get",
      "/customer/info",
      { params: { query: { end_user_id: id ?? "" } } },
      { enabled },
    );
  });
});

describe("customer mutations", () => {
  it("invalidates both list and detail prefixes after a successful update", async () => {
    const { queryClient, wrapper } = createWrapper();
    const listKey = ["get", "/customer/list", {}];
    const detailKey = ["get", "/customer/info", { params: { query: { end_user_id: "customer-id" } } }];
    const otherDetailKey = ["get", "/customer/info", { params: { query: { end_user_id: "another-id" } } }];
    const unrelatedKey = ["get", "/key/list", {}];
    queryClient.setQueryData(listKey, []);
    queryClient.setQueryData(detailKey, {});
    queryClient.setQueryData(otherDetailKey, {});
    queryClient.setQueryData(unrelatedKey, []);
    post.mockResolvedValue({ user_id: "customer-id" });
    const { result } = renderHook(() => useUpdateCustomer(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync({
        user_id: "customer-id",
        alias: "Alias",
      });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(queryClient.getQueryState(listKey)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(detailKey)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(otherDetailKey)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(unrelatedKey)?.isInvalidated).toBe(false);
  });

  it.each([
    { name: "create", useHook: useCreateCustomer },
    { name: "update", useHook: useUpdateCustomer },
  ])("rejects $name without an access token", async ({ useHook }) => {
    authorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    const { wrapper } = createWrapper();
    const { result } = renderHook(useHook, { wrapper });

    await act(async () => {
      await expect(result.current.mutateAsync({ user_id: "customer-id" })).rejects.toThrow("Access token is required");
    });
    expect(post).not.toHaveBeenCalled();
  });

  it("rejects deletion without an access token", async () => {
    authorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    const { wrapper } = createWrapper();
    const { result } = renderHook(useDeleteCustomers, { wrapper });

    await act(async () => {
      await expect(result.current.mutateAsync(["customer-id"])).rejects.toThrow("Access token is required");
    });
    expect(post).not.toHaveBeenCalled();
  });

  it("rejects block state changes without an access token", async () => {
    authorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    const { wrapper } = createWrapper();
    const { result } = renderHook(useSetCustomerBlockedState, { wrapper });

    await act(async () => {
      await expect(result.current.mutateAsync({ userId: "customer-id", blocked: true })).rejects.toThrow(
        "Access token is required",
      );
    });
    expect(post).not.toHaveBeenCalled();
  });

  it.each([
    {
      blocked: true,
      path: "/customer/block",
      response: { blocked_users: [{ user_id: "customer-id", blocked: true }] },
    },
    {
      blocked: false,
      path: "/customer/unblock",
      response: { blocked_users: ["remaining-id"] },
    },
  ])("invalidates customer data after a per-row request to $path", async ({ blocked, path, response }) => {
    const { queryClient, wrapper } = createWrapper();
    const listKey = ["get", "/customer/list", {}];
    const detailKey = ["get", "/customer/info", { params: { query: { end_user_id: "customer-id" } } }];
    const unrelatedKey = ["get", "/key/list", {}];
    queryClient.setQueryData(listKey, []);
    queryClient.setQueryData(detailKey, {});
    queryClient.setQueryData(unrelatedKey, []);
    post.mockResolvedValue(response);
    const { result } = renderHook(useSetCustomerBlockedState, { wrapper });

    await act(async () => {
      await expect(result.current.mutateAsync({ userId: "customer-id", blocked })).resolves.toEqual(response);
    });

    expect(post).toHaveBeenCalledExactlyOnceWith(path, {
      accessToken: "access-token",
      body: { user_ids: ["customer-id"] },
    });
    expect(queryClient.getQueryState(listKey)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(detailKey)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(unrelatedKey)?.isInvalidated).toBe(false);
  });

  it("propagates per-row unblock errors and retains customer cache data", async () => {
    const { queryClient, wrapper } = createWrapper();
    const listKey = ["get", "/customer/list", {}];
    const detailKey = ["get", "/customer/info", { params: { query: { end_user_id: "customer-id" } } }];
    const customer = { user_id: "customer-id", blocked: true };
    queryClient.setQueryData(listKey, [customer]);
    queryClient.setQueryData(detailKey, customer);
    const error = new Error("Blocked user check was never set");
    post.mockRejectedValue(error);
    const { result } = renderHook(useSetCustomerBlockedState, { wrapper });

    await act(async () => {
      await expect(result.current.mutateAsync({ userId: "customer-id", blocked: false })).rejects.toBe(error);
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(queryClient.getQueryData(listKey)).toEqual([customer]);
    expect(queryClient.getQueryData(detailKey)).toEqual(customer);
    expect(queryClient.getQueryState(listKey)?.isInvalidated).toBe(false);
    expect(queryClient.getQueryState(detailKey)?.isInvalidated).toBe(false);
  });

  it.each([true, false])("invalidates customer data after bulk blocked state becomes %s", async (blocked) => {
    const { queryClient, wrapper } = createWrapper();
    const userIds = ["customer-id", "another-id"];
    const listKey = ["get", "/customer/list", {}];
    const detailKeys = userIds.map((id) => ["get", "/customer/info", { params: { query: { end_user_id: id } } }]);
    queryClient.setQueryData(listKey, []);
    detailKeys.forEach((key) => queryClient.setQueryData(key, {}));
    const response = blocked
      ? {
          blocked_users: userIds.map((user_id) => ({
            user_id,
            blocked: true,
          })),
        }
      : { blocked_users: ["remaining-id"] };
    post.mockResolvedValue(response);
    const { result } = renderHook(useSetCustomersBlockedState, { wrapper });

    await act(async () => {
      await expect(result.current.mutateAsync({ userIds, blocked })).resolves.toEqual(response);
    });

    expect(post).toHaveBeenCalledExactlyOnceWith(blocked ? "/customer/block" : "/customer/unblock", {
      accessToken: "access-token",
      body: { user_ids: userIds },
    });
    expect(queryClient.getQueryState(listKey)?.isInvalidated).toBe(true);
    detailKeys.forEach((key) => expect(queryClient.getQueryState(key)?.isInvalidated).toBe(true));
  });

  it.each([true, false])("retains customer cache data when a bulk action fails with blocked %s", async (blocked) => {
    const { queryClient, wrapper } = createWrapper();
    const listKey = ["get", "/customer/list", {}];
    const detailKey = ["get", "/customer/info", { params: { query: { end_user_id: "customer-id" } } }];
    const customer = { user_id: "customer-id", blocked: !blocked };
    queryClient.setQueryData(listKey, [customer]);
    queryClient.setQueryData(detailKey, customer);
    const error = new Error(blocked ? "Bulk block failed" : "Blocked user check was never set");
    post.mockRejectedValue(error);
    const { result } = renderHook(useSetCustomersBlockedState, { wrapper });

    await act(async () => {
      await expect(result.current.mutateAsync({ userIds: ["customer-id"], blocked })).rejects.toBe(error);
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(queryClient.getQueryData(listKey)).toEqual([customer]);
    expect(queryClient.getQueryData(detailKey)).toEqual(customer);
    expect(queryClient.getQueryState(listKey)?.isInvalidated).toBe(false);
    expect(queryClient.getQueryState(detailKey)?.isInvalidated).toBe(false);
  });

  it("rejects bulk blocked state changes without an access token", async () => {
    authorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    const { wrapper } = createWrapper();
    const { result } = renderHook(useSetCustomersBlockedState, { wrapper });

    await act(async () => {
      await expect(result.current.mutateAsync({ userIds: ["customer-id"], blocked: true })).rejects.toThrow(
        "Access token is required",
      );
    });
    expect(post).not.toHaveBeenCalled();
  });
});

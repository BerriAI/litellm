import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "@/components/networking";
import {
  bulkUpdateCustomers,
  createCustomer,
  deleteCustomers,
  setCustomerBlockedState,
  setCustomersBlockedState,
  updateCustomer,
  type BulkUpdateCustomersRequest,
  type BulkUpdateCustomersResponse,
} from "./customerApi";

vi.mock("@/components/networking", () => ({ apiClient: { post: vi.fn() } }));

const post = vi.mocked(apiClient.post);

describe("customer API", () => {
  beforeEach(() => {
    post.mockReset();
  });

  it.each([
    { name: "create", action: createCustomer, path: "/customer/new" },
    { name: "update", action: updateCustomer, path: "/customer/update" },
  ])("posts the $name payload and returns the customer", async ({ action, path }) => {
    const body = { user_id: "customer-id", alias: "Alias" };
    const response = { ...body, blocked: false, spend: 0 };
    post.mockResolvedValue(response);

    await expect(action("access-token", body)).resolves.toEqual(response);
    expect(post).toHaveBeenCalledWith(path, {
      accessToken: "access-token",
      body,
    });
  });

  it("posts the customer IDs to delete", async () => {
    const userIds = ["customer-id", "another-id"];
    const response = {
      deleted_customers: userIds.length,
      message: "Customers deleted",
    };
    post.mockResolvedValue(response);

    await expect(deleteCustomers("access-token", userIds)).resolves.toEqual(response);
    expect(post).toHaveBeenCalledWith("/customer/delete", {
      accessToken: "access-token",
      body: { user_ids: userIds },
    });
  });

  it("posts all customer patches in one request and returns each result", async () => {
    const body: BulkUpdateCustomersRequest = {
      customers: [
        { user_id: "customer-id", object_permission: { mcp_toolsets: ["toolset-1"] } },
        { user_id: "another-id", models: [] },
      ],
    };
    const response: BulkUpdateCustomersResponse = {
      total_requested: 2,
      successful_updates: 2,
      failed_updates: 0,
      results: body.customers.map(({ user_id }) => ({
        user_id,
        success: true,
        updated_customer: { user_id, blocked: false, spend: 0 },
        error: null,
      })),
    };
    post.mockResolvedValue(response);

    await expect(bulkUpdateCustomers("access-token", body)).resolves.toBe(response);
    expect(post).toHaveBeenCalledExactlyOnceWith("/customer/bulk_update", {
      accessToken: "access-token",
      body,
    });
  });

  it("preserves failed customer results and counts from a successful batch response", async () => {
    const body: BulkUpdateCustomersRequest = {
      customers: [
        { user_id: "customer-id", object_permission: { mcp_toolsets: ["toolset-1"] } },
        { user_id: "missing-id", object_permission: { mcp_toolsets: ["toolset-1"] } },
      ],
    };
    const response: BulkUpdateCustomersResponse = {
      total_requested: 2,
      successful_updates: 1,
      failed_updates: 1,
      results: [
        {
          user_id: "customer-id",
          success: true,
          updated_customer: { user_id: "customer-id", blocked: false, spend: 0 },
          error: null,
        },
        { user_id: "missing-id", success: false, updated_customer: null, error: "Customer does not exist" },
      ],
    };
    post.mockResolvedValue(response);

    await expect(bulkUpdateCustomers("access-token", body)).resolves.toBe(response);
    expect(post).toHaveBeenCalledExactlyOnceWith("/customer/bulk_update", {
      accessToken: "access-token",
      body,
    });
  });

  it("propagates a bulk update request failure", async () => {
    const error = new Error("Bulk update unavailable");
    post.mockRejectedValue(error);

    await expect(
      bulkUpdateCustomers("access-token", { customers: [{ user_id: "customer-id", models: [] }] }),
    ).rejects.toBe(error);
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
  ])("posts one customer ID to $path", async ({ blocked, path, response }) => {
    const userId = "customer-id";
    post.mockResolvedValue(response);

    await expect(setCustomerBlockedState("access-token", { userId, blocked })).resolves.toEqual(response);
    expect(post).toHaveBeenCalledExactlyOnceWith(path, {
      accessToken: "access-token",
      body: { user_ids: [userId] },
    });
  });

  it.each([true, false])("propagates per-row action failures when blocked is %s", async (blocked) => {
    const error = new Error(blocked ? "Customer action failed" : "Blocked user check was never set");
    post.mockRejectedValue(error);

    await expect(
      setCustomerBlockedState("access-token", {
        userId: "customer-id",
        blocked,
      }),
    ).rejects.toBe(error);
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
  ])("posts all selected customer IDs to $path", async ({ blocked, path, response }) => {
    const userIds = ["customer-id", "another-id"];
    post.mockResolvedValue(response);

    await expect(setCustomersBlockedState("access-token", { userIds, blocked })).resolves.toEqual(response);
    expect(post).toHaveBeenCalledExactlyOnceWith(path, {
      accessToken: "access-token",
      body: { user_ids: userIds },
    });
  });

  it.each([true, false])("propagates bulk action failures when blocked is %s", async (blocked) => {
    const error = new Error(blocked ? "Customer action failed" : "Blocked user check was never set");
    post.mockRejectedValue(error);

    await expect(
      setCustomersBlockedState("access-token", {
        userIds: ["customer-id", "another-id"],
        blocked,
      }),
    ).rejects.toBe(error);
  });
});

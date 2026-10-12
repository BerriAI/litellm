import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "@/components/networking";
import {
  createCustomer,
  deleteCustomers,
  setCustomerBlockedState,
  setCustomersBlockedState,
  updateCustomer,
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

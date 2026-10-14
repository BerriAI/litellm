import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { BulkUpdateCustomersResponse } from "@/app/(dashboard)/hooks/customers/customerApi";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { ApiError } from "@/lib/http/client";
import { toast } from "@/lib/toast";
import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";
import { BulkEditCustomersModal } from "./BulkEditCustomersModal";

const { post, get, fetchServers, fetchGroups, fetchToolsets, fetchModels } = vi.hoisted(() => ({
  post: vi.fn(),
  get: vi.fn(),
  fetchServers: vi.fn(),
  fetchGroups: vi.fn(),
  fetchToolsets: vi.fn(),
  fetchModels: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userId: "admin", userRole: "Admin", premiumUser: false }),
}));
vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  apiClient: { post, get },
  fetchMCPServers: fetchServers,
  fetchMCPAccessGroups: fetchGroups,
  fetchMCPToolsets: fetchToolsets,
  modelAvailableCall: fetchModels,
}));

const customers: EndUser[] = [
  { user_id: "customer-a", alias: "Original A", blocked: true, spend: 0, models: ["old-model-a"] },
  { user_id: "customer-b", alias: "Original B", blocked: false, spend: 0, models: ["old-model-b"] },
];
const response = (failedIds: string[] = [], rows = customers): BulkUpdateCustomersResponse => ({
  total_requested: rows.length,
  successful_updates: rows.length - failedIds.length,
  failed_updates: failedIds.length,
  results: rows.map((customer) => ({
    user_id: customer.user_id,
    success: !failedIds.includes(customer.user_id),
    error: failedIds.includes(customer.user_id) ? `Cannot update ${customer.user_id}` : null,
    updated_customer: failedIds.includes(customer.user_id) ? null : customer,
  })),
});

const mountModal = (rows = customers) => {
  const onCancel = vi.fn();
  const onCompleted = vi.fn();
  const view = renderWithProviders(
    <BulkEditCustomersModal open selectedCustomers={rows} onCancel={onCancel} onCompleted={onCompleted} />,
  );
  return { ...view, onCancel, onCompleted };
};

const optIntoToolsets = async (user: ReturnType<typeof userEvent.setup>) => {
  const checkbox = screen.getByRole("checkbox", { name: "Update MCP toolsets" });
  await waitFor(() => expect(checkbox).not.toHaveAttribute("aria-disabled", "true"));
  await user.click(checkbox);
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "MCP toolsets" }), "Research tools");
  await user.keyboard("{Escape}");
};

describe("Bulk customer editor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    get.mockResolvedValue([{ budget_id: "replacement-budget" }]);
    fetchModels.mockResolvedValue({ data: [{ id: "model-new" }] });
    fetchServers.mockResolvedValue([{ server_id: "server-new", server_name: "Research server" }]);
    fetchGroups.mockResolvedValue(["research-group"]);
    fetchToolsets.mockResolvedValue([{ toolset_id: "toolset-new", toolset_name: "Research tools", tools: [] }]);
    post.mockResolvedValue(response());
  });

  it("freezes selected IDs and sends a toolsets-only patch before completing all customers", async () => {
    const user = userEvent.setup();
    const { rerender, onCancel, onCompleted } = mountModal();
    rerender(
      <BulkEditCustomersModal
        open
        selectedCustomers={[{ ...customers[0], user_id: "later-selection" }]}
        onCancel={onCancel}
        onCompleted={onCompleted}
      />,
    );
    const targets = screen.getByRole("region", { name: "Customers to update" });
    expect(targets).toHaveTextContent("customer-a");
    expect(targets).toHaveTextContent("customer-b");
    expect(targets).not.toHaveTextContent("later-selection");
    await optIntoToolsets(user);
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(onCancel).toHaveBeenCalledOnce());
    expect(onCompleted).toHaveBeenCalledExactlyOnceWith(["customer-a", "customer-b"]);
    expect(post).toHaveBeenCalledExactlyOnceWith("/customer/bulk_update", {
      accessToken: "sk-test",
      body: {
        customers: [
          { user_id: "customer-a", object_permission: { mcp_toolsets: ["toolset-new"] } },
          { user_id: "customer-b", object_permission: { mcp_toolsets: ["toolset-new"] } },
        ],
      },
    });
  });

  it("wires independently enabled budget, models, region, default model, servers and groups", async () => {
    const user = userEvent.setup();
    const { onCancel } = mountModal();
    await waitFor(() =>
      expect(screen.getByRole("checkbox", { name: "Update budget" })).not.toHaveAttribute("aria-disabled", "true"),
    );
    await user.click(screen.getByRole("checkbox", { name: "Update budget" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget" }), "replacement-budget");
    await user.click(screen.getByRole("checkbox", { name: "Update Allowed models" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Allowed models" }), "model-new");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("checkbox", { name: "Update allowed region" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Allowed region" }), "EU");
    await user.click(screen.getByRole("checkbox", { name: "Update default model" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Default model" }), "model-new");
    await user.click(screen.getByRole("checkbox", { name: "Update MCP servers" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "MCP servers" }), "Research server");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("checkbox", { name: "Update MCP access groups" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "MCP access groups" }), "research-group");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(onCancel).toHaveBeenCalledOnce());
    expect(post).toHaveBeenCalledExactlyOnceWith("/customer/bulk_update", {
      accessToken: "sk-test",
      body: {
        customers: customers.map((customer) => ({
          user_id: customer.user_id,
          budget_id: "replacement-budget",
          models: ["model-new"],
          allowed_model_region: "eu",
          default_model: "model-new",
          object_permission: {
            mcp_servers: ["server-new"],
            mcp_access_groups: ["research-group"],
            mcp_tool_permissions: {},
          },
        })),
      },
    });
  });

  it("removes deselected server tool grants and retries only the failed customer's own permissions", async () => {
    const rows = customers.map((customer, index) => ({
      ...customer,
      object_permission: {
        mcp_servers: ["server-new", "server-removed"],
        mcp_tool_permissions: { "Research server": [`read-${index}`], "server-removed": ["delete"] },
      },
    }));
    fetchServers.mockResolvedValue([
      { server_id: "server-new", server_name: "Research server" },
      { server_id: "server-removed", server_name: "Removed server" },
    ]);
    post.mockResolvedValueOnce(response(["customer-b"], rows)).mockResolvedValueOnce(response([], [rows[1]]));
    const user = userEvent.setup();
    const { onCancel } = mountModal(rows);
    const checkbox = screen.getByRole("checkbox", { name: "Update MCP servers" });
    await waitFor(() => expect(checkbox).not.toHaveAttribute("aria-disabled", "true"));
    await user.click(checkbox);
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "MCP servers" }), "Research server");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    expect(await screen.findByRole("alert", { name: "Bulk update results" })).toHaveTextContent("customer-b");
    expect(post).toHaveBeenNthCalledWith(1, "/customer/bulk_update", {
      accessToken: "sk-test",
      body: {
        customers: rows.map((customer, index) => ({
          user_id: customer.user_id,
          object_permission: { mcp_servers: ["server-new"], mcp_tool_permissions: { "server-new": [`read-${index}`] } },
        })),
      },
    });
    await user.click(screen.getByRole("button", { name: "Retry failed customers" }));
    await waitFor(() => expect(onCancel).toHaveBeenCalledOnce());
    expect(post).toHaveBeenNthCalledWith(2, "/customer/bulk_update", {
      accessToken: "sk-test",
      body: {
        customers: [
          {
            user_id: "customer-b",
            object_permission: { mcp_servers: ["server-new"], mcp_tool_permissions: { "server-new": ["read-1"] } },
          },
        ],
      },
    });
  });

  it("sends an explicitly empty array when an enabled MCP list is left empty", async () => {
    const user = userEvent.setup();
    const { onCancel } = mountModal();
    const checkbox = screen.getByRole("checkbox", { name: "Update MCP servers" });
    await waitFor(() => expect(checkbox).not.toHaveAttribute("aria-disabled", "true"));
    await user.click(checkbox);
    expect(
      screen.getByText(
        "An empty selection clears direct server grants. Tool grants for servers outside this selection are removed. Clearing all customer MCP grants removes the customer restriction.",
      ),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(onCancel).toHaveBeenCalledOnce());
    expect(post).toHaveBeenCalledExactlyOnceWith("/customer/bulk_update", {
      accessToken: "sk-test",
      body: {
        customers: customers.map((customer) => ({
          user_id: customer.user_id,
          object_permission: { mcp_servers: [], mcp_tool_permissions: {} },
        })),
      },
    });
  });

  it.each([
    { failedIds: ["customer-b"], successToast: "1 customer updated" },
    { failedIds: ["customer-a", "customer-b"], successToast: "2 customers updated" },
  ])("keeps failure IDs and errors for retry: %j", async ({ failedIds, successToast }) => {
    post.mockResolvedValueOnce(response(failedIds));
    const user = userEvent.setup();
    const { onCancel, onCompleted } = mountModal();
    await optIntoToolsets(user);
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    const summary = await screen.findByRole("alert", { name: "Bulk update results" });
    expect(summary).toHaveTextContent(`${2 - failedIds.length} succeeded, ${failedIds.length} failed out of 2`);
    for (const id of failedIds) expect(summary).toHaveTextContent(`Cannot update ${id}`);
    const targets = screen.getByRole("region", { name: "Customers to update" });
    expect(within(targets).getAllByRole("listitem")).toHaveLength(failedIds.length);
    expect(onCancel).not.toHaveBeenCalled();
    if (failedIds.length === 1) expect(onCompleted).toHaveBeenCalledExactlyOnceWith(["customer-a"]);
    else expect(onCompleted).not.toHaveBeenCalled();
    const retryRows = customers.filter((customer) => failedIds.includes(customer.user_id));
    post.mockResolvedValueOnce(response([], retryRows));
    await user.click(screen.getByRole("button", { name: "Retry failed customers" }));
    await waitFor(() => expect(onCancel).toHaveBeenCalledOnce());
    expect(post).toHaveBeenLastCalledWith("/customer/bulk_update", {
      accessToken: "sk-test",
      body: {
        customers: retryRows.map((customer) => ({
          user_id: customer.user_id,
          object_permission: { mcp_toolsets: ["toolset-new"] },
        })),
      },
    });
    expect(onCompleted).toHaveBeenLastCalledWith(failedIds);
    expect(toast.success).toHaveBeenCalledExactlyOnceWith(successToast);
  });

  it("keeps all targets after an HTTP error and allows the same request to be retried", async () => {
    post.mockRejectedValueOnce(new ApiError("Proxy is unavailable", 503, { error: "Proxy is unavailable" }));
    const user = userEvent.setup();
    const { onCancel, onCompleted } = mountModal();
    await optIntoToolsets(user);
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    expect(await screen.findByRole("alert", { name: "Bulk update error" })).toHaveTextContent("Proxy is unavailable");
    expect(onCancel).not.toHaveBeenCalled();
    expect(onCompleted).not.toHaveBeenCalled();
    expect(screen.getByRole("region", { name: "Customers to update" })).toHaveTextContent("customer-b");
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(onCancel).toHaveBeenCalledOnce());
    expect(post).toHaveBeenCalledTimes(2);
    expect(post.mock.calls[1]).toEqual(post.mock.calls[0]);
  });

  it.each([
    "{'error': 'Proxy is unavailable'}",
    JSON.stringify({ error: { message: "{'error': 'Proxy is unavailable'}" } }),
  ])("shows the readable transport error from a wrapped proxy response: %s", async (message) => {
    post.mockRejectedValueOnce(new ApiError(message, 503, { error: message }));
    const user = userEvent.setup();
    const { onCancel, onCompleted } = mountModal();
    await optIntoToolsets(user);
    await user.click(screen.getByRole("button", { name: "Apply changes" }));

    expect(await screen.findByRole("alert", { name: "Bulk update error" })).toHaveTextContent(/^Proxy is unavailable$/);
    expect(onCancel).not.toHaveBeenCalled();
    expect(onCompleted).not.toHaveBeenCalled();
    expect(within(screen.getByRole("region", { name: "Customers to update" })).getAllByRole("listitem")).toHaveLength(
      2,
    );
  });

  it.each([
    "{'error': 'Cannot update this customer'}",
    JSON.stringify({ error: { message: "{'error': 'Cannot update this customer'}" } }),
  ])("shows a readable per-customer error and keeps the failed target: %s", async (message) => {
    const failedResponse = response(["customer-b"]);
    post.mockResolvedValueOnce({
      ...failedResponse,
      results: failedResponse.results.map((result) => (result.success ? result : { ...result, error: message })),
    });
    const user = userEvent.setup();
    const { onCancel, onCompleted } = mountModal();
    await optIntoToolsets(user);
    await user.click(screen.getByRole("button", { name: "Apply changes" }));

    const summary = await screen.findByRole("alert", { name: "Bulk update results" });
    expect(within(summary).getByRole("listitem")).toHaveTextContent(/^customer-b: Cannot update this customer$/);
    expect(screen.getByRole("region", { name: "Customers to update" })).toHaveTextContent(/^customer-b$/);
    expect(onCancel).not.toHaveBeenCalled();
    expect(onCompleted).toHaveBeenCalledExactlyOnceWith(["customer-a"]);
  });

  it("prevents duplicate submissions and closing while the request is pending", async () => {
    const deferred = Promise.withResolvers<BulkUpdateCustomersResponse>();
    post.mockReturnValueOnce(deferred.promise);
    const user = userEvent.setup();
    const { onCancel } = mountModal();
    await optIntoToolsets(user);
    const form = screen.getByRole("form", { name: "Bulk customer edits" });
    fireEvent.submit(form);
    fireEvent.submit(form);
    expect(await screen.findByRole("button", { name: "Applying changes…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Close" })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(onCancel).not.toHaveBeenCalled();
    expect(post).toHaveBeenCalledOnce();
    deferred.resolve(response());
    await waitFor(() => expect(onCancel).toHaveBeenCalledOnce());
  });

  it("shows unavailable catalogs and leaves other fields editable, then retries the catalog", async () => {
    const loadingModels = Promise.withResolvers<{ data: { id: string }[] }>();
    fetchModels.mockReturnValueOnce(loadingModels.promise);
    fetchToolsets.mockRejectedValueOnce(new Error("Catalog unavailable"));
    const user = userEvent.setup();
    mountModal();
    expect(screen.getByText("Loading Allowed models…")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Update Allowed models" })).toHaveAttribute("aria-disabled", "true");
    expect(await screen.findByText("MCP toolsets could not be loaded")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Update MCP toolsets" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("checkbox", { name: "Update allowed region" })).not.toHaveAttribute(
      "aria-disabled",
      "true",
    );
    await user.click(screen.getByRole("button", { name: "Retry MCP toolsets" }));
    await waitFor(() =>
      expect(screen.getByRole("checkbox", { name: "Update MCP toolsets" })).not.toHaveAttribute(
        "aria-disabled",
        "true",
      ),
    );
    expect(screen.queryByText("MCP toolsets could not be loaded")).not.toBeInTheDocument();
    loadingModels.resolve({ data: [{ id: "model-new" }] });
    await waitFor(() =>
      expect(screen.getByRole("checkbox", { name: "Update Allowed models" })).not.toHaveAttribute(
        "aria-disabled",
        "true",
      ),
    );
  });

  it("disables maximum edits for linked budgets and explains the effect on unselected customers", async () => {
    const user = userEvent.setup();
    mountModal([customers[0], { ...customers[1], budget_id: "shared-budget" }]);
    expect(screen.getByRole("note")).toHaveTextContent("can affect unselected customers");
    await user.click(screen.getByRole("combobox", { name: "Budget update" }));
    expect(await screen.findByRole("option", { name: "Set a maximum budget" })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("spinbutton", { name: "Max Budget (USD)" })).not.toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });

  it("sets a positive maximum for unlinked customers without sending a budget link", async () => {
    const user = userEvent.setup();
    const { onCancel } = mountModal();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget update" }), "Set a maximum budget");
    await user.click(screen.getByRole("checkbox", { name: "Update budget" }));
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max Budget (USD)" }), { target: { value: "25" } });
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(onCancel).toHaveBeenCalledOnce());
    expect(post).toHaveBeenCalledExactlyOnceWith("/customer/bulk_update", {
      accessToken: "sk-test",
      body: { customers: customers.map((customer) => ({ user_id: customer.user_id, max_budget: 25 })) },
    });
  });

  it("allows a failed catalog edit to be unchecked so another independent edit can proceed", async () => {
    const user = userEvent.setup();
    const { onCancel } = mountModal();
    await optIntoToolsets(user);
    await user.click(screen.getByRole("checkbox", { name: "Update allowed region" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Allowed region" }), "US");
    fetchToolsets.mockRejectedValueOnce(new Error("Catalog unavailable"));
    await act(async () => {
      await testQueryClient.invalidateQueries({ queryKey: ["mcpToolsets"] });
    });
    expect(await screen.findByText("MCP toolsets could not be loaded")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    expect(await screen.findByRole("alert", { name: "Bulk update error" })).toHaveTextContent(
      "MCP toolsets must finish loading",
    );
    expect(post).not.toHaveBeenCalled();
    const checkbox = screen.getByRole("checkbox", { name: "Update MCP toolsets" });
    expect(checkbox).not.toHaveAttribute("aria-disabled", "true");
    await user.click(checkbox);
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(onCancel).toHaveBeenCalledOnce());
    expect(post).toHaveBeenCalledExactlyOnceWith("/customer/bulk_update", {
      accessToken: "sk-test",
      body: { customers: customers.map((customer) => ({ user_id: customer.user_id, allowed_model_region: "us" })) },
    });
  });

  it("requires an opted-in field and cancels without updating the selected customers", async () => {
    const user = userEvent.setup();
    const { onCancel, onCompleted } = mountModal();
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    expect(await screen.findByRole("alert", { name: "Bulk update error" })).toHaveTextContent(
      "Choose at least one field to update",
    );
    expect(post).not.toHaveBeenCalled();
    expect(onCompleted).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledOnce();
  });
});

import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Toaster, toast as sonner } from "sonner";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/http/client";

import type { BulkUpdateCustomersResponse } from "@/app/(dashboard)/hooks/customers/customerApi";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";
import CustomersPanel from "./CustomersPanel";

const { post, get, useQuery, fetchServers, fetchToolsets, listTools, refetch, authorization } = vi.hoisted(() => ({
  post: vi.fn(),
  get: vi.fn(),
  useQuery: vi.fn(),
  fetchServers: vi.fn(),
  fetchToolsets: vi.fn(),
  listTools: vi.fn(),
  refetch: vi.fn(),
  authorization: { accessToken: "sk-test", userRole: "Admin", userId: "admin", isViewOnly: false, premiumUser: false },
}));

vi.unmock("@/lib/toast");

vi.mock("@/lib/http/api", () => ({ $api: { useQuery } }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => authorization }));
vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  apiClient: { post, get },
  fetchMCPServers: fetchServers,
  listMCPTools: listTools,
  fetchMCPToolsets: fetchToolsets,
  fetchMCPAccessGroups: vi.fn().mockResolvedValue([]),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

const activeCustomer: EndUser = {
  user_id: "alice@example.test",
  alias: "Alice",
  blocked: false,
  spend: 12.5,
  models: [],
  object_permission: {
    object_permission_id: "permission-alice",
    mcp_servers: [],
    mcp_access_groups: ["research"],
    mcp_toolsets: ["toolset-1"],
    mcp_tool_permissions: {},
    agents: [],
    agent_access_groups: [],
    models: [],
    search_tools: [],
    vector_stores: [],
    blocked_tools: [],
  },
};

const blockedCustomer: EndUser = {
  user_id: "bob@example.test",
  alias: "Bob",
  blocked: true,
  spend: 0,
  models: [],
};

const customers = [activeCustomer, blockedCustomer];

afterEach(() => vi.unstubAllGlobals());

const confirmBlocking = async (user: ReturnType<typeof userEvent.setup>, action: string, count: number) => {
  const dialog = await screen.findByRole("dialog", { name: `${action} ${count} customer${count === 1 ? "" : "s"}?` });
  expect(post).not.toHaveBeenCalled();
  await user.click(within(dialog).getByRole("button", { name: action }));
};

describe("Customers management", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sonner.dismiss();
    testQueryClient.clear();
    authorization.userRole = "Admin";
    authorization.isViewOnly = false;
    authorization.premiumUser = false;
    get.mockResolvedValue([]);
    post.mockResolvedValue(activeCustomer);
    fetchServers.mockResolvedValue([]);
    fetchToolsets.mockResolvedValue([]);
    listTools.mockResolvedValue({ tools: [], error: false });
    useQuery.mockImplementation(
      (_method: string, path: string, init: { params?: { query?: { end_user_id?: string } } }) => ({
        data:
          path === "/customer/list"
            ? customers
            : customers.find((customer) => customer.user_id === init.params?.query?.end_user_id),
        isLoading: false,
        isError: false,
        isFetching: false,
        refetch,
      }),
    );
  });

  it("refreshes the customer list", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />);
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(refetch).toHaveBeenCalledOnce();
  });

  it("edits selected customers across pages in one toolsets-only batch without Enterprise", async () => {
    const rows = Array.from(
      { length: 26 },
      (_, index): EndUser => ({
        ...activeCustomer,
        user_id: `customer-${String(index).padStart(2, "0")}`,
        alias: null,
      }),
    );
    const queryResult = { data: rows, isLoading: false, isError: false, isFetching: false, refetch };
    useQuery.mockReturnValue(queryResult);
    fetchToolsets.mockResolvedValue([{ toolset_id: "toolset-new", toolset_name: "Research tools", tools: [] }]);
    const response = {
      results: rows.map((row) => ({ user_id: row.user_id, success: true, updated_customer: row, error: null })),
      total_requested: 26,
      successful_updates: 26,
      failed_updates: 0,
    };
    post.mockResolvedValue(response);
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />, { searchParams: "page_size=25" });
    expect(screen.queryByRole("button", { name: "Edit selected" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    await user.click(screen.getByRole("button", { name: "Go to next page" }));
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    expect(screen.getByText("26 selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Block selected" })).toHaveAttribute("aria-disabled", "true");
    await user.click(screen.getByRole("button", { name: "Edit selected" }));
    const dialog = await screen.findByRole("dialog", { name: "Edit selected customers" });
    await user.click(within(dialog).getByRole("checkbox", { name: "Update MCP toolsets" }));
    await user.click(within(dialog).getByRole("combobox", { name: "MCP toolsets" }));
    await user.click(await screen.findByRole("option", { name: "Research tools" }));
    await user.keyboard("{Escape}");
    await user.click(within(dialog).getByRole("button", { name: "Apply changes" }));
    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/bulk_update", {
        accessToken: "sk-test",
        body: {
          customers: rows.map((row) => ({
            user_id: row.user_id,
            object_permission: { mcp_toolsets: ["toolset-new"], mcp_tool_permissions: {} },
          })),
        },
      }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("checkbox", { name: "Select customer customer-25" })).not.toBeChecked();
    expect(screen.queryByRole("button", { name: "Edit selected" })).not.toBeInTheDocument();
  });

  it("guards a shared maximum even when the other customer is hidden by search", async () => {
    const rows = customers.map((customer) => ({ ...customer, budget_id: "shared-budget" }));
    const queryResult = { data: rows, isLoading: false, isError: false, isFetching: false, refetch };
    useQuery.mockReturnValue(queryResult);
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />, { searchParams: "search=Alice" });
    expect(
      screen.queryByRole("checkbox", { name: `Select customer ${blockedCustomer.user_id}` }),
    ).not.toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: `Select customer ${activeCustomer.user_id}` }));
    await user.click(screen.getByRole("button", { name: "Edit selected" }));
    await user.click(screen.getByRole("checkbox", { name: "Update budget" }));
    expect(await screen.findByRole("note")).toHaveTextContent("Some selected customers share a budget");
    await user.click(screen.getByRole("combobox", { name: "Budget update" }));
    expect(await screen.findByRole("option", { name: "Set a maximum budget" })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    expect(post).not.toHaveBeenCalled();
  });

  it("keeps only failed customers selected after cancellation and list refresh before reopening", async () => {
    const { $api } = await vi.importActual<typeof import("@/lib/http/api")>("@/lib/http/api");
    useQuery.mockImplementation($api.useQuery);
    const extraCustomer: EndUser = { ...activeCustomer, user_id: "new@example.test", alias: "New customer" };
    const refreshedCustomers: EndUser[] = [
      { ...activeCustomer, alias: "Updated customer", spend: 25 },
      { ...blockedCustomer, alias: "Refreshed failed customer" },
      extraCustomer,
    ];
    const network = vi.fn((request: Request) => {
      expect(new URL(request.url).pathname).toBe("/customer/list");
      return Promise.resolve(Response.json(refreshedCustomers));
    });
    network.mockImplementationOnce(() => Promise.resolve(Response.json(customers)));
    vi.stubGlobal("fetch", network);
    fetchToolsets.mockResolvedValue([{ toolset_id: "toolset-new", toolset_name: "Research tools", tools: [] }]);
    const mixedResponse: BulkUpdateCustomersResponse = {
      total_requested: 2,
      successful_updates: 1,
      failed_updates: 1,
      results: [
        { user_id: activeCustomer.user_id, success: true, updated_customer: refreshedCustomers[0], error: null },
        { user_id: blockedCustomer.user_id, success: false, updated_customer: null, error: "Customer update failed" },
      ],
    };
    const retryResponse: BulkUpdateCustomersResponse = {
      total_requested: 1,
      successful_updates: 1,
      failed_updates: 0,
      results: [
        { user_id: blockedCustomer.user_id, success: true, updated_customer: refreshedCustomers[1], error: null },
      ],
    };
    post.mockResolvedValueOnce(mixedResponse).mockResolvedValueOnce(retryResponse);
    const user = userEvent.setup();
    const selectToolset = async () => {
      await user.click(screen.getByRole("checkbox", { name: "Update MCP toolsets" }));
      await chooseSelectOption(user, screen.getByRole("combobox", { name: "MCP toolsets" }), "Research tools");
      await user.keyboard("{Escape}");
    };
    renderWithProviders(<CustomersPanel />);
    await screen.findByRole("checkbox", { name: `Select customer ${activeCustomer.user_id}` });
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    await user.click(screen.getByRole("button", { name: "Edit selected" }));
    await selectToolset();
    await user.click(screen.getByRole("button", { name: "Apply changes" }));

    expect(await screen.findByRole("alert", { name: "Bulk update results" })).toHaveTextContent(
      "1 succeeded, 1 failed out of 2",
    );
    expect(screen.getByRole("region", { name: "Customers to update" })).toHaveTextContent(/^bob@example\.test$/);
    expect(network).toHaveBeenCalledTimes(2);
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(
      await screen.findByRole("checkbox", { name: `Select customer ${activeCustomer.user_id}` }),
    ).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: `Select customer ${blockedCustomer.user_id}` })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: `Select customer ${extraCustomer.user_id}` })).not.toBeChecked();
    expect(screen.getByText("1 selected")).toBeInTheDocument();
    expect(screen.getByText("Updated customer")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(network).toHaveBeenCalledTimes(3));
    await user.click(screen.getByRole("button", { name: "Edit selected" }));
    expect(screen.getByRole("region", { name: "Customers to update" })).toHaveTextContent(/^bob@example\.test$/);
    await selectToolset();
    await user.click(screen.getByRole("button", { name: "Apply changes" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    expect(post).toHaveBeenNthCalledWith(1, "/customer/bulk_update", {
      accessToken: "sk-test",
      body: {
        customers: customers.map((customer) => ({
          user_id: customer.user_id,
          object_permission: { mcp_toolsets: ["toolset-new"], mcp_tool_permissions: {} },
        })),
      },
    });
    expect(post).toHaveBeenNthCalledWith(2, "/customer/bulk_update", {
      accessToken: "sk-test",
      body: {
        customers: [
          {
            user_id: blockedCustomer.user_id,
            object_permission: { mcp_toolsets: ["toolset-new"], mcp_tool_permissions: {} },
          },
        ],
      },
    });
    expect(post).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("checkbox", { name: `Select customer ${blockedCustomer.user_id}` })).not.toBeChecked();
    expect(screen.queryByRole("button", { name: "Edit selected" })).not.toBeInTheDocument();
  });

  it.each([
    { userRole: "Admin", isViewOnly: true },
    { userRole: "Admin Viewer", isViewOnly: false },
  ])("hides bulk editing for $userRole with view-only $isViewOnly", ({ userRole, isViewOnly }) => {
    authorization.userRole = userRole;
    authorization.isViewOnly = isViewOnly;
    renderWithProviders(<CustomersPanel />);
    expect(screen.queryByRole("checkbox", { name: "Select all rows" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit selected" })).not.toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });

  it("blocks selected customers in one batch and clears selection after success", async () => {
    authorization.premiumUser = true;
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />);
    expect(screen.queryByRole("button", { name: "Block selected" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    expect(screen.getByText("2 selected")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Block selected" }));
    await confirmBlocking(user, "Block", 2);
    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/block", {
        accessToken: "sk-test",
        body: { user_ids: customers.map((customer) => customer.user_id) },
      }),
    );
    await waitFor(() => expect(screen.queryByRole("button", { name: "Block selected" })).not.toBeInTheDocument());
    expect(screen.getByRole("checkbox", { name: `Select customer ${activeCustomer.user_id}` })).not.toBeChecked();
  });

  it.each(["Block", "Unblock"])("gates bulk %s with the Enterprise explanation", async (action) => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />);
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    const button = screen.getByRole("button", { name: `${action} selected` });
    expect(button).toHaveAttribute("aria-disabled", "true");
    await user.click(screen.getByRole("textbox", { name: "Search customers" }));
    await user.tab();
    await user.tab();
    if (action === "Unblock") await user.tab();
    expect(button).toHaveFocus();
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "This is a LiteLLM Enterprise feature, and requires a valid key to use.",
    );
    expect(button).toHaveAccessibleDescription(
      "This is a LiteLLM Enterprise feature, and requires a valid key to use.",
    );
    await user.keyboard("{Enter}");
    expect(post).not.toHaveBeenCalled();
  });

  it("unblocks only selected IDs in one batch with Enterprise", async () => {
    authorization.premiumUser = true;
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />);
    await user.click(screen.getByRole("checkbox", { name: `Select customer ${blockedCustomer.user_id}` }));
    await user.click(screen.getByRole("button", { name: "Unblock selected" }));
    await confirmBlocking(user, "Unblock", 1);
    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/unblock", {
        accessToken: "sk-test",
        body: { user_ids: [blockedCustomer.user_id] },
      }),
    );
  });

  it("keeps selection and blocked state when bulk Unblock has no configured callback", async () => {
    authorization.premiumUser = true;
    const user = userEvent.setup();
    const message = "Blocked user check was never set";
    post.mockRejectedValue(new ApiError(message, 400, { detail: { error: message } }));
    renderWithProviders(
      <>
        <CustomersPanel />
        <Toaster />
      </>,
    );
    await user.click(screen.getByRole("checkbox", { name: `Select customer ${blockedCustomer.user_id}` }));
    await user.click(screen.getByRole("button", { name: "Unblock selected" }));
    await confirmBlocking(user, "Unblock", 1);
    expect(await screen.findByText(message)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.getByRole("checkbox", { name: `Select customer ${blockedCustomer.user_id}` })).toBeChecked();
    expect(within(screen.getByRole("row", { name: /bob@example.test/ })).getByText("Blocked")).toBeInTheDocument();
    expect(screen.queryByText("1 customer unblocked")).not.toBeInTheDocument();
    expect(post).toHaveBeenCalledExactlyOnceWith("/customer/unblock", {
      accessToken: "sk-test",
      body: { user_ids: [blockedCustomer.user_id] },
    });
  });

  it("uses the shared money display for zero and small spend", () => {
    useQuery.mockReturnValue({
      data: [{ ...activeCustomer, spend: 0.001 }, blockedCustomer],
      isLoading: false,
      isError: false,
    });
    renderWithProviders(<CustomersPanel />);
    expect(within(screen.getByRole("row", { name: /alice@example.test/ })).getByText("< $0.01")).toBeInTheDocument();
    expect(within(screen.getByRole("row", { name: /bob@example.test/ })).getByText("$0.00")).toBeInTheDocument();
  });

  it.each([
    { maxBudget: 0, budgetLabel: "$0.00" },
    { maxBudget: null, budgetLabel: "Unlimited" },
  ])(
    "distinguishes a $budgetLabel budget from zero spend in the list and details",
    async ({ maxBudget, budgetLabel }) => {
      const user = userEvent.setup();
      const customer: EndUser = {
        ...activeCustomer,
        spend: 0,
        litellm_budget_table: {
          max_budget: maxBudget,
          tpm_limit: 600,
          rpm_limit: 6,
          created_at: "2026-01-01T00:00:00Z",
        },
      };
      useQuery.mockImplementation((_method: string, path: string) => ({
        data: path === "/customer/list" ? [customer] : customer,
        isLoading: false,
        isError: false,
      }));
      renderWithProviders(<CustomersPanel />);

      const row = screen.getByRole("row", { name: /alice@example.test/ });
      expect(within(row).getByText(maxBudget === null ? "· Unlimited" : `of ${budgetLabel}`)).toBeInTheDocument();
      await user.click(within(row).getByRole("button", { name: /Alice.*alice@example.test/ }));
      expect(await screen.findByText("Budget Status")).toBeInTheDocument();
      expect(screen.getAllByText("$0.00")).toHaveLength(maxBudget === 0 ? 2 : 1);
      expect(screen.getAllByText(budgetLabel)).toHaveLength(maxBudget === 0 ? 2 : 1);
      await user.click(screen.getByRole("tab", { name: "Settings" }));
      expect(await screen.findByText("Max Budget")).toBeInTheDocument();
      expect(screen.getByText(budgetLabel)).toBeInTheDocument();
    },
  );

  it("keeps model and MCP details optional while showing each customer status", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />);
    expect(screen.queryByRole("columnheader", { name: "MCP Access" })).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Models" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Columns" }));
    await user.click(await screen.findByRole("menuitemcheckbox", { name: "MCP Access" }));
    await user.click(await screen.findByRole("menuitemcheckbox", { name: "Models" }));
    await user.keyboard("{Escape}");

    const alice = screen.getByRole("row", { name: /alice@example.test/ });
    expect(within(alice).getByText("Active")).toBeInTheDocument();
    expect(within(alice).getByText("All models the key can use")).toBeInTheDocument();
    expect(within(alice).getByText("1 group, 1 toolset")).toBeInTheDocument();
    const bob = screen.getByRole("row", { name: /bob@example.test/ });
    expect(within(bob).getByText("Blocked")).toBeInTheDocument();
    expect(within(bob).getByText("No restriction")).toBeInTheDocument();
  });

  it.each([
    { surface: "row", action: "Block", customer: activeCustomer },
    { surface: "detail", action: "Block", customer: activeCustomer },
    { surface: "row", action: "Unblock", customer: blockedCustomer },
    { surface: "detail", action: "Unblock", customer: blockedCustomer },
  ])("disables $action without Enterprise in the $surface menu", async ({ surface, action, customer }) => {
    const user = userEvent.setup();
    renderWithProviders(
      <CustomersPanel />,
      surface === "detail" ? { searchParams: `customer=${encodeURIComponent(customer.user_id)}` } : undefined,
    );
    await user.click(
      screen.getByRole("button", {
        name: surface === "detail" ? "More customer actions" : `Actions for ${customer.user_id}`,
      }),
    );
    const menuAction = await screen.findByRole("menuitem", { name: action });
    expect(menuAction).toHaveAttribute("aria-disabled", "true");
    await user.keyboard("{Home}");
    if (surface === "row") await user.keyboard("{ArrowDown}");
    expect(menuAction).toHaveFocus();
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "This is a LiteLLM Enterprise feature, and requires a valid key to use.",
    );
    expect(menuAction).toHaveAccessibleDescription(
      "This is a LiteLLM Enterprise feature, and requires a valid key to use.",
    );
    await user.keyboard("{Enter}");
    expect(post).not.toHaveBeenCalled();
  });

  it.each([
    { surface: "row", action: "Block", endpoint: "/customer/block", customer: activeCustomer },
    { surface: "detail", action: "Block", endpoint: "/customer/block", customer: activeCustomer },
    { surface: "row", action: "Unblock", endpoint: "/customer/unblock", customer: blockedCustomer },
    { surface: "detail", action: "Unblock", endpoint: "/customer/unblock", customer: blockedCustomer },
  ])("uses $endpoint with Enterprise in the $surface menu", async ({ surface, action, endpoint, customer }) => {
    authorization.premiumUser = true;
    const user = userEvent.setup();
    renderWithProviders(
      <CustomersPanel />,
      surface === "detail" ? { searchParams: `customer=${encodeURIComponent(customer.user_id)}` } : undefined,
    );
    expect(screen.queryByRole("button", { name: action })).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("button", {
        name: surface === "detail" ? "More customer actions" : `Actions for ${customer.user_id}`,
      }),
    );
    const menuAction = await screen.findByRole("menuitem", { name: action });
    expect(menuAction).not.toHaveAttribute("aria-disabled", "true");
    await user.click(menuAction);
    await confirmBlocking(user, action, 1);
    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith(endpoint, {
        accessToken: "sk-test",
        body: { user_ids: [customer.user_id] },
      }),
    );
  });

  it.each(["row", "detail", "bulk"])("allows cancelling a block from the %s surface", async (surface) => {
    authorization.premiumUser = true;
    const user = userEvent.setup();
    renderWithProviders(
      <CustomersPanel />,
      surface === "detail" ? { searchParams: "customer=alice%40example.test" } : undefined,
    );
    if (surface === "bulk") {
      await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
      await user.click(screen.getByRole("button", { name: "Block selected" }));
    } else {
      await user.click(
        screen.getByRole("button", {
          name: surface === "detail" ? "More customer actions" : `Actions for ${activeCustomer.user_id}`,
        }),
      );
      await user.click(await screen.findByRole("menuitem", { name: "Block" }));
    }
    const count = surface === "bulk" ? 2 : 1;
    const dialog = await screen.findByRole("dialog", { name: `Block ${count} customer${count === 1 ? "" : "s"}?` });
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(post).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it.each([null, activeCustomer.user_id])("shows the customer ID once when alias is %s", (alias) => {
    useQuery.mockImplementation((_method: string, path: string) => ({
      data: path === "/customer/list" ? [{ ...activeCustomer, alias }] : { ...activeCustomer, alias },
      isLoading: false,
      isError: false,
    }));
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test" });
    expect(screen.getAllByText(activeCustomer.user_id)).toHaveLength(1);
    expect(screen.getByRole("heading", { name: /alice@example.test/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy Customer ID" })).toBeInTheDocument();
  });

  it.each(["Alice", "alice@example.test"])("filters by ID or alias using %s", async (search) => {
    renderWithProviders(<CustomersPanel />);
    fireEvent.change(screen.getByRole("textbox", { name: "Search customers" }), { target: { value: search } });

    await waitFor(() => expect(screen.queryByRole("row", { name: /bob@example.test/ })).not.toBeInTheDocument());
    expect(screen.getByRole("row", { name: /alice@example.test/ })).toBeInTheDocument();
  });

  it("creates a customer with the filled form values", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />);
    await user.click(screen.getByRole("button", { name: "Create Customer" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("Customer ID"), { target: { value: "  new@example.test  " } });
    fireEvent.change(within(dialog).getByLabelText("Alias"), { target: { value: "New Customer" } });
    fireEvent.change(within(dialog).getByLabelText("Max Budget (USD)"), { target: { value: "25" } });
    await user.click(within(dialog).getByRole("button", { name: "Create Customer" }));

    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/new", {
        accessToken: "sk-test",
        body: { user_id: "new@example.test", alias: "New Customer", max_budget: 25 },
      }),
    );
  });

  it("requires a customer ID before creating", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />);
    await user.click(screen.getByRole("button", { name: "Create Customer" }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Create Customer" }));

    expect(await within(dialog).findByText("Customer ID is required")).toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });

  it("edits a deep-linked customer without rewriting MCP permissions", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test", onUrlUpdate });
    expect(screen.getByRole("heading", { name: /^Alice/ })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    await user.click(await screen.findByRole("button", { name: "Edit Settings" }));
    const alias = await screen.findByLabelText("Alias");
    expect(screen.getByLabelText("Customer ID")).toBeDisabled();
    fireEvent.change(alias, { target: { value: "Updated Alice" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/update", {
        accessToken: "sk-test",
        body: { user_id: activeCustomer.user_id, alias: "Updated Alice", models: [] },
      }),
    );
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          searchParams: expect.any(URLSearchParams),
        }),
      ),
    );
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("edit")).toBeNull();
  });

  it("preserves an unrestricted server when editing only the alias after tools load", async () => {
    const user = userEvent.setup();
    const directCustomer: EndUser = {
      ...activeCustomer,
      object_permission: {
        ...activeCustomer.object_permission!,
        mcp_servers: ["server-1"],
        mcp_access_groups: [],
        mcp_toolsets: [],
      },
    };
    useQuery.mockImplementation((_method: string, path: string) => ({
      data: path === "/customer/list" ? [directCustomer] : directCustomer,
      isLoading: false,
      isError: false,
    }));
    fetchServers.mockResolvedValue([
      {
        server_id: "server-1",
        server_name: "Research Server",
        created_at: "",
        created_by: "admin",
        updated_at: "",
        updated_by: "admin",
      },
    ]);
    listTools.mockResolvedValue({
      error: false,
      tools: [
        { name: "read_document", description: "Read a document", inputSchema: {} },
        { name: "delete_document", description: "Delete a document", inputSchema: {} },
      ],
    });
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test&edit=true" });
    await screen.findByRole("radio", { name: "Flat List" });
    fireEvent.change(screen.getByLabelText("Alias"), { target: { value: "Updated Alice" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/update", {
        accessToken: "sk-test",
        body: { user_id: activeCustomer.user_id, alias: "Updated Alice", models: [] },
      }),
    );
  });

  it("shows a catalog failure when removing an MCP grant and keeps the customer unsaved", async () => {
    const user = userEvent.setup();
    const directCustomer: EndUser = {
      ...activeCustomer,
      object_permission: {
        object_permission_id: "permission-alice",
        mcp_servers: ["server-1"],
        mcp_access_groups: [],
        mcp_toolsets: [],
        mcp_tool_permissions: { "server-1": ["read_document"] },
        agent_access_groups: [],
        agents: [],
        blocked_tools: [],
        models: [],
        search_tools: [],
        skills: [],
        vector_stores: [],
      },
    };
    useQuery.mockImplementation((_method: string, path: string) => ({
      data: path === "/customer/list" ? [directCustomer] : directCustomer,
      isLoading: false,
      isError: false,
    }));
    fetchServers.mockRejectedValue(new Error("Catalog unavailable"));
    renderWithProviders(
      <>
        <CustomersPanel />
        <Toaster />
      </>,
      { searchParams: "customer=alice%40example.test&edit=true" },
    );

    await user.click(await screen.findByRole("button", { name: "Clear all" }));
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(
      await screen.findByText("MCP catalogs must finish loading before permission changes can be saved"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save Changes" })).toBeInTheDocument();
    expect(screen.queryByText("Customer updated")).not.toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });

  it("rejects an edit budget of zero because customer updates cannot apply it", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test&edit=true" });
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "0" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByText("Enter a budget greater than zero when editing")).toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });

  it("confirms deletion and clears the open customer from the URL", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test", onUrlUpdate });
    await user.click(screen.getByRole("button", { name: "More customer actions" }));
    await user.click(await screen.findByRole("menuitem", { name: "Delete Customer" }));
    const dialog = await screen.findByRole("dialog", { name: "Delete Customer?" });
    expect(within(dialog).getByText(/recreates the customer with no restrictions/)).toBeInTheDocument();
    const confirm = within(dialog).getByRole("button", { name: "Delete" });
    expect(confirm).toBeDisabled();
    fireEvent.change(within(dialog).getByPlaceholderText(activeCustomer.user_id), {
      target: { value: activeCustomer.user_id },
    });
    await user.click(confirm);

    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/delete", {
        accessToken: "sk-test",
        body: { user_ids: [activeCustomer.user_id] },
      }),
    );
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalledWith(expect.objectContaining({ queryString: "" })));
    expect(screen.getByRole("heading", { name: "Customers" })).toBeInTheDocument();
  });

  it("keeps Admin Viewer read-only, including an edit deep link", async () => {
    authorization.isViewOnly = true;
    const user = userEvent.setup();
    const list = renderWithProviders(<CustomersPanel />);
    expect(screen.queryByRole("checkbox", { name: "Select all rows" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Customer" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Actions for alice@example.test" }));
    expect(await screen.findByRole("menuitem", { name: "Copy ID" })).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Edit" })).not.toBeInTheDocument();
    list.unmount();
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test&edit=true" });
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    expect(screen.queryByRole("button", { name: "Edit Settings" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });
  it("uses the shared spend/budget cell and hides model details by default", () => {
    const customer: EndUser = {
      ...activeCustomer,
      budget_id: "shared-budget",
      litellm_budget_table: {
        budget_id: "shared-budget",
        max_budget: 100,
        tpm_limit: 600,
        rpm_limit: 6,
        created_at: "2026-01-01T00:00:00Z",
      },
    };
    useQuery.mockReturnValue({ data: [customer], isLoading: false, isError: false });
    renderWithProviders(<CustomersPanel />);

    expect(screen.getByText("Manage customer spend, budgets, rate limits, model and MCP access")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Spend / Budget" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Models" })).not.toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "TPM / RPM Limits" })).toBeInTheDocument();
    const row = screen.getByRole("row", { name: /alice@example.test/ });
    expect(within(row).getByText("$12.50")).toBeInTheDocument();
    expect(within(row).getByText("of $100.00")).toBeInTheDocument();
    expect(within(row).getByText("TPM: 600")).toBeInTheDocument();
    expect(within(row).getByText("RPM: 6")).toBeInTheDocument();
    expect(within(row).queryByText("All models the key can use")).not.toBeInTheDocument();
  });

  it("shows Overview cards and shared permissions before entering Settings", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    const customer: EndUser = {
      ...activeCustomer,
      object_permission: {
        ...activeCustomer.object_permission!,
        vector_stores: ["vector-1"],
        search_tools: ["search-1"],
        skills: ["skill-1"],
      },
      allowed_model_region: "eu",
      default_model: "default-model",
      litellm_budget_table: {
        max_budget: 100,
        budget_duration: "30d",
        tpm_limit: 600,
        rpm_limit: 6,
        created_at: "2026-01-01T00:00:00Z",
      },
    };
    useQuery.mockImplementation((_method: string, path: string) => ({
      data: path === "/customer/list" ? [customer] : customer,
      isLoading: false,
      isError: false,
    }));
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test", onUrlUpdate });

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Budget Status")).toBeInTheDocument();
    expect(screen.getByText("$100.00")).toBeInTheDocument();
    expect(screen.getByText("Reset: monthly")).toBeInTheDocument();
    expect(screen.getByText("Rate Limits")).toBeInTheDocument();
    expect(screen.getByText("MCP Servers")).toBeInTheDocument();
    for (const unenforced of ["Vector Stores", "Search tools", "Skills", "vector-1", "search-1", "skill-1"]) {
      expect(screen.queryByText(unenforced)).not.toBeInTheDocument();
    }
    expect(screen.getByText("All models the key can use")).toBeInTheDocument();
    expect(screen.getByText("EU")).toBeInTheDocument();
    expect(screen.getByText("default-model")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit Settings" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    expect(await screen.findByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("customer_tab")).toBe("settings");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("customer")).toBe(activeCustomer.user_id);
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].options.history).toBe("push");
  });

  it("opens a Settings deep link without enabling the edit form", () => {
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test&customer_tab=settings" });
    expect(screen.getByRole("tab", { name: "Settings" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
  });

  it("updates a customer's own maximum without replacing its budget link", async () => {
    const user = userEvent.setup();
    const budget = { budget_id: "own-budget", max_budget: 100, created_at: "2026-01-01T00:00:00Z" };
    const customer: EndUser = { ...activeCustomer, budget_id: budget.budget_id, litellm_budget_table: budget };
    get.mockResolvedValue([budget]);
    useQuery.mockImplementation((_method: string, path: string) => ({
      data: path === "/customer/list" ? [customer, blockedCustomer] : customer,
      isLoading: false,
      isError: false,
    }));
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test&edit=true" });
    expect(screen.getByLabelText("Budget")).toHaveTextContent("Keep current budget");
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "25" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/update", {
        accessToken: "sk-test",
        body: { user_id: customer.user_id, alias: "Alice", models: [], max_budget: 25 },
      }),
    );
  });

  it.each(["loading", "error"])("keeps maximum edits guarded while the customer list is %s", async (status) => {
    const user = userEvent.setup();
    const budget = { budget_id: "budget-id", max_budget: 100, created_at: "2026-01-01T00:00:00Z" };
    const customer: EndUser = { ...activeCustomer, budget_id: budget.budget_id, litellm_budget_table: budget };
    useQuery.mockImplementation((_method: string, path: string) => ({
      data: path === "/customer/list" ? undefined : customer,
      isLoading: path === "/customer/list" && status === "loading",
      isError: path === "/customer/list" && status === "error",
    }));
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test&edit=true" });
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "25" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(
      await screen.findByText("Unable to verify budget sharing. Reload customers before changing Max Budget"),
    ).toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });

  it("rejects a max budget when keeping a shared budget link", async () => {
    const user = userEvent.setup();
    const budget = { budget_id: "shared-budget", max_budget: 100, created_at: "2026-01-01T00:00:00Z" };
    const linkedCustomers: EndUser[] = customers.map((customer) => ({
      ...customer,
      budget_id: budget.budget_id,
      litellm_budget_table: budget,
    }));
    get.mockResolvedValue([budget]);
    useQuery.mockImplementation((_method: string, path: string) => ({
      data: path === "/customer/list" ? linkedCustomers : linkedCustomers[0],
      isLoading: false,
      isError: false,
    }));
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test&edit=true" });
    expect(screen.getByRole("tab", { name: "Settings" })).toHaveAttribute("aria-selected", "true");
    await chooseSelectOption(user, screen.getByLabelText("Budget"), "Keep current budget");
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "25" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(
      await screen.findByText("This customer has a shared budget. Leave Max Budget empty or choose another budget"),
    ).toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/update", {
        accessToken: "sk-test",
        body: { user_id: activeCustomer.user_id, alias: "Alice", models: [] },
      }),
    );
  });

  it.each(["row", "detail"])(
    "shows an Unblock callback error and keeps the customer blocked in the %s menu",
    async (surface) => {
      authorization.premiumUser = true;
      const user = userEvent.setup();
      const message = "Blocked user check was never set";
      post.mockRejectedValue(new ApiError(message, 400, { detail: { error: message } }));
      renderWithProviders(
        <>
          <CustomersPanel />
          <Toaster />
        </>,
        surface === "detail" ? { searchParams: "customer=bob%40example.test" } : undefined,
      );
      await user.click(
        screen.getByRole("button", {
          name: surface === "detail" ? "More customer actions" : `Actions for ${blockedCustomer.user_id}`,
        }),
      );
      await user.click(await screen.findByRole("menuitem", { name: "Unblock" }));
      await confirmBlocking(user, "Unblock", 1);

      expect(await screen.findByText(message)).toBeInTheDocument();
      expect(screen.getByText("Request Error")).toBeInTheDocument();
      expect(screen.queryByText("Customer unblocked")).not.toBeInTheDocument();
      expect(screen.getByText("Blocked")).toBeInTheDocument();
      expect(post).toHaveBeenCalledExactlyOnceWith("/customer/unblock", {
        accessToken: "sk-test",
        body: { user_ids: [blockedCustomer.user_id] },
      });
    },
  );
});

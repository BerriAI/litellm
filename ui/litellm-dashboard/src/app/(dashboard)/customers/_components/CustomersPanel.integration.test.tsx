import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Toaster, toast as sonner } from "sonner";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/http/client";

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
    expect(screen.getByRole("button", { name: "Block selected" })).toBeDisabled();
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
            object_permission: { mcp_toolsets: ["toolset-new"] },
          })),
        },
      }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("checkbox", { name: "Select customer customer-25" })).not.toBeChecked();
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
    expect(screen.getByRole("button", { name: "Block selected" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Unblock selected" })).toBeDisabled();
    await user.hover(screen.getByLabelText(`Bulk ${action} requires Enterprise`));
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "This is a LiteLLM Enterprise feature, and requires a valid key to use.",
    );
    await user.click(screen.getByLabelText(`Bulk ${action} requires Enterprise`));
    expect(post).not.toHaveBeenCalled();
  });

  it("unblocks only selected IDs in one batch with Enterprise", async () => {
    authorization.premiumUser = true;
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />);
    await user.click(screen.getByRole("checkbox", { name: `Select customer ${blockedCustomer.user_id}` }));
    await user.click(screen.getByRole("button", { name: "Unblock selected" }));
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
    expect(await screen.findByText(message)).toBeInTheDocument();
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
    expect(
      within(screen.getByRole("row", { name: /bob@example.test/ })).queryByText(/\$0\.00/),
    ).not.toBeInTheDocument();
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
      expect(within(row).getByText(budgetLabel)).toBeInTheDocument();
      await user.click(within(row).getByRole("button", { name: /Alice.*alice@example.test/ }));
      expect(await screen.findByText("Budget Status")).toBeInTheDocument();
      expect(screen.getAllByText("$0.00")).toHaveLength(maxBudget === 0 ? 2 : 1);
      expect(screen.getAllByText(budgetLabel)).toHaveLength(maxBudget === 0 ? 2 : 1);
      await user.click(screen.getByRole("tab", { name: "Settings" }));
      expect(await screen.findByText("Max Budget")).toBeInTheDocument();
      expect(screen.getByText(budgetLabel)).toBeInTheDocument();
    },
  );

  it("keeps MCP access optional while showing each customer status", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPanel />);
    expect(screen.queryByRole("columnheader", { name: "MCP Access" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Columns" }));
    await user.click(await screen.findByRole("menuitemcheckbox", { name: "MCP Access" }));
    await user.keyboard("{Escape}");

    const alice = screen.getByRole("row", { name: /alice@example.test/ });
    expect(within(alice).getByText("Active")).toBeInTheDocument();
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
    expect(await screen.findByRole("menuitem", { name: action })).toHaveAttribute("aria-disabled", "true");
    await user.hover(screen.getByLabelText(`${action} requires Enterprise`));
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "This is a LiteLLM Enterprise feature, and requires a valid key to use.",
    );
    await user.click(screen.getByLabelText(`${action} requires Enterprise`));
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
    await waitFor(() =>
      expect(post).toHaveBeenCalledExactlyOnceWith(endpoint, {
        accessToken: "sk-test",
        body: { user_ids: [customer.user_id] },
      }),
    );
  });

  it.each([null, activeCustomer.user_id])("shows the customer ID once when alias is %s", (alias) => {
    useQuery.mockReturnValue({ data: { ...activeCustomer, alias }, isLoading: false, isError: false });
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
    useQuery.mockReturnValue({ data: directCustomer, isLoading: false, isError: false });
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
    useQuery.mockReturnValue({ data: directCustomer, isLoading: false, isError: false });
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
  it("matches the Organizations money, model and rate-limit columns", () => {
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

    expect(screen.getByText("Manage end-user spend, budgets, rate limits, model and MCP access")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Spend (USD)" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Budget (USD)" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "TPM / RPM Limits" })).toBeInTheDocument();
    const row = screen.getByRole("row", { name: /alice@example.test/ });
    expect(within(row).getByText("$12.50")).toBeInTheDocument();
    expect(within(row).getByText("$100.00")).toBeInTheDocument();
    expect(within(row).getByText("TPM: 600")).toBeInTheDocument();
    expect(within(row).getByText("RPM: 6")).toBeInTheDocument();
    expect(within(row).getByText("All Proxy Models")).toBeInTheDocument();
  });

  it("shows Overview cards and shared permissions before entering Settings", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    const customer: EndUser = {
      ...activeCustomer,
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
    useQuery.mockReturnValue({ data: customer, isLoading: false, isError: false });
    renderWithProviders(<CustomersPanel />, { searchParams: "customer=alice%40example.test", onUrlUpdate });

    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Budget Status")).toBeInTheDocument();
    expect(screen.getByText("$100.00")).toBeInTheDocument();
    expect(screen.getByText("Reset: monthly")).toBeInTheDocument();
    expect(screen.getByText("Rate Limits")).toBeInTheDocument();
    expect(screen.getByText("Object Permissions")).toBeInTheDocument();
    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
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
      await screen.findByText("This customer has a linked budget. Leave Max Budget empty or choose another budget"),
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

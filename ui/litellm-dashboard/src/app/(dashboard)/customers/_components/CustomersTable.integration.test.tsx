import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Papa from "papaparse";
import { afterEach, describe, expect, it, vi } from "vitest";

import { toast } from "@/lib/toast";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { chooseSelectOption, renderWithProviders } from "../../../../../tests/test-utils";
import CustomersTable from "./CustomersTable";

const customer: EndUser = {
  user_id: "customer-01",
  alias: "Research",
  blocked: false,
  spend: 0,
  models: ["model-1", "model-2"],
  litellm_budget_table: { max_budget: 0, created_at: "2026-01-01T00:00:00Z" },
  object_permission: {
    object_permission_id: "permission-1",
    mcp_servers: ["server-1"],
    mcp_access_groups: ["group-1"],
    mcp_toolsets: ["toolset-1"],
    mcp_tool_permissions: {},
    agent_access_groups: [],
    agents: [],
    models: [],
    blocked_tools: [],
    search_tools: [],
    vector_stores: [],
  },
};
const props = {
  data: [customer],
  isLoading: false,
  isError: false,
  isRefreshing: false,
  isBlocking: false,
  canModify: true,
  premiumUser: true,
  onRefresh: vi.fn(),
  onView: vi.fn(),
  onEdit: vi.fn(),
  onDelete: vi.fn(),
  onSetBlocked: vi.fn(),
  onBulkSetBlocked: vi.fn().mockResolvedValue(true),
};

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("CustomersTable controls", () => {
  it("combines all three filters, writes URL state, and resets them", async () => {
    const rows: EndUser[] = [
      customer,
      { ...customer, user_id: "customer-02", alias: "Blocked", blocked: true },
      { ...customer, user_id: "customer-03", alias: "Other", object_permission: null, litellm_budget_table: null },
    ];
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomersTable {...props} data={rows} />, { onUrlUpdate });
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    await user.click(screen.getByRole("button", { name: "Filters" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Status" }), "Active");
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Budget" }), "Has budget");
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "MCP access" }), "Has MCP access");
    await user.click(screen.getByRole("button", { name: "Apply Filters" }));

    await waitFor(() => expect(screen.queryByRole("row", { name: /customer-02/ })).not.toBeInTheDocument());
    expect(screen.queryByRole("row", { name: /customer-03/ })).not.toBeInTheDocument();
    expect(screen.getByRole("row", { name: /customer-01/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Unblock selected" })).not.toBeInTheDocument();
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const params = onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;
    expect(params.get("filter_blocked")).toBe("active");
    expect(params.get("filter_max_budget")).toBe("yes");
    expect(params.get("filter_mcp_access")).toBe("yes");
    await user.click(screen.getByRole("button", { name: /^Filters/ }));
    await user.click(screen.getByRole("button", { name: "Reset" }));
    await user.click(screen.getByRole("button", { name: "Apply Filters" }));
    expect(await screen.findByRole("row", { name: /customer-02/ })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /customer-03/ })).toBeInTheDocument();
  });

  it("restores negative filters from the URL and clears selection on search changes", async () => {
    const blocked: EndUser = {
      ...customer,
      user_id: "customer-02",
      blocked: true,
      object_permission: null,
      litellm_budget_table: null,
    };
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomersTable {...props} data={[customer, blocked]} />, {
      searchParams: "filter_blocked=blocked&filter_max_budget=no&filter_mcp_access=no",
      onUrlUpdate,
    });
    expect(screen.getByRole("row", { name: /customer-02/ })).toBeInTheDocument();
    expect(screen.queryByRole("row", { name: /customer-01/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Search customers" }), { target: { value: "missing" } });
    await waitFor(() => expect(screen.queryByRole("row", { name: /customer-02/ })).not.toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Unblock selected" })).not.toBeInTheDocument();
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenLastCalledWith(
        expect.objectContaining({
          searchParams: expect.any(URLSearchParams),
          queryString: expect.stringContaining("search=missing"),
        }),
      ),
    );
    fireEvent.change(screen.getByRole("textbox", { name: "Search customers" }), { target: { value: "" } });
    expect(await screen.findByRole("checkbox", { name: "Select customer customer-02" })).not.toBeChecked();
  });

  it("selects the current page and retains IDs selected on other pages", async () => {
    const rows = Array.from(
      { length: 26 },
      (_, index): EndUser => ({
        ...customer,
        user_id: `customer-${String(index).padStart(2, "0")}`,
        alias: null,
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(<CustomersTable {...props} data={rows} />, { searchParams: "page_size=25" });
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    expect(screen.getByText("25 selected")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Go to next page" }));
    expect(await screen.findByRole("checkbox", { name: "Select customer customer-25" })).not.toBeChecked();
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    expect(screen.getByText("26 selected")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Unblock selected" }));
    const dialog = await screen.findByRole("dialog", { name: "Unblock 26 customers?" });
    expect(props.onBulkSetBlocked).not.toHaveBeenCalled();
    await user.click(within(dialog).getByRole("button", { name: "Unblock" }));
    expect(props.onBulkSetBlocked).toHaveBeenCalledExactlyOnceWith(
      rows.map((row) => row.user_id),
      false,
    );
    await waitFor(() => expect(screen.queryByRole("button", { name: "Unblock selected" })).not.toBeInTheDocument());
  });

  it.each([
    { blocked: false, action: "Block" },
    { blocked: true, action: "Unblock" },
  ])("disables selection, row $action, and batch controls during a block mutation", async ({ blocked, action }) => {
    const data = [{ ...customer, blocked }];
    const user = userEvent.setup();
    const view = renderWithProviders(<CustomersTable {...props} data={data} />);
    await user.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    view.rerender(<CustomersTable {...props} data={data} isBlocking />);
    expect(screen.getByRole("button", { name: "Block selected" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Unblock selected" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Clear selection" })).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: "Select customer customer-01" }));
    expect(screen.getByRole("checkbox", { name: "Select customer customer-01" })).toBeChecked();
    await user.click(screen.getByRole("button", { name: "Actions for customer-01" }));
    expect(await screen.findByRole("menuitem", { name: action })).toHaveAttribute("aria-disabled", "true");
    expect(props.onBulkSetBlocked).not.toHaveBeenCalled();
    expect(props.onSetBlocked).not.toHaveBeenCalled();
  });

  it.each([
    { search: "matching", count: 26 },
    { search: "customer-00", count: 1 },
  ])("exports $count matching customers using search and filters", async ({ search, count }) => {
    const rows = Array.from(
      { length: 28 },
      (_, index): EndUser => ({
        ...customer,
        user_id: `customer-${String(index).padStart(2, "0")}`,
        alias: index === 27 ? "Other" : "Matching",
        blocked: index === 26,
      }),
    );
    const createObjectURL = vi.fn().mockReturnValue("blob:customers-export");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal(
      "URL",
      class extends URL {
        static createObjectURL = createObjectURL;
        static revokeObjectURL = revokeObjectURL;
      },
    );
    const download = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const user = userEvent.setup();
    renderWithProviders(<CustomersTable {...props} data={rows} />, {
      searchParams: `page_size=25&search=${search}&filter_blocked=active&filter_mcp_access=yes`,
    });
    expect(screen.getByText(`Showing 1-${Math.min(count, 25)} of ${count}`)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Export CSV" }));
    const blob = createObjectURL.mock.calls[0][0] as Blob;
    const csv = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(blob);
    });
    const parsed = Papa.parse<Record<string, string>>(csv, { header: true });
    expect(parsed.errors).toEqual([]);
    expect(parsed.data.map((row) => row["Customer ID"])).toEqual(rows.slice(0, count).map((row) => row.user_id));
    expect(download).toHaveBeenCalledOnce();
    expect(toast.success).toHaveBeenCalledWith(`Exported ${count} customer${count === 1 ? "" : "s"}`);
    expect(revokeObjectURL).toHaveBeenCalledExactlyOnceWith("blob:customers-export");
  });

  it("wires the Resources badges into the customer table", () => {
    renderWithProviders(<CustomersTable {...props} />);
    expect(screen.getByRole("columnheader", { name: "Resources" })).toBeInTheDocument();
    const row = screen.getByRole("row", { name: /customer-01/ });
    expect(within(row).getByLabelText("2 models")).toHaveTextContent("2");
    expect(within(row).getByLabelText("1 MCP server")).toHaveTextContent("1");
    expect(within(row).getByLabelText("1 MCP access group")).toHaveTextContent("1");
    expect(within(row).getByLabelText("1 MCP toolset")).toHaveTextContent("1");
  });
});

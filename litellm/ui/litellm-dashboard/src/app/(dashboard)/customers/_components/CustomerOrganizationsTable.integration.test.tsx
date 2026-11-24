import { afterEach, expect, it, vi } from "vitest";
import { renderWithProviders, screen, jsonResponse } from "../../../../../tests/test-utils";
import CustomerOrganizationsTable from "./CustomerOrganizationsTable";
import { customerOrganizationRows } from "./customerOrganizations";
import { mergeCustomerActivityPages } from "./customerUsageData";

afterEach(() => vi.unstubAllGlobals());

it("shows a custom-only organization cap without calling it uncapped", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse({ default_budgets: [] })),
  );
  renderWithProviders(
    <CustomerOrganizationsTable
      rows={customerOrganizationRows(
        [
          {
            customer_org_id: "custom",
            alias: "Custom",
            budget_groups_revision: "revision",
            default_budgets: [],
            spend: 0,
            member_count: 0,
            created_at: "2026-11-01T00:00:00Z",
            updated_at: "2026-11-01T00:00:00Z",
            created_by: "admin",
            updated_by: "admin",
            budget: { budget_limits: [{ budget_duration: "7d", max_budget: 10 }] },
          },
        ],
        mergeCustomerActivityPages([]),
      )}
      sorting={[]}
      onSort={vi.fn()}
      loading={false}
      searching={false}
      startDate="2026-11-01"
      endDate="2026-11-18"
      usageTotals={null}
      canManage
      onEdit={vi.fn()}
    />,
  );
  expect(screen.getByRole("cell", { name: "$10.00/7d" })).toBeInTheDocument();
  expect(screen.queryByText("No limit")).not.toBeInTheDocument();
});

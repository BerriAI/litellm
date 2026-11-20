import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  chooseSelectOption,
  fireEvent,
  renderWithProviders,
  screen,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import CustomerMemberSpend from "./CustomerMemberSpend";
import type { CustomerHierarchyUsage } from "./customerUsageApi";
import { mergeCustomerActivityPages } from "./customerUsageData";

const members = Array.from({ length: 26 }, (_, index) => ({
  id: `member-${index + 1}`,
  label: `Member ${index + 1}`,
  spend: index + 1,
  budgetLimit: null,
  budgetDuration: null,
  budgetLimits: [],
  budgetResetAt: null,
}));
const data: CustomerHierarchyUsage = {
  ...mergeCustomerActivityPages([]),
  organization: { ...members[0], id: "org-1", label: "Organization" },
  team: null,
  member: null,
  teams: [],
  members: [],
  api_keys: [],
  teamResults: [],
  memberResults: [],
  memberBudgets: members,
  monthlyOrganizationSpend: members.reduce((sum, member) => sum + member.spend, 0),
  monthlyMembers: members.map((member) => ({ ...member, tokens: 100, requests: 1 })),
  yearlyMembers: [],
};
const visibleMembers = () =>
  within(screen.getByRole("table", { name: "Customer member spend" }))
    .getAllByRole("button", { name: /^Member \d+$/ })
    .map((button) => button.textContent);
const names = (highest: number, count: number) =>
  Array.from({ length: count }, (_, index) => `Member ${highest - index}`);

describe("CustomerMemberSpend pagination", () => {
  it("shows custom-only caps without describing the customer as uncapped", () => {
    const custom = {
      ...members[0],
      budgetLimits: [
        { budget_duration: "7d", max_budget: 10 },
        { budget_duration: "48h", max_budget: 2 },
      ],
    };
    renderWithProviders(
      <CustomerMemberSpend
        data={{ ...data, memberBudgets: [custom], monthlyMembers: [{ ...custom, tokens: 100, requests: 1 }] }}
        onMemberClick={vi.fn()}
      />,
    );
    expect(screen.getByRole("cell", { name: "$10.00" })).toBeInTheDocument();
    expect(screen.queryByText("No limit")).not.toBeInTheDocument();
    expect(screen.getByText(/\(7d\)/)).toBeInTheDocument();
    expect(screen.getByText(/\(48h\)/)).toBeInTheDocument();
  });

  it("shows the five highest spenders by default and changes page size without losing customer filters", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerMemberSpend data={data} onMemberClick={vi.fn()} />, {
      searchParams: "?org=org-1&app=app-1&from=2026-11-01&to=2026-11-18",
      onUrlUpdate,
    });

    expect(visibleMembers()).toEqual(names(26, 5));
    await user.click(screen.getByRole("button", { name: "Go to next page" }));
    expect(visibleMembers()).toEqual(names(21, 5));
    await chooseSelectOption(user, screen.getByTestId("pagination-page-size"), "10");
    expect(visibleMembers()).toEqual(names(26, 10));
    expect(screen.getByRole("button", { name: "Go to previous page" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Go to next page" }));
    expect(visibleMembers()).toEqual(names(16, 10));
    await chooseSelectOption(user, screen.getByTestId("pagination-page-size"), "25");
    expect(visibleMembers()).toEqual(names(26, 25));
    await waitFor(() =>
      expect(onUrlUpdate).toHaveBeenLastCalledWith(
        expect.objectContaining({
          queryString: "?org=org-1&app=app-1&from=2026-11-01&to=2026-11-18&memberPageSize=25",
        }),
      ),
    );
  });

  it("restores the selected page size from the URL and searches all members from the first page", async () => {
    const user = userEvent.setup();
    const onMemberClick = vi.fn();
    renderWithProviders(<CustomerMemberSpend data={data} onMemberClick={onMemberClick} />, {
      searchParams: "?memberPageSize=10&memberPage=2",
    });

    expect(visibleMembers()).toEqual(names(16, 10));
    fireEvent.change(screen.getByRole("textbox", { name: "Search customer members" }), {
      target: { value: "Member 26" },
    });
    expect(visibleMembers()).toEqual(["Member 26"]);
    expect(screen.getByRole("button", { name: "Go to previous page" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Go to next page" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Member 26", exact: true }));
    expect(onMemberClick).toHaveBeenCalledWith("member-26");
  });

  it("falls back to five rows for an unsupported page size and clamps an out-of-range page", () => {
    renderWithProviders(<CustomerMemberSpend data={data} onMemberClick={vi.fn()} />, {
      searchParams: "?memberPageSize=0&memberPage=999",
    });

    expect(visibleMembers()).toEqual(names(1, 1));
    expect(screen.getByTestId("pagination-page-size")).toHaveTextContent("5");
    expect(screen.getByRole("button", { name: "Go to next page" })).toBeDisabled();
  });
});

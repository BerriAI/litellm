import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { CustomerBudgetStrip } from "./CustomerContext";

it("shows settled spend and remaining customer org budget without a customer member", () => {
  render(
    <CustomerBudgetStrip
      context={{
        organization: {
          customer_org_id: "customer-org",
          alias: "customer-alias",
          spend: 3,
          budget: { max_budget: 10 },
          member_count: 0,
          default_budgets: [],
          created_at: "",
          updated_at: "",
          created_by: "",
          updated_by: "",
        },
        team: null,
        member: null,
      }}
      error={null}
      loading={false}
      unavailable={false}
    />,
  );
  expect(screen.getByRole("region", { name: "Customer budgets" })).toHaveTextContent(
    "Customer org · customer-alias Spend $3.00 · Remaining $7.00",
  );
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("shows a loading state without inventing spend values", () => {
  render(<CustomerBudgetStrip error={null} loading unavailable={false} />);
  expect(screen.getByRole("region", { name: "Customer budgets" })).toHaveTextContent("Refreshing customer budgets");
  expect(screen.queryByText(/Spend \$/)).not.toBeInTheDocument();
});

it("shows the budget_exceeded code and message, not a generic request error", () => {
  render(
    <CustomerBudgetStrip
      error={{ status: 422, message: "budget_exceeded: customer member limit reached" }}
      loading={false}
      unavailable={false}
    />,
  );
  expect(screen.getByRole("alert")).toHaveTextContent("HTTP 422 · budget_exceeded: customer member limit reached");
});

it("makes a failed budget refresh visible", () => {
  render(<CustomerBudgetStrip error={null} loading={false} unavailable />);
  expect(screen.getByRole("alert")).toHaveTextContent("Could not refresh customer budgets");
});

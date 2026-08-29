import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { GET, PUT, DELETE } = vi.hoisted(() => ({ GET: vi.fn(), PUT: vi.fn(), DELETE: vi.fn() }));
vi.mock("@/lib/http/api", () => ({ fetchClient: { GET, PUT, DELETE } }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userRole: "Admin" }),
}));

import AccessGroupBudgetsPanel from "./AccessGroupBudgetsPanel";

const BUDGETED_GROUP = {
  access_group: "premium",
  model_names: ["premium-nano"],
  deployment_count: 1,
  spend: 1.25,
  budget: {
    budget_id: "budget-1",
    max_budget: 2.5,
    soft_budget: null,
    budget_duration: "30d",
    budget_reset_at: null,
  },
};

const FREE_GROUP = {
  access_group: "shared",
  model_names: ["shared-nano"],
  deployment_count: 2,
  spend: 0,
  budget: null,
};

const renderPanel = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <AccessGroupBudgetsPanel />
    </QueryClientProvider>,
  );
};

const openActions = async (accessGroup: string) => {
  await userEvent.click(await screen.findByTestId(`access-group-actions-${accessGroup}`));
};

describe("AccessGroupBudgetsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    GET.mockResolvedValue({ data: { access_groups: [BUDGETED_GROUP, FREE_GROUP] } });
    PUT.mockResolvedValue({ data: { access_group: "shared", spend: 0, budget: null } });
    DELETE.mockResolvedValue({ data: { access_group: "premium", budget_deleted: true, message: "ok" } });
  });

  it("lists each group with the spend drawn against its shared budget", async () => {
    renderPanel();

    expect(await screen.findByText("premium")).toBeInTheDocument();
    expect(screen.getByText("$1.2500")).toBeInTheDocument();
    expect(screen.getByText("of $2.50")).toBeInTheDocument();
    expect(screen.getByText("monthly")).toBeInTheDocument();
    expect(GET).toHaveBeenCalledWith("/access_group/list");
  });

  it("shows a group with no budget as unlimited and offers nothing to clear", async () => {
    renderPanel();

    expect(await screen.findByText("· Unlimited")).toBeInTheDocument();

    await openActions("shared");

    expect(await screen.findByText("Set budget")).toBeInTheDocument();
    expect(screen.getByTestId("access-group-action-clear-budget")).toHaveAttribute("aria-disabled", "true");
  });

  it("sends the filled fields to the group's budget route", async () => {
    renderPanel();
    await openActions("shared");
    await userEvent.click(await screen.findByText("Set budget"));

    fireEvent.change(await screen.findByLabelText(/Max Budget/), { target: { value: "12.5" } });
    await userEvent.click(screen.getByRole("button", { name: "Save Budget" }));

    await waitFor(() =>
      expect(PUT).toHaveBeenCalledWith("/access_group/{access_group}/budget", {
        params: { path: { access_group: "shared" } },
        body: { max_budget: 12.5 },
      }),
    );
  });

  it("pre-fills the modal from the budget the group already has", async () => {
    renderPanel();
    await openActions("premium");
    await userEvent.click(await screen.findByText("Edit budget"));

    expect(await screen.findByLabelText(/Max Budget/)).toHaveValue(2.5);
  });

  it("refuses to save a budget with every field blank", async () => {
    renderPanel();
    await openActions("shared");
    await userEvent.click(await screen.findByText("Set budget"));
    await userEvent.click(await screen.findByRole("button", { name: "Save Budget" }));

    expect(await screen.findByText(/Set at least one of max budget/)).toBeInTheDocument();
    expect(PUT).not.toHaveBeenCalled();
  });

  it("clears a budget only after the confirmation is accepted", async () => {
    renderPanel();
    await openActions("premium");
    await userEvent.click(await screen.findByRole("menuitem", { name: /clear budget/i }));

    expect(DELETE).not.toHaveBeenCalled();

    await userEvent.click(await screen.findByRole("button", { name: /^delete$/i }));

    await waitFor(() =>
      expect(DELETE).toHaveBeenCalledWith("/access_group/{access_group}/budget", {
        params: { path: { access_group: "premium" } },
      }),
    );
  });
});

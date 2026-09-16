import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { fireEvent, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";

const { GET, PUT, DELETE, userRole } = vi.hoisted(() => ({
  GET: vi.fn(),
  PUT: vi.fn(),
  DELETE: vi.fn(),
  userRole: { current: "Admin" },
}));
vi.mock("@/lib/http/api", () => ({ fetchClient: { GET, PUT, DELETE } }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userRole: userRole.current }),
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

const renderPanel = (url: { searchParams?: string; onUrlUpdate?: OnUrlUpdateFunction } = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return renderWithProviders(
    <QueryClientProvider client={queryClient}>
      <AccessGroupBudgetsPanel />
    </QueryClientProvider>,
    url,
  );
};

const groupNamesInOrder = () =>
  screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell")[0].textContent);

const openActions = async (accessGroup: string) => {
  await userEvent.click(await screen.findByTestId(`access-group-actions-${accessGroup}`));
};

describe("AccessGroupBudgetsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    userRole.current = "Admin";
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

  it("keeps a sub-cent budget readable instead of rounding it away to $0.00", async () => {
    GET.mockResolvedValue({
      data: { access_groups: [{ ...BUDGETED_GROUP, budget: { ...BUDGETED_GROUP.budget, max_budget: 0.00002 } }] },
    });
    renderPanel();

    expect(await screen.findByText("of $0.00002")).toBeInTheDocument();
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

  it("offers an admin viewer no way to start a write the proxy would reject with a 403", async () => {
    userRole.current = "Admin Viewer";
    renderPanel();

    expect(await screen.findByText("premium")).toBeInTheDocument();

    await openActions("premium");

    expect(await screen.findByTestId("access-group-action-set-budget")).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByTestId("access-group-action-clear-budget")).toHaveAttribute("aria-disabled", "true");
  });

  it("does not offer a budget on a group whose name a path segment cannot carry", async () => {
    GET.mockResolvedValue({ data: { access_groups: [{ ...FREE_GROUP, access_group: "openai/prod" }] } });
    renderPanel();

    await openActions("openai/prod");

    expect(await screen.findByTestId("access-group-action-set-budget")).toHaveAttribute("aria-disabled", "true");
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

  describe("URL table state", () => {
    const manyGroups = Array.from({ length: 27 }, (_, index) => ({
      ...FREE_GROUP,
      access_group: `group-${String(index + 1).padStart(2, "0")}`,
    }));

    it("sorts by the column and direction named in the URL", async () => {
      renderPanel({ searchParams: "?ag_budgets_sort_by=spend&ag_budgets_sort_order=asc" });

      await screen.findByText("premium");

      expect(groupNamesInOrder()).toEqual(["shared", "premium"]);
    });

    it("writes a header sort to the URL", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderPanel({ onUrlUpdate });
      await screen.findByText("premium");

      await userEvent.click(screen.getByTestId("sort-header-access_group"));

      await waitFor(() =>
        expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("ag_budgets_sort_order")).toBe("desc"),
      );
      expect(groupNamesInOrder()).toEqual(["shared", "premium"]);
    });

    it("opens on the page named in the URL once the groups load", async () => {
      GET.mockResolvedValue({ data: { access_groups: manyGroups } });

      renderPanel({ searchParams: "?ag_budgets_page=2" });

      expect(await screen.findByText("group-27")).toBeInTheDocument();
      expect(groupNamesInOrder()).toEqual(["group-26", "group-27"]);
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
    });

    it("writes a page change to the URL", async () => {
      GET.mockResolvedValue({ data: { access_groups: manyGroups } });
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderPanel({ onUrlUpdate });
      await screen.findByText("group-01");

      await userEvent.click(screen.getByTestId("pagination-next"));

      await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("ag_budgets_page")).toBe("2"));
      expect(groupNamesInOrder()).toEqual(["group-26", "group-27"]);
    });
  });
});

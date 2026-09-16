import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction, type UrlUpdateEvent } from "nuqs/adapters/testing";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import { ApiError } from "@/lib/http/client";

import BudgetPanel from "./budget_panel";

const { getMock, budgetDeleteMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  budgetDeleteMock: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  apiClient: { get: getMock },
  budgetCreateCall: vi.fn(),
  budgetUpdateCall: vi.fn(),
  budgetDeleteCall: budgetDeleteMock,
  getProxyBaseUrl: () => "",
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userRole: "Admin", userId: "u1" }),
}));

interface BudgetSeed {
  budget_id: string;
  max_budget: number | null;
  budget_duration: string | null;
}

const budgetRow = (seed: BudgetSeed) => ({
  soft_budget: null,
  tpm_limit: 1000,
  rpm_limit: 10,
  budget_reset_at: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  ...seed,
});

const FORBIDDEN_PROBLEM = {
  type: "about:blank",
  title: "Forbidden",
  status: 403,
  detail: "Only proxy admins can view budgets",
};

const DEFAULT_ROWS = [
  budgetRow({ budget_id: "ecc1869c-6231-4380-a56d-1a0be457477d", max_budget: 100, budget_duration: "30d" }),
];

const respondWith = (rows: ReturnType<typeof budgetRow>[], totalCount: number) => {
  getMock.mockResolvedValue({
    data: rows,
    meta: { total_count: totalCount, page: 1, page_size: 50, total_pages: Math.ceil(totalCount / 50) },
  });
};

type QueryRecord = Record<string, string | number>;

const queries = (): QueryRecord[] => getMock.mock.calls.map((call) => (call[1] as { query: QueryRecord }).query);
const lastQuery = (): QueryRecord => queries()[queries().length - 1];
const paths = (): string[] => getMock.mock.calls.map((call) => String(call[0]));

const freshClient = () => new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });

const renderPanel = (searchParams = "", onUrlUpdate?: OnUrlUpdateFunction) =>
  renderWithProviders(
    <QueryClientProvider client={freshClient()}>
      <BudgetPanel accessToken="sk-test" />
    </QueryClientProvider>,
    { searchParams, onUrlUpdate },
  );

const renderPanelKeepingMountWrites = (searchParams: string, onUrlUpdate: OnUrlUpdateFunction) =>
  render(
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={freshClient()}>
        <BudgetPanel accessToken="sk-test" />
      </QueryClientProvider>
    </NuqsTestingAdapter>,
  );

type UrlUpdateMock = ReturnType<typeof vi.fn<OnUrlUpdateFunction>>;

const lastUrl = (onUrlUpdate: UrlUpdateMock): URLSearchParams => {
  const events = onUrlUpdate.mock.calls.map(([event]: [UrlUpdateEvent]) => event);
  return events[events.length - 1]?.searchParams ?? new URLSearchParams();
};

const openFilters = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByTestId("datatable-filters-trigger"));
  await screen.findByTestId("filter-drawer-body");
};

describe("Budget Panel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    respondWith(DEFAULT_ROWS, 1);
  });

  it("renders the standard page header with the sidebar's Budgets icon", async () => {
    const { container } = renderPanel();

    expect(await screen.findByRole("heading", { level: 1, name: "Budgets" })).toBeInTheDocument();
    expect(screen.getByText("Spend, TPM and RPM limits you can assign to customers.")).toBeInTheDocument();
    expect(container.querySelector(".lucide-wallet")).not.toBeNull();
  });

  it("loads the first page of budgets, newest first", async () => {
    renderPanel();
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    expect(paths()[0]).toBe("/management/v1/budgets");
    expect(queries()[0]).toEqual({ page: 1, page_size: 50, sort: "-created_at" });
    expect(await screen.findByText("ecc1869c-6231-4380-a56d-1a0be457477d")).toBeInTheDocument();
  });

  it("asks the server to sort when a sortable header is clicked", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("", onUrlUpdate);
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    await user.click(screen.getByTestId("sort-header-max_budget"));
    await waitFor(() => expect(lastQuery().sort).toBe("-max_budget"));
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("sort_by")).toBe("max_budget"));
    expect(lastUrl(onUrlUpdate).has("sort_order")).toBe(false);

    await user.click(screen.getByTestId("sort-header-max_budget"));
    await waitFor(() => expect(lastQuery().sort).toBe("max_budget"));
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("sort_order")).toBe("asc"));

    await user.click(screen.getByTestId("sort-header-budget_id"));
    await waitFor(() => expect(lastQuery().sort).toBe("budget_id"));
  });

  it("keeps the sort for every header the table offers", async () => {
    const user = userEvent.setup();
    renderPanel();
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    const headerIds = screen
      .getAllByTestId(/^sort-header-/)
      .map((header) => header.dataset.testid?.replace("sort-header-", "") ?? "");
    expect(headerIds).toContain("tpd_limit");
    for (const id of headerIds) {
      await user.click(screen.getByTestId(`sort-header-${id}`));
      await waitFor(() => expect(String(lastQuery().sort).replace(/^-/, "")).toBe(id));
    }
  });

  it("searches on budget_id with a debounced q", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("", onUrlUpdate);
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "ecc" } });
    await waitFor(() => expect(lastQuery().q).toBe("ecc"));
    expect(queries().some((query) => query.q === "e" || query.q === "ec")).toBe(false);
    expect(lastUrl(onUrlUpdate).get("q")).toBe("ecc");
  });

  it("filters by reset duration and clears it again", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("", onUrlUpdate);
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    await openFilters(user);
    await user.click(screen.getByTestId("budget-filter-duration-7d"));
    await user.click(screen.getByTestId("budget-filter-duration-30d"));
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastQuery()["filter[budget_duration][in]"]).toBe("7d,30d"));
    expect(lastUrl(onUrlUpdate).get("filter_duration")).toBe("7d,30d");

    await user.click(screen.getByTestId("filter-chip-remove-budget_duration"));
    await waitFor(() => expect(lastQuery()).not.toHaveProperty("filter[budget_duration][in]"));
    expect(lastUrl(onUrlUpdate).has("filter_duration")).toBe(false);
  });

  it("filters by budgets with no reset duration", async () => {
    const user = userEvent.setup();
    renderPanel();
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    await openFilters(user);
    await user.click(screen.getByTestId("budget-filter-duration-__unset__"));
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastQuery()["filter[budget_duration][is_null]"]).toBe("true"));
    expect(lastQuery()).not.toHaveProperty("filter[budget_duration][in]");
  });

  it("filters by a max budget range and clears it again", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("", onUrlUpdate);
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    await openFilters(user);
    fireEvent.change(screen.getByTestId("budget-filter-max-budget-min"), { target: { value: "10" } });
    fireEvent.change(screen.getByTestId("budget-filter-max-budget-max"), { target: { value: "500" } });
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastQuery()["filter[max_budget][gte]"]).toBe("10"));
    expect(lastQuery()["filter[max_budget][lte]"]).toBe("500");
    expect(lastUrl(onUrlUpdate).get("filter_max_min")).toBe("10");
    expect(lastUrl(onUrlUpdate).get("filter_max_max")).toBe("500");

    await user.click(screen.getByTestId("datatable-clear-filters"));
    await waitFor(() => expect(lastQuery()).not.toHaveProperty("filter[max_budget][gte]"));
    expect(lastQuery()).not.toHaveProperty("filter[max_budget][lte]");
    expect(lastUrl(onUrlUpdate).has("filter_max_min")).toBe(false);
    expect(lastUrl(onUrlUpdate).has("filter_max_max")).toBe(false);
  });

  it("filters to unlimited budgets only", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("", onUrlUpdate);
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    await openFilters(user);
    fireEvent.change(screen.getByTestId("budget-filter-max-budget-min"), { target: { value: "10" } });
    await user.click(screen.getByTestId("budget-filter-max-budget-unlimited"));
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastQuery()["filter[max_budget][is_null]"]).toBe("true"));
    expect(lastQuery()).not.toHaveProperty("filter[max_budget][gte]");
    expect(lastUrl(onUrlUpdate).get("filter_unlimited")).toBe("true");
    expect(lastUrl(onUrlUpdate).has("filter_max_min")).toBe(false);
  });

  it("filters by a created date range covering whole local days", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("", onUrlUpdate);
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    await openFilters(user);
    fireEvent.change(screen.getByTestId("budget-filter-created-from"), { target: { value: "2026-01-05" } });
    fireEvent.change(screen.getByTestId("budget-filter-created-to"), { target: { value: "2026-01-06" } });
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() =>
      expect(lastQuery()["filter[created_at][gte]"]).toBe(new Date("2026-01-05T00:00:00.000").toISOString()),
    );
    expect(lastQuery()["filter[created_at][lte]"]).toBe(new Date("2026-01-06T23:59:59.999").toISOString());
    expect(lastUrl(onUrlUpdate).get("filter_created_from")).toBe("2026-01-05");
    expect(lastUrl(onUrlUpdate).get("filter_created_to")).toBe("2026-01-06");
  });

  it("pages through the results and changes page size", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    respondWith(DEFAULT_ROWS, 400);
    renderPanel("", onUrlUpdate);
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    await user.click(screen.getByTestId("pagination-next"));
    await waitFor(() => expect(lastQuery().page).toBe(2));
    expect(lastQuery().page_size).toBe(50);
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("page")).toBe("2"));

    await user.click(screen.getByTestId("pagination-page-size"));
    await user.click(await screen.findByRole("option", { name: "25" }));
    await waitFor(() => expect(lastQuery().page_size).toBe(25));
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("page_size")).toBe("25"));
  });

  it("opens a shared link on the same page, sort, search and filters", async () => {
    const user = userEvent.setup();
    respondWith(DEFAULT_ROWS, 400);
    renderPanel(
      "?page=3&page_size=25&sort_by=max_budget&sort_order=asc&q=ecc&filter_duration=7d,30d" +
        "&filter_max_min=10&filter_max_max=500&filter_created_from=2026-01-05&filter_created_to=2026-01-06",
    );

    const expectedQuery: QueryRecord = {
      page: 3,
      page_size: 25,
      sort: "max_budget",
      q: "ecc",
      "filter[budget_duration][in]": "7d,30d",
      "filter[max_budget][gte]": "10",
      "filter[max_budget][lte]": "500",
      "filter[created_at][gte]": new Date("2026-01-05T00:00:00.000").toISOString(),
      "filter[created_at][lte]": new Date("2026-01-06T23:59:59.999").toISOString(),
    };
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    expect(queries()[0]).toEqual(expectedQuery);
    expect(screen.getByTestId("datatable-search")).toHaveValue("ecc");
    expect(screen.getByTestId("filter-chip-remove-budget_duration")).toBeInTheDocument();
    expect(screen.getByTestId("filter-chip-remove-max_budget")).toBeInTheDocument();
    expect(screen.getByTestId("filter-chip-remove-created_at")).toBeInTheDocument();

    await openFilters(user);
    expect(screen.getByTestId("budget-filter-duration-7d")).toBeChecked();
    expect(screen.getByTestId("budget-filter-duration-30d")).toBeChecked();
    expect(screen.getByTestId("budget-filter-duration-1h")).not.toBeChecked();
    expect(screen.getByTestId("budget-filter-max-budget-min")).toHaveValue(10);
    expect(screen.getByTestId("budget-filter-created-to")).toHaveValue("2026-01-06");
  });

  it("opens a shared link filtered to unlimited budgets with no reset duration", async () => {
    renderPanel("?filter_unlimited=true&filter_duration=__unset__");

    const expectedQuery: QueryRecord = {
      page: 1,
      page_size: 50,
      sort: "-created_at",
      "filter[max_budget][is_null]": "true",
      "filter[budget_duration][is_null]": "true",
    };
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    expect(queries()[0]).toEqual(expectedQuery);
  });

  it("opens on the tab named in the URL and records tab switches", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("?tab=examples", onUrlUpdate);

    expect(await screen.findByRole("tab", { name: "Examples" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Budgets" })).toHaveAttribute("aria-selected", "false");

    await user.click(screen.getByRole("tab", { name: "Budgets" }));
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrl(onUrlUpdate).has("tab")).toBe(false);
    expect(screen.getByRole("tab", { name: "Budgets" })).toHaveAttribute("aria-selected", "true");

    await user.click(screen.getByRole("tab", { name: "Examples" }));
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("tab")).toBe("examples"));
  });

  it("falls back to the Budgets tab and clears a tab the page does not have", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    respondWith(DEFAULT_ROWS, 400);
    renderPanelKeepingMountWrites("?tab=settings&page=2", onUrlUpdate);

    expect(await screen.findByRole("tab", { name: "Budgets" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrl(onUrlUpdate).has("tab")).toBe(false);
    expect(lastUrl(onUrlUpdate).get("page")).toBe("2");
  });

  it("renders an access-denied state when the route rejects the caller", async () => {
    getMock.mockRejectedValue(new ApiError("Only proxy admins can view budgets", 403, FORBIDDEN_PROBLEM));
    renderPanel();
    expect(await screen.findByText("You do not have access to budgets")).toBeInTheDocument();
    expect(screen.queryByText("No budgets yet")).not.toBeInTheDocument();
  });

  it("deletes a budget from the actions menu", async () => {
    const user = userEvent.setup();
    budgetDeleteMock.mockResolvedValue(undefined);
    renderPanel();
    await screen.findByText("ecc1869c-6231-4380-a56d-1a0be457477d");

    await user.click(screen.getByTestId("budget-actions-ecc1869c-6231-4380-a56d-1a0be457477d"));
    await user.click(await screen.findByTestId("budget-action-delete"));
    await screen.findByText("Delete Budget?");
    await user.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() =>
      expect(budgetDeleteMock).toHaveBeenCalledWith("sk-test", "ecc1869c-6231-4380-a56d-1a0be457477d"),
    );
  });

  it("refetches the current page after a delete", async () => {
    const user = userEvent.setup();
    budgetDeleteMock.mockResolvedValue(undefined);
    renderPanel();
    await screen.findByText("ecc1869c-6231-4380-a56d-1a0be457477d");
    const before = getMock.mock.calls.length;

    await user.click(screen.getByTestId("budget-actions-ecc1869c-6231-4380-a56d-1a0be457477d"));
    await user.click(await screen.findByTestId("budget-action-delete"));
    await screen.findByText("Delete Budget?");
    await user.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(getMock.mock.calls.length).toBeGreaterThan(before));
  });
});

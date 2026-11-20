import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { uiHref } from "@/utils/uiHref";
import { fireEvent, renderWithProviders, screen, waitFor, within } from "../../../../../tests/test-utils";
import type { ApplicationRow } from "./applications";
import ApplicationsTable from "./ApplicationsTable";
import { appFixture, mockAppRequests } from "./appTestUtils";

const rows: ApplicationRow[] = Array.from({ length: 30 }, (_, index) => {
  const app = appFixture({
    resource_id: `resource-${index}`,
    app_id: `app-${index}`,
    alias: `App ${String(index).padStart(2, "0")}`,
  });
  return { resource: app, app };
});
const onOpen = vi.fn();
const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
const visibleNames = () =>
  within(screen.getByRole("table"))
    .getAllByRole("link")
    .map((link) => link.textContent);
const latestParams = () => onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

beforeEach(() => {
  onOpen.mockClear();
  onUrlUpdate.mockClear();
  mockAppRequests({
    "GET /management/v1/app_budget_groups": () => ({ default_budgets: [] }),
    "GET /management/v1/customer_organizations/daily/activity": (request) => {
      const url = new URL(request.url);
      const index = Number(url.searchParams.get("app_id")?.replace("app-", ""));
      const metrics = { spend: index, total_tokens: index * 1000, api_requests: 1 };
      return {
        results: [{ date: url.searchParams.get("start_date"), metrics }],
        metadata: { has_more: false, total_pages: 1 },
      };
    },
  });
});
afterEach(() => vi.unstubAllGlobals());

it("sorts across all pages, updates the URL and preserves usage links and Edit", async () => {
  const searchParams = { page: "2", app: "resource-3" };
  renderWithProviders(<ApplicationsTable rows={rows} loading={false} onOpen={onOpen} />, { searchParams, onUrlUpdate });
  expect(visibleNames()[0]).toBe("App 25");
  fireEvent.click(screen.getByRole("button", { name: "Monthly spend" }));
  await waitFor(() => expect(visibleNames()[0]).toBe("App 29"));
  expect(visibleNames()).toHaveLength(25);
  await waitFor(() => expect(latestParams()?.get("sort_by")).toBe("spend"));
  expect(latestParams()?.get("page")).toBeNull();
  expect(latestParams()?.get("app")).toBe("resource-3");
  expect(screen.getByRole("columnheader", { name: "Monthly spend" })).toHaveAttribute("aria-sort", "descending");
  expect(screen.getByRole("link", { name: "App 29" })).toHaveAttribute("href", `${uiHref("apps/usage")}?app_id=app-29`);
  fireEvent.click(screen.getByRole("button", { name: "Edit App 29" }));
  expect(onOpen).toHaveBeenCalledWith("resource-29");
  fireEvent.click(screen.getByRole("button", { name: "Monthly spend" }));
  await waitFor(() => expect(visibleNames()[0]).toBe("App 00"));
  expect(screen.getByRole("columnheader", { name: "Monthly spend" })).toHaveAttribute("aria-sort", "ascending");
  expect(within(screen.getByRole("columnheader", { name: "Actions" })).queryByRole("button")).not.toBeInTheDocument();
  expect(within(screen.getAllByRole("rowgroup")[0]).getAllByRole("button")).toHaveLength(7);
});

it("restores sorting and search from the URL and resets pagination when searching", async () => {
  const searchParams = { search: "App", sort_by: "application", sort_order: "desc", page: "2" };
  renderWithProviders(<ApplicationsTable rows={rows} loading={false} />, { searchParams, onUrlUpdate });
  expect(visibleNames()).toEqual(["App 04", "App 03", "App 02", "App 01", "App 00"]);
  fireEvent.change(screen.getByRole("textbox", { name: "Search applications" }), {
    target: { value: "resource-2" },
  });
  await waitFor(() => expect(visibleNames()[0]).toBe("App 29"));
  expect(visibleNames()).toHaveLength(11);
  await waitFor(() => expect(latestParams()?.get("page")).toBeNull());
  expect(latestParams()?.get("search")).toBe("resource-2");
  expect(latestParams()?.get("sort_order")).toBe("desc");
});

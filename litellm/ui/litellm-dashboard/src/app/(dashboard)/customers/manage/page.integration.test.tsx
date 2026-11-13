import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { uiHref } from "@/utils/uiHref";
import { renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../../../tests/test-utils";
import CustomerHierarchyManager from "./page";

const auth = vi.hoisted(() => ({ userRole: "Admin", accessToken: "test-session", isLoading: false }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => auth }));
vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useRouter: () => ({ push: vi.fn() }),
}));

const audit = {
  spend: 12,
  created_at: "2026-11-01T00:00:00Z",
  created_by: "admin",
  updated_at: "2026-11-01T00:00:00Z",
  updated_by: "admin",
};
const organization = {
  ...audit,
  customer_org_id: "org-1",
  alias: "Example customer",
  budget: { max_budget: 100 },
  default_budgets: [],
};
const product = {
  ...audit,
  customer_team_id: "team-a",
  parent_org_id: "org-1",
  alias: "Product",
  budget: { max_budget: 50 },
};
const operations = { ...audit, customer_team_id: "team-b", parent_org_id: "org-1", alias: "Operations", budget: null };
const member = {
  ...audit,
  customer_member_id: "member-a",
  parent_team_id: "team-a",
  alias: "Member One",
  effective_budget: { max_budget: 10 },
};
const fetchMock = vi.fn<typeof fetch>();
const json = (body: unknown) => new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
const requestUrl = (input: Parameters<typeof fetch>[0]) =>
  new URL(input instanceof Request ? input.url : String(input), "http://localhost");
const requestedUrls = () => fetchMock.mock.calls.map(([input]) => requestUrl(input));

function serve(input: Parameters<typeof fetch>[0]): Response {
  const url = requestUrl(input);
  if (url.pathname === "/management/v1/customer_organizations/org-1") return json(organization);
  if (url.pathname === "/management/v1/customer_teams/team-a") return json(product);
  if (url.pathname === "/management/v1/customer_teams/team-b")
    return json({ ...operations, parent_org_id: "another-org" });
  if (url.pathname === "/management/v1/customer_members/member-a") return json(member);
  if (url.pathname === "/management/v1/customer_teams" && url.searchParams.get("parent_org_id") === "org-1") {
    const page = Number(url.searchParams.get("page") ?? 1);
    return json({
      data: page === 1 ? [product] : [operations],
      meta: { total_count: 21, page, page_size: 20, total_pages: 2 },
    });
  }
  if (url.pathname === "/management/v1/customer_members" && url.searchParams.get("parent_team_id") === "team-a") {
    return json({ data: [member], meta: { total_count: 1, page: 1, page_size: 20, total_pages: 1 } });
  }
  throw new Error(`Unexpected customer request ${url.pathname}`);
}

describe("customer hierarchy management", () => {
  beforeEach(() => {
    auth.userRole = "Admin";
    testQueryClient.clear();
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockImplementation(async (input) => serve(input));
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    fetchMock.mockReset();
    testQueryClient.clear();
  });

  it("should render the selected organization and request its paginated teams", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1" });
    const table = await screen.findByRole("table", { name: "Teams" });
    expect(within(table).getByRole("button", { name: "Product" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Example customer" })).toBeInTheDocument();
    expect(
      requestedUrls()
        .find((url) => url.pathname === "/management/v1/customer_teams")
        ?.searchParams.get("page_size"),
    ).toBe("20");
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(await screen.findByRole("button", { name: "Operations" })).toBeInTheDocument();
    expect(
      requestedUrls()
        .filter((url) => url.pathname === "/management/v1/customer_teams")
        .map((url) => [url.searchParams.get("parent_org_id"), url.searchParams.get("page")]),
    ).toEqual([
      ["org-1", "1"],
      ["org-1", "2"],
    ]);
  });

  it.each([2, 0])(
    "should reset an out-of-range hierarchy deep link when the collection has %i pages",
    async (totalPages) => {
      fetchMock.mockImplementation(async (input) => {
        const url = requestUrl(input);
        if (url.pathname !== "/management/v1/customer_teams") return serve(input);
        const page = Number(url.searchParams.get("page"));
        return json({
          data: page === 1 && totalPages > 0 ? [product] : [],
          meta: { total_count: totalPages * 20, page, page_size: 20, total_pages: totalPages },
        });
      });
      const onUrlUpdate = vi.fn();
      renderWithProviders(<CustomerHierarchyManager />, {
        searchParams: "?org=org-1&hierarchyPage=999&source=portal",
        onUrlUpdate,
      });
      await waitFor(() =>
        expect(onUrlUpdate).toHaveBeenLastCalledWith(
          expect.objectContaining({ queryString: "?org=org-1&source=portal" }),
        ),
      );
      if (totalPages > 0) expect(await screen.findByRole("button", { name: "Product" })).toBeInTheDocument();
      else expect(await screen.findByText("No teams in this organization.")).toBeInTheDocument();
      expect(
        requestedUrls()
          .filter((url) => url.pathname === "/management/v1/customer_teams")
          .map((url) => url.searchParams.get("page")),
      ).toEqual(["999", "1"]);
    },
  );

  it("should navigate from a team to member settings and keep scoped usage links", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&source=portal", onUrlUpdate });
    await user.click(await screen.findByRole("button", { name: "Product" }));
    const members = await screen.findByRole("table", { name: "Members" });
    expect(
      requestedUrls()
        .find((url) => url.pathname === "/management/v1/customer_members")
        ?.searchParams.get("parent_team_id"),
    ).toBe("team-a");
    await user.click(within(members).getByRole("button", { name: "Edit Member One" }));
    expect(await screen.findByRole("heading", { name: "Member One" })).toBeInTheDocument();
    const usage = screen.getByRole("link", { name: "Usage" });
    expect(usage).toHaveAttribute("href", `${uiHref("customers/usage")}?org=org-1&team=team-a&member=member-a`);
    expect(screen.getByRole("link", { name: "Settings" })).toHaveAttribute("aria-current", "page");
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({ queryString: "?org=org-1&source=portal&team=team-a&member=member-a" }),
    );
  });

  it("should resolve the member parent for a direct member deep link", async () => {
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&member=member-a" });
    expect(await screen.findByRole("heading", { name: "Member One" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Customer hierarchy" })).toHaveTextContent("Product");
    expect(requestedUrls().some((url) => url.pathname === "/management/v1/customer_teams/team-a")).toBe(true);
    expect(requestedUrls().some((url) => url.pathname === "/management/v1/customer_members")).toBe(false);
  });

  it("should reject a team outside the selected organization before exposing settings", async () => {
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&team=team-b&tab=settings" });
    expect(
      await screen.findByText("This team or member does not belong to the selected organization."),
    ).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Settings" })).not.toBeInTheDocument();
    expect(requestedUrls().some((url) => url.pathname === "/management/v1/customer_members")).toBe(false);
  });

  it("should keep the management surface unavailable to a view-only admin", async () => {
    auth.userRole = "Admin Viewer";
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1" });
    expect(
      screen.getByText("Only proxy admins can manage customer organizations, teams, and members."),
    ).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

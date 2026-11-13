import userEvent from "@testing-library/user-event";
import { parseAsString, useQueryStates } from "nuqs";
import { toast } from "sonner";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { uiHref } from "@/utils/uiHref";
import {
  act,
  fireEvent,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../../tests/test-utils";
import CustomerHierarchyManager from "./page";

const auth = vi.hoisted(() => ({
  userRole: "Admin",
  accessToken: "test-session",
  userId: "admin-1",
  isLoading: false,
}));
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
  parent_team_ids: ["team-a"],
  alias: "Member One",
  effective_budget: { max_budget: 10 },
};
const supportMember = { ...member, customer_member_id: "member-b", parent_team_ids: ["team-c"], alias: "Member Two" };
const fetchMock = vi.fn<typeof fetch>();
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const requestUrl = (input: Parameters<typeof fetch>[0]) =>
  new URL(input instanceof Request ? input.url : String(input), "http://localhost");
const requestedUrls = () => fetchMock.mock.calls.map(([input]) => requestUrl(input));

function serve(input: Parameters<typeof fetch>[0]): Response {
  const url = requestUrl(input);
  if (url.pathname === "/models")
    return json({ data: [{ id: "configured-model", object: "model", created: 0, owned_by: "openai" }] });
  if (url.pathname === "/management/v1/customer_organizations/org-1") return json(organization);
  if (url.pathname === "/management/v1/customer_teams/team-a") return json(product);
  if (url.pathname === "/management/v1/customer_teams/team-b")
    return json({ ...operations, parent_org_id: "another-org" });
  if (url.pathname === "/management/v1/customer_members/member-a") return json(member);
  if (url.pathname === "/management/v1/customer_teams/team-c")
    return json({ ...product, customer_team_id: "team-c", alias: "Support" });
  if (url.pathname === "/management/v1/customer_members/member-b") return json(supportMember);
  if (url.pathname === "/management/v1/customer_members" && url.searchParams.get("parent_org_id") === "org-1") {
    const page = Number(url.searchParams.get("page"));
    return json({
      data: page === 1 ? [member] : [supportMember],
      meta: { total_count: 21, page, page_size: 20, total_pages: 2 },
    });
  }
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

function BrowserNavigationProbe() {
  const parsers = { org: parseAsString, team: parseAsString, member: parseAsString, tab: parseAsString };
  const [scope, setScope] = useQueryStates(parsers);
  const nextScope = { org: "org-2", team: null, member: null, tab: null };
  return (
    <>
      <button type="button" onClick={() => void setScope(nextScope)}>
        Switch customer organization
      </button>
      <output aria-label="Current customer route">
        {scope.org} / {scope.team ?? "none"} / {scope.member ?? "none"}
      </output>
    </>
  );
}

describe("customer hierarchy management", () => {
  it("should preserve a draft on refresh and conflict, then reload and save the current configuration", async () => {
    const user = userEvent.setup();
    const standard = {
      budget_id: "standard",
      name: "Standard",
      budget_limits: [{ budget_duration: "1mo", max_budget: 10 }],
    };
    const alternate = { budget_id: "alternate", name: "Alternate", max_budget: 20, budget_duration: "7d" };
    const initial = {
      ...organization,
      budget_groups_revision: "a".repeat(64),
      default_budgets: [standard, alternate],
      default_budget_template_id: "standard",
    };
    const external = { budget_id: "external", name: "Added elsewhere", max_budget: 30 };
    const refreshed = {
      ...initial,
      budget_groups_revision: "b".repeat(64),
      alias: "Refreshed customer",
      default_budgets: [...initial.default_budgets, external],
    };
    const saved = {
      ...refreshed,
      budget_groups_revision: "c".repeat(64),
      default_budget_template_id: "alternate",
      default_budgets: [
        { ...standard, budget_limits: [{ budget_duration: "1mo", max_budget: 15 }] },
        alternate,
        external,
      ],
    };
    fetchMock.mockImplementation(async () => json(initial));
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&tab=budget_groups" });
    const monthly = await screen.findByRole("spinbutton", { name: "Group 1 monthly budget (USD)" });
    fireEvent.change(monthly, { target: { value: "15" } });
    await user.click(screen.getByRole("radio", { name: "Use Alternate as default" }));
    fetchMock.mockImplementation(async () => json(refreshed));
    await act(() => testQueryClient.invalidateQueries({ queryKey: ["customer-hierarchy"] }));
    expect(await screen.findByRole("heading", { name: "Refreshed customer" })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" })).toHaveValue(15);
    expect(screen.getByRole("radio", { name: "Use Alternate as default" })).toBeChecked();
    fetchMock.mockClear();
    fetchMock.mockImplementation(async () => json({ detail: "Budget groups changed. Reload before saving." }, 409));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Budget groups changed.");
    expect(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" })).toHaveValue(15);
    const request = fetchMock.mock.calls[0][0];
    if (!(request instanceof Request)) throw new Error("Expected HTTP request");
    expect(await request.clone().json()).toMatchObject({ expected_revision: initial.budget_groups_revision });
    fetchMock.mockImplementation(async () => json(refreshed));
    await user.click(screen.getByRole("button", { name: "Discard edits and reload" }));
    expect(await screen.findByDisplayValue("Added elsewhere")).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" })).toHaveValue(10);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" }), {
      target: { value: "15" },
    });
    await user.click(screen.getByRole("radio", { name: "Use Alternate as default" }));
    fetchMock.mockClear();
    fetchMock.mockImplementation(async () => json(saved));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled());
    expect(screen.getByRole("spinbutton", { name: "Group 1 monthly budget (USD)" })).toHaveValue(15);
    expect(screen.getByRole("radio", { name: "Use Alternate as default" })).toBeChecked();
    const writes = fetchMock.mock.calls
      .map(([input]) => input)
      .filter((input): input is Request => input instanceof Request && input.method !== "GET");
    expect(writes).toHaveLength(1);
    expect(await writes[0].clone().json()).toMatchObject({
      expected_revision: refreshed.budget_groups_revision,
      default_budgets: [
        { budget_id: "standard", name: "Standard", budget_limits: [{ budget_duration: "1mo", max_budget: 15 }] },
        { budget_id: "alternate", name: "Alternate" },
        { budget_id: "external", name: "Added elsewhere" },
      ],
    });
  });

  it("should show budget groups only at organization scope without requesting child tables", async () => {
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&tab=budget_groups" });
    expect(await screen.findByRole("link", { name: "Budget groups" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Budget groups" })).toHaveAttribute(
      "href",
      `${uiHref("customers/manage")}?org=org-1&tab=budget_groups`,
    );
    expect(screen.getByRole("button", { name: "Add budget group" })).toBeEnabled();
    expect(requestedUrls().map((url) => url.pathname)).toEqual(["/management/v1/customer_organizations/org-1"]);
  });

  beforeEach(() => {
    auth.userRole = "Admin";
    vi.spyOn(toast, "success").mockReturnValue("test-toast");
    testQueryClient.clear();
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockImplementation(async (input) => serve(input));
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    fetchMock.mockReset();
    testQueryClient.clear();
  });

  it("should render the selected organization and request its paginated teams", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1" });
    const table = await screen.findByRole("table", { name: "Customer teams" });
    expect(within(table).getByRole("button", { name: "Product" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Example customer" })).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Recorded budget spend" })).toHaveAttribute("aria-valuenow", "12");
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

  it("should show effective member window limits without treating them as an absent budget", async () => {
    fetchMock.mockImplementation(async (input) =>
      requestUrl(input).pathname === "/management/v1/customer_members/member-a"
        ? json({
            ...member,
            effective_budget: {
              max_budget: null,
              budget_limits: [
                { budget_duration: "1mo", max_budget: 10 },
                { budget_duration: "12mo", max_budget: 100 },
              ],
            },
          })
        : serve(input),
    );
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&team=team-a&member=member-a" });
    const summary = await screen.findByRole("region", { name: "Current budget" });
    expect(within(summary).getByText("Unavailable / $10.00")).toBeInTheDocument();
    expect(within(summary).getByText("Unavailable / $100.00")).toBeInTheDocument();
    expect(within(summary).getByRole("link", { name: "Edit budget" })).toHaveAttribute(
      "href",
      `${uiHref("customers/manage")}?org=org-1&team=team-a&member=member-a&tab=settings`,
    );
    expect(within(summary).queryByText(/No limit/)).not.toBeInTheDocument();
    expect(within(summary).queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("should list customer members across an organization and open their customer team context", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&tab=members", onUrlUpdate });
    const table = await screen.findByRole("table", { name: "Customer members" });
    expect(screen.getByRole("link", { name: "Customer teams", exact: true })).toHaveAttribute(
      "href",
      `${uiHref("customers/manage")}?org=org-1`,
    );
    expect(screen.getByRole("link", { name: "Customer members", exact: true })).toHaveAttribute("aria-current", "page");
    expect(within(table).getByRole("columnheader", { name: "Customer team" })).toBeInTheDocument();
    expect(within(table).getByRole("button", { name: "team-a" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Choose a customer team" })).toHaveAttribute(
      "href",
      `${uiHref("customers/manage")}?org=org-1&tab=teams`,
    );
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Member Two" }));
    expect(await screen.findByRole("heading", { name: "Member Two" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Customer hierarchy" })).toHaveTextContent("Support");
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({ queryString: "?org=org-1&team=team-c&member=member-b" }),
    );
    expect(
      requestedUrls()
        .filter((url) => url.pathname === "/management/v1/customer_members")
        .map((url) => [
          url.searchParams.get("parent_org_id"),
          url.searchParams.get("page_size"),
          url.searchParams.get("page"),
        ]),
    ).toEqual([
      ["org-1", "20", "1"],
      ["org-1", "20", "2"],
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
      else expect(await screen.findByText("No customer teams in this customer organization.")).toBeInTheDocument();
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
    const members = await screen.findByRole("table", { name: "Customer members" });
    expect(screen.getByText("Manage customer members and settings for this customer team.")).toBeInTheDocument();
    expect(
      requestedUrls()
        .find((url) => url.pathname === "/management/v1/customer_members")
        ?.searchParams.get("parent_team_id"),
    ).toBe("team-a");
    await user.click(within(members).getByRole("button", { name: "Edit Member One" }));
    expect(await screen.findByRole("heading", { name: "Member One" })).toBeInTheDocument();
    expect(screen.getByText("Manage settings and budget for this customer member.")).toBeInTheDocument();
    const usage = screen.getByRole("link", { name: "Usage" });
    expect(usage).toHaveAttribute("href", `${uiHref("customers/usage")}?org=org-1&team=team-a&member=member-a`);
    expect(screen.getByRole("link", { name: "Settings" })).toHaveAttribute("aria-current", "page");
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({ queryString: "?org=org-1&source=portal&team=team-a&member=member-a" }),
    );
  });

  it.each(["team", "member"] as const)(
    "should open a newly created customer %s in its scoped settings",
    async (kind) => {
      const user = userEvent.setup();
      const isTeam = kind === "team";
      const createdId = `created-${kind}`;
      const alias = `New customer ${kind}`;
      const collection = isTeam ? "customer_teams" : "customer_members";
      const parent = isTeam ? { parent_org_id: "org-1" } : { parent_team_ids: ["team-a"] };
      const created = isTeam
        ? { ...product, ...parent, customer_team_id: createdId, alias }
        : { ...member, ...parent, customer_member_id: createdId, alias, budget_id: "template-default" };
      const posts: unknown[] = [];
      fetchMock.mockImplementation(async (input, init) => {
        const url = requestUrl(input);
        const method = input instanceof Request ? input.method : init?.method;
        if (url.pathname === `/management/v1/${collection}` && method === "POST") {
          const body: unknown = input instanceof Request ? await input.clone().json() : JSON.parse(String(init?.body));
          posts.push(body);
          return json(created, 201);
        }
        if (url.pathname === `/management/v1/${collection}/${createdId}`) return json(created);
        if (url.pathname === "/management/v1/customer_organizations/org-1")
          return json({
            ...organization,
            default_budget_template_id: "template-default",
            default_budgets: [{ budget_id: "template-default", name: "Default", is_default: true, max_budget: 20 }],
          });
        return serve(input);
      });
      const onUrlUpdate = vi.fn();
      renderWithProviders(<CustomerHierarchyManager />, {
        searchParams: isTeam ? "?org=org-1" : "?org=org-1&team=team-a",
        onUrlUpdate,
      });
      await user.click(
        await screen.findByRole("button", { name: isTeam ? "Create customer team" : "Create customer member" }),
      );
      act(() => fireEvent.change(screen.getByLabelText("Display name"), { target: { value: alias } }));
      await user.click(screen.getByRole("button", { name: "Create", exact: true }));
      expect(await screen.findByRole("heading", { name: alias })).toBeInTheDocument();
      expect(posts).toEqual([expect.objectContaining(parent)]);
      expect(screen.getByLabelText("Display name")).toHaveValue(alias);
      const queryString = isTeam
        ? "?org=org-1&team=created-team&tab=settings"
        : "?org=org-1&team=team-a&member=created-member";
      expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString }));
      expect(screen.getByRole("link", { name: "Settings" })).toHaveAttribute("aria-current", "page");
    },
  );

  it("should keep the current customer scope after a creation completes on a previous customer", async () => {
    const user = userEvent.setup();
    const creation = Promise.withResolvers<Response>();
    const created = { ...product, customer_team_id: "created-team", alias: "Created team" };
    fetchMock.mockImplementation(async (input, init) => {
      const url = requestUrl(input);
      const method = input instanceof Request ? input.method : init?.method;
      if (url.pathname === "/management/v1/customer_teams" && method === "POST") return creation.promise;
      if (url.pathname === "/management/v1/customer_organizations/org-2")
        return json({ ...organization, customer_org_id: "org-2", alias: "Other customer" });
      if (url.pathname === "/management/v1/customer_teams" && url.searchParams.get("parent_org_id") === "org-2")
        return json({ data: [], meta: { total_count: 0, page: 1, page_size: 20, total_pages: 0 } });
      if (url.pathname === "/management/v1/customer_teams/created-team") return json(created);
      return serve(input);
    });
    renderWithProviders(
      <>
        <CustomerHierarchyManager />
        <BrowserNavigationProbe />
      </>,
      { searchParams: "?org=org-1" },
    );
    await user.click(await screen.findByRole("button", { name: "Create customer team" }));
    act(() => fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Created team" } }));
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));
    expect(screen.getByRole("button", { name: "Creating…" })).toBeDisabled();
    act(() => fireEvent.click(screen.getByRole("button", { name: "Switch customer organization", hidden: true })));
    expect(await screen.findByRole("heading", { name: "Other customer" })).toBeInTheDocument();
    await act(async () => {
      creation.resolve(json(created, 201));
    });
    await waitFor(() => expect(toast.success).toHaveBeenCalled());
    expect(screen.getByRole("status", { name: "Current customer route" })).toHaveTextContent("org-2 / none / none");
    expect(screen.getByRole("heading", { name: "Other customer" })).toBeInTheDocument();
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
      await screen.findByText(
        "This customer team or customer member does not belong to the selected customer organization.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Settings" })).not.toBeInTheDocument();
    expect(requestedUrls().some((url) => url.pathname === "/management/v1/customer_members")).toBe(false);
  });

  it("should keep the management surface unavailable to a view-only admin", async () => {
    auth.userRole = "Admin Viewer";
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1" });
    expect(
      screen.getByText("Only proxy admins can manage customer organizations, customer teams, and customer members."),
    ).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

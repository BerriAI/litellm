import userEvent from "@testing-library/user-event";
import { parseAsString, useQueryStates } from "nuqs";
import { toast } from "sonner";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { uiHref } from "@/utils/uiHref";
import { EMPTY_DAILY_ACTIVITY_METADATA } from "@/components/UsagePage/dailyActivityApi";
import {
  jsonResponse as json,
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
const navigation = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => auth }));
vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useRouter: () => navigation,
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
const requestUrl = (input: Parameters<typeof fetch>[0]) =>
  new URL(input instanceof Request ? input.url : String(input), "http://localhost");
const requestedUrls = () => fetchMock.mock.calls.map(([input]) => requestUrl(input));

function serve(input: Parameters<typeof fetch>[0]): Response {
  const url = requestUrl(input);
  if (url.pathname === "/management/v1/apps/definition") return json({ resource_type: null });
  if (url.pathname.endsWith("/daily/activity/aggregated"))
    return json({ results: [], metadata: EMPTY_DAILY_ACTIVITY_METADATA });
  if (url.pathname.endsWith("/daily/activity")) return json({ results: [], metadata: EMPTY_DAILY_ACTIVITY_METADATA });
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
  it.each([
    { scope: "&tab=teams", label: "Product", id: "team-a", usage: "&team=team-a" },
    { scope: "&tab=members", label: "Member One", id: "member-a", usage: "&member=member-a" },
    {
      scope: "&team=team-a",
      label: "Member One",
      id: "member-a",
      usage: "&team=team-a&member=member-a",
    },
  ])("opens usage from the $scope row while preserving filters", async ({ scope, label, id, usage }) => {
    const user = userEvent.setup();
    const filters = "&from=2026-11-01&to=2026-11-03&app_id=app-1";
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: `?org=org-1${scope}${filters}` });
    const row = await screen.findByRole("row", { name: new RegExp(label) });
    const href = `${uiHref("customers/usage")}?org=org-1${usage}${filters}`;
    expect(within(row).getByRole("link", { name: label })).toHaveAttribute("href", href);

    await user.click(within(row).getByText(id, { exact: true }));

    expect(navigation.push).toHaveBeenCalledExactlyOnceWith(href);
  });

  it("opens the parent team without also navigating to member usage", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&tab=members", onUrlUpdate });

    await user.click(await screen.findByRole("button", { name: "team-a" }));

    expect(await screen.findByRole("heading", { name: "Product" })).toBeInTheDocument();
    expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString: "?org=org-1&team=team-a" }));
    expect(navigation.push).not.toHaveBeenCalled();
  });

  it.each(["teams", "members"])(
    "should request %s budget sorting before pagination and store it in the URL",
    async (kind) => {
      const onUrlUpdate = vi.fn();
      const options = { searchParams: `?org=org-1&tab=${kind}&hierarchyPage=2&source=portal`, onUrlUpdate };
      renderWithProviders(<CustomerHierarchyManager />, options);
      await screen.findByRole("link", { name: kind === "teams" ? "Operations" : "Member Two" }, { timeout: 10_000 });
      const table = screen.getByRole("table", { name: kind === "teams" ? "Customer teams" : "Customer members" });
      const sort = within(table).getByRole("button", { name: "Max Budget (USD)" });
      act(() => {
        fireEvent.click(sort);
      });
      await waitFor(() =>
        expect(onUrlUpdate).toHaveBeenLastCalledWith(
          expect.objectContaining({ queryString: `?org=org-1&tab=${kind}&source=portal&budgetSort=budget_asc` }),
        ),
      );
      const listRequests = () => requestedUrls().filter((url) => url.pathname === `/management/v1/customer_${kind}`);
      await waitFor(() => expect(listRequests().at(-1)?.searchParams.get("sort")).toBe("budget_asc"));
      expect(listRequests().at(-1)?.searchParams.get("page")).toBe("1");
      act(() => {
        fireEvent.click(within(screen.getByRole("table")).getByRole("button", { name: "Max Budget (USD)" }));
      });
      await waitFor(() => expect(listRequests().at(-1)?.searchParams.get("sort")).toBe("budget_desc"));
    },
  );

  it.each(["", "Member One"])(
    "should move a member through its row menu with search %j without editing budgets",
    async (search) => {
      const user = userEvent.setup();
      const moved = { ...member, parent_team_ids: ["team-b"] };
      fetchMock.mockImplementation(async (input, init) => {
        const url = requestUrl(input);
        const method = input instanceof Request ? input.method : init?.method;
        if (url.pathname === "/management/v1/customer_members/member-a" && method === "PATCH") return json(moved);
        return serve(input);
      });
      const options = { searchParams: `?org=org-1&tab=members&memberSearch=${encodeURIComponent(search)}` };
      renderWithProviders(<CustomerHierarchyManager />, options);
      await user.click(await screen.findByRole("button", { name: "Actions for Member One" }));
      expect(screen.getAllByRole("searchbox", { name: "Search members by name or ID" })).toHaveLength(1);
      expect(screen.getByRole("searchbox", { name: "Search members by name or ID" })).toHaveValue(search);
      expect(
        requestedUrls()
          .find((url) => url.pathname === "/management/v1/customer_members")
          ?.searchParams.get("search"),
      ).toBe(search || null);
      await user.keyboard("{ArrowDown}");
      await user.click(await screen.findByRole("menuitem", { name: "Move teams" }));
      const dialog = screen.getByRole("dialog", { name: "Move teams" });
      const teams = within(dialog).getByRole("combobox", { name: "Choose teams" });
      await waitFor(() => expect(teams).toBeEnabled());
      await user.click(teams);
      await user.click(screen.getByRole("option", { name: /Operations/ }));
      await user.click(screen.getByRole("option", { name: /Product/ }));
      await user.keyboard("{Escape}");
      await user.click(within(dialog).getByRole("button", { name: "Move teams" }));
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      const write = fetchMock.mock.calls
        .map(([input]) => input)
        .find((input) => input instanceof Request && input.method === "PATCH");
      expect(write).toBeInstanceOf(Request);
      expect(await (write as Request).clone().json()).toEqual({ parent_team_ids: ["team-b"] });
      expect(navigation.push).not.toHaveBeenCalled();
    },
  );

  it.each(["teams", "members"] as const)(
    "should show usage-style budgets, spend share, tokens and actions for customer %s",
    async (kind) => {
      const id = kind === "teams" ? "team-a" : "member-a";
      const label = kind === "teams" ? "Product" : "Member One";
      const collection = kind === "teams" ? "customer_teams" : "customer_members";
      const budgetUsage = [
        { source: "scalar", budget_duration: "1mo", max_budget: 10, spend: 3, reset_at: null },
        { source: "window", budget_duration: "12mo", max_budget: 100, spend: 7, reset_at: null },
      ];
      fetchMock.mockImplementation(async (input) => {
        const url = requestUrl(input);
        if (url.pathname === `/management/v1/${collection}`) {
          return json({
            data: [
              { ...(kind === "teams" ? product : member), budget_usage: budgetUsage },
              {
                ...(kind === "teams" ? product : member),
                [kind === "teams" ? "customer_team_id" : "customer_member_id"]: `${id},other`,
                alias: "Comma ID",
              },
            ],
            meta: { total_count: 2, page: 1, page_size: 20, total_pages: 1 },
          });
        }
        if (url.pathname.endsWith("/daily/activity/aggregated"))
          return json({ results: [], metadata: { ...EMPTY_DAILY_ACTIVITY_METADATA, total_spend: 12 } });
        if (!url.pathname.endsWith("/daily/activity")) return serve(input);
        const spend = url.pathname.includes("customer_organizations") ? 12 : 3;
        const metrics = {
          spend,
          total_tokens: 250,
          prompt_tokens: 250,
          completion_tokens: 0,
          api_requests: 1,
          successful_requests: 1,
          failed_requests: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        };
        return json({
          results: [
            {
              date: url.searchParams.get("start_date"),
              metrics,
              breakdown: {
                models: {},
                model_groups: {},
                providers: {},
                mcp_servers: {},
                api_keys: {},
                entities: { [id]: { metrics, metadata: {}, api_key_breakdown: {} } },
              },
            },
          ],
          metadata: {
            ...EMPTY_DAILY_ACTIVITY_METADATA,
            total_spend: spend,
            total_tokens: 250,
            total_api_requests: 1,
          },
        });
      });
      renderWithProviders(<CustomerHierarchyManager />, { searchParams: `?org=org-1&tab=${kind}` });
      const table = await screen.findByRole("table", {
        name: kind === "teams" ? "Customer teams" : "Customer members",
      });
      const progress = await within(table).findByRole("progressbar", { name: "Monthly spend / limit" });
      expect(progress).toHaveAttribute("aria-valuenow", "30");
      expect(within(table).getByText("$3.00 / $10.00")).toBeInTheDocument();
      expect(within(table).getByText("$7.00 left")).toBeInTheDocument();
      expect(within(table).getByText("$7.00 / $100.00")).toBeInTheDocument();
      expect(within(table).getByRole("meter", { name: "Share of customer spend" })).toHaveAttribute(
        "aria-valuenow",
        "25",
      );
      expect(within(table).getByRole("cell", { name: "250" })).toBeInTheDocument();
      expect(within(table).getByText("Within budget")).toBeInTheDocument();
      expect(
        within(table).getByRole("button", { name: kind === "teams" ? `Edit ${label}` : `Actions for ${label}` }),
      ).toBeEnabled();
      const validRow = within(table).getByRole("row", { name: new RegExp(label) });
      expect(within(validRow).getByRole("link", { name: label })).toHaveAttribute(
        "href",
        `${uiHref("customers/usage")}?org=org-1&${kind === "teams" ? "team" : "member"}=${id}`,
      );
      const unavailableRow = within(table).getByRole("row", { name: /Comma ID/ });
      expect(within(unavailableRow).getByRole("status")).toHaveTextContent("Unavailable");
      expect(within(unavailableRow).queryByRole("meter")).not.toBeInTheDocument();
      const activity = requestedUrls().filter((url) => url.pathname.endsWith("/daily/activity"));
      expect(activity).toHaveLength(2);
      expect(
        activity
          .filter((url) => url.pathname.includes(collection))
          .every((url) => url.searchParams.get(kind === "teams" ? "customer_team_ids" : "customer_member_ids") === id),
      ).toBe(true);
    },
  );

  it.each(["/daily/activity", "/daily/activity/aggregated"])(
    "should keep row actions without showing zero usage when %s fails",
    async (route) => {
      fetchMock.mockImplementation(async (input) =>
        requestUrl(input).pathname.endsWith(route) ? json({ detail: "Usage unavailable" }, 503) : serve(input),
      );
      renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1" });
      const table = await screen.findByRole("table", { name: "Customer teams" });
      expect(await screen.findByRole("alert")).toHaveTextContent("Usage unavailable");
      expect(within(table).getByRole("status")).toHaveTextContent("Unavailable");
      expect(within(table).queryByRole("meter")).not.toBeInTheDocument();
      expect(within(table).getByRole("button", { name: "Edit Product" })).toBeEnabled();
    },
  );

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
    await screen.findByRole("row", { name: "Standard" });
    const monthlyInput = () =>
      within(screen.getByRole("row", { name: "Standard" })).getByRole("spinbutton", {
        name: "Additional Max Budget (USD)",
      });
    act(() => {
      fireEvent.change(monthlyInput(), { target: { value: "15" } });
    });
    await user.click(screen.getByRole("combobox", { name: "New members join" }));
    await user.click(await screen.findByRole("option", { name: "Alternate", exact: true }));
    fetchMock.mockImplementation(async () => json(refreshed));
    await act(() => testQueryClient.invalidateQueries({ queryKey: ["customer-hierarchy"] }));
    expect(await screen.findByRole("heading", { name: "Refreshed customer" })).toBeInTheDocument();
    expect(monthlyInput()).toHaveValue(15);
    expect(screen.getByRole("combobox", { name: "New members join" })).toHaveTextContent("Alternate");
    fetchMock.mockClear();
    fetchMock.mockImplementation(async () => json({ detail: "Budget groups changed. Reload before saving." }, 409));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Budget groups changed.");
    expect(monthlyInput()).toHaveValue(15);
    const request = fetchMock.mock.calls[0][0];
    if (!(request instanceof Request)) throw new Error("Expected HTTP request");
    expect(await request.clone().json()).toMatchObject({ expected_revision: initial.budget_groups_revision });
    fetchMock.mockImplementation(async () => json(refreshed));
    await user.click(screen.getByRole("button", { name: "Discard edits and reload" }));
    expect(await screen.findByDisplayValue("Added elsewhere")).toBeInTheDocument();
    expect(monthlyInput()).toHaveValue(10);
    act(() => {
      fireEvent.change(monthlyInput(), {
        target: { value: "15" },
      });
    });
    await user.click(screen.getByRole("combobox", { name: "New members join" }));
    await user.click(await screen.findByRole("option", { name: "Alternate", exact: true }));
    fetchMock.mockClear();
    fetchMock.mockImplementation(async () => json(saved));
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled());
    expect(monthlyInput()).toHaveValue(15);
    expect(screen.getByRole("combobox", { name: "New members join" })).toHaveTextContent("Alternate");
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
    navigation.push.mockReset();
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

  it("should keep usage dates while opening team settings and returning to usage", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&from=2026-11-01&to=2026-11-03" });
    await screen.findByRole("table", { name: "Customer teams" });
    await user.click(screen.getByRole("button", { name: "Edit Product" }));
    await screen.findByRole("heading", { name: "Product" });
    expect(screen.getByRole("link", { name: "Usage", exact: true })).toHaveAttribute(
      "href",
      `${uiHref("customers/usage")}?org=org-1&team=team-a&from=2026-11-01&to=2026-11-03`,
    );
    expect(screen.getByRole("link", { name: "All customers", exact: true })).toHaveAttribute(
      "href",
      `${uiHref("customers")}?from=2026-11-01&to=2026-11-03`,
    );
  });

  it("should render the selected organization and request its paginated teams", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1" });
    const table = await screen.findByRole("table", { name: "Customer teams" });
    expect(within(table).getByRole("link", { name: "Product" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Example customer" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Current budget" })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "App budgets" })).not.toBeInTheDocument();
    expect(requestedUrls().some((url) => url.pathname.startsWith("/management/v1/apps"))).toBe(false);
    expect(screen.getByRole("progressbar", { name: "Recorded budget spend" })).toHaveAttribute("aria-valuenow", "12");
    expect(
      requestedUrls()
        .find((url) => url.pathname === "/management/v1/customer_teams")
        ?.searchParams.get("page_size"),
    ).toBe("20");
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(await screen.findByRole("link", { name: "Operations" })).toBeInTheDocument();
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
    expect(within(summary).getByRole("button", { name: "Edit budget" })).toBeEnabled();
    expect(within(summary).queryByText(/No limit/)).not.toBeInTheDocument();
    expect(within(summary).queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("shows member budget groups and opens budget editing without leaving the member list", async () => {
    const groupedMember = { ...member, budget_id: "member-standard" };
    const groupedOrg = {
      ...organization,
      default_budgets: [{ budget_id: "member-standard", name: "Member Standard", max_budget: 10 }],
    };
    fetchMock.mockImplementation(async (input) => {
      const url = requestUrl(input);
      if (url.pathname === "/management/v1/customer_organizations/org-1") return json(groupedOrg);
      if (url.pathname === "/management/v1/customer_members/member-a") return json(groupedMember);
      if (url.pathname === "/management/v1/customer_members")
        return json({ data: [groupedMember], meta: { total_count: 1, page: 1, page_size: 20, total_pages: 1 } });
      return serve(input);
    });
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&tab=members" });
    const table = await screen.findByRole("table", { name: "Customer members" });
    expect(within(table).getByRole("columnheader", { name: "Budget group" })).toBeInTheDocument();
    expect(within(table).getByRole("cell", { name: "Member Standard", exact: true })).toBeInTheDocument();
    await user.click(within(table).getByRole("button", { name: "Actions for Member One" }));
    await user.click(await screen.findByRole("menuitem", { name: "Edit budget", exact: true }));
    const dialog = within(await screen.findByRole("dialog", { name: "Edit member budget", exact: true }));
    expect(await dialog.findByRole("combobox", { name: "Member limits" })).toHaveTextContent("Member Standard");
    await user.click(dialog.getByRole("button", { name: "Cancel", exact: true }));
    expect(screen.getByRole("link", { name: "Customer members", exact: true })).toHaveAttribute("aria-current", "page");
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
    await user.click(screen.getByRole("button", { name: "Create customer member" }));
    expect(screen.getByRole("dialog", { name: "Create customer member" })).toHaveTextContent("Example customer");
    expect(screen.getByRole("combobox", { name: "Member limits" })).toHaveTextContent("Custom");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Actions for Member Two" }));
    await user.click(await screen.findByRole("menuitem", { name: "Settings" }));
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

  it("should search customer members on the server from the first page and ignore a slower earlier search", async () => {
    const slow = Promise.withResolvers<Response>();
    fetchMock.mockImplementation(async (input) => {
      const url = requestUrl(input);
      const search = url.searchParams.get("search");
      if (url.pathname !== "/management/v1/customer_members" || search === null) return serve(input);
      if (search === "one") return slow.promise;
      const data = search === "two" ? [supportMember] : [];
      return json({ data, meta: { total_count: data.length, page: 1, page_size: 20, total_pages: 1 } });
    });
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyManager />, {
      searchParams: "?org=org-1&tab=members&hierarchyPage=2",
      onUrlUpdate,
    });
    const table = await screen.findByRole("table", { name: "Customer members" });
    expect(screen.queryByRole("region", { name: "App budgets" })).not.toBeInTheDocument();
    expect(requestedUrls().some((url) => url.pathname.startsWith("/management/v1/apps"))).toBe(false);
    expect(within(table).queryByRole("link", { name: "Member One" })).not.toBeInTheDocument();
    const search = screen.getByRole("searchbox", { name: "Search members by name or ID" });
    const searches = () =>
      requestedUrls()
        .filter((url) => url.pathname === "/management/v1/customer_members" && url.searchParams.has("search"))
        .map((url) => [
          url.searchParams.get("search"),
          url.searchParams.get("parent_org_id"),
          url.searchParams.get("page"),
        ]);

    act(() => {
      fireEvent.change(search, { target: { value: " one " } });
    });
    await waitFor(() => expect(searches()).toEqual([["one", "org-1", "1"]]));
    expect(onUrlUpdate).toHaveBeenLastCalledWith(
      expect.objectContaining({ queryString: "?org=org-1&tab=members&memberSearch=one" }),
    );
    act(() => {
      fireEvent.change(search, { target: { value: "two" } });
    });
    expect(await screen.findByRole("link", { name: "Member Two" })).toBeInTheDocument();
    slow.resolve(json({ data: [member], meta: { total_count: 1, page: 1, page_size: 20, total_pages: 1 } }));
    await act(async () => {
      await slow.promise;
    });
    expect(screen.queryByRole("link", { name: "Member One" })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Member Two" })).toBeInTheDocument();

    act(() => {
      fireEvent.change(search, { target: { value: "nobody" } });
    });
    expect(await screen.findByText("No customer members match your search.")).toBeInTheDocument();
    expect(search).toHaveValue("nobody");
    act(() => {
      fireEvent.change(search, { target: { value: "" } });
    });
    expect(await screen.findByRole("link", { name: "Member One" })).toBeInTheDocument();
    expect(onUrlUpdate).toHaveBeenLastCalledWith(expect.objectContaining({ queryString: "?org=org-1&tab=members" }));
    expect(searches().map(([value]) => value)).toEqual(["one", "two", "nobody"]);
  });

  it("should restore a customer member search from the URL within the selected customer team", async () => {
    renderWithProviders(<CustomerHierarchyManager />, { searchParams: "?org=org-1&team=team-a&memberSearch=Member" });
    expect(await screen.findByRole("link", { name: "Member One" })).toBeInTheDocument();
    expect(screen.getByRole("searchbox", { name: "Search members by name or ID" })).toHaveValue("Member");
    expect(
      requestedUrls()
        .filter((url) => url.pathname === "/management/v1/customer_members")
        .map((url) => [url.searchParams.get("parent_team_id"), url.searchParams.get("search")]),
    ).toEqual([["team-a", "Member"]]);
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
      if (totalPages > 0) expect(await screen.findByRole("link", { name: "Product" })).toBeInTheDocument();
      else expect(await screen.findByText("No customer teams in this customer organization.")).toBeInTheDocument();
      expect(
        requestedUrls()
          .filter((url) => url.pathname === "/management/v1/customer_teams")
          .map((url) => url.searchParams.get("page")),
      ).toEqual(["999", "1"]);
    },
  );

  it("should open member settings from the row menu and keep scoped usage links", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomerHierarchyManager />, {
      searchParams: "?org=org-1&source=portal&team=team-a",
      onUrlUpdate,
    });
    const members = await screen.findByRole("table", { name: "Customer members" });
    expect(screen.getByRole("heading", { name: "Product" })).toBeInTheDocument();
    expect(
      requestedUrls()
        .find((url) => url.pathname === "/management/v1/customer_members")
        ?.searchParams.get("parent_team_id"),
    ).toBe("team-a");
    await user.click(within(members).getByRole("button", { name: "Actions for Member One" }));
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("menuitem", { name: "Settings" }));
    expect(await screen.findByRole("heading", { name: "Member One" })).toBeInTheDocument();
    const usage = screen.getByRole("link", { name: "Usage" });
    expect(usage).toHaveAttribute("href", `${uiHref("customers/usage")}?org=org-1&team=team-a&member=member-a`);
    expect(screen.getByRole("link", { name: "Settings" })).toHaveAttribute("aria-current", "page");
    expect(navigation.push).not.toHaveBeenCalled();
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

  it("shows the resolved member hierarchy and lets each breadcrumb open its parent", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomerHierarchyManager />, {
      searchParams: "?org=org-1&member=member-a&from=2026-11-01&to=2026-11-03",
    });
    expect(await screen.findByRole("heading", { name: "Member One" })).toBeInTheDocument();
    const path = () => within(screen.getByRole("navigation", { name: "Customer hierarchy" }));
    expect(
      path()
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual(["All customers", "Example customer", "Product", "Member One"]);
    expect(path().getByRole("link", { name: "Member One", exact: true })).toHaveAttribute("aria-current", "page");
    expect(path().getByRole("link", { name: "Example customer", exact: true })).toHaveAttribute(
      "href",
      `${uiHref("customers/manage")}?org=org-1&from=2026-11-01&to=2026-11-03`,
    );
    expect(requestedUrls().some((url) => url.pathname === "/management/v1/customer_teams/team-a")).toBe(true);
    expect(requestedUrls().some((url) => url.pathname === "/management/v1/customer_members")).toBe(false);
    await user.click(path().getByRole("link", { name: "Product", exact: true }));
    expect(await screen.findByRole("heading", { name: "Product" })).toBeInTheDocument();
    expect(path().getByRole("link", { name: "Product", exact: true })).toHaveAttribute("aria-current", "page");
    await user.click(path().getByRole("link", { name: "Example customer", exact: true }));
    expect(await screen.findByRole("heading", { name: "Example customer" })).toBeInTheDocument();
    expect(
      path()
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual(["All customers", "Example customer"]);
    expect(path().getByRole("link", { name: "Example customer", exact: true })).toHaveAttribute("aria-current", "page");
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

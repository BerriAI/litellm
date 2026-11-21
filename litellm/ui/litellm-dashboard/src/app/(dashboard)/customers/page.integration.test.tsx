import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import type { SpendMetrics } from "@/components/UsagePage/types";
import type { components } from "@/lib/http/schema";
import {
  act,
  fireEvent,
  renderWithProviders,
  screen,
  testQueryClient,
  waitFor,
  within,
} from "../../../../tests/test-utils";
import CustomersPage from "./page";
import type { CustomerOrganization } from "./_components/customerOrganizations";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
const NOW = new Date("2024-03-15T12:00:00Z");
const original: CustomerOrganization[] = Array.from({ length: 30 }, (_, index) => ({
  customer_org_id: `customer-${String(index).padStart(2, "0")}`,
  alias: `Customer ${String(index).padStart(2, "0")}`,
  spend: index === 0 ? 8 : 0,
  budget: index === 0 ? { max_budget: 10, budget_duration: "1mo" } : null,
  budget_usage:
    index === 0
      ? [
          { source: "scalar", budget_duration: "1mo", spend: 8, max_budget: 10, reset_at: null },
          { source: "window", budget_duration: "12mo", spend: 100, max_budget: 100, reset_at: null },
        ]
      : [],
  member_count: 0,
  default_budgets: [],
  budget_groups_revision: "revision",
  created_at: new Date(Date.UTC(2024, 1, index + 1)).toISOString(),
  updated_at: NOW.toISOString(),
  created_by: "admin",
  updated_by: "admin",
}));
const mockCustomerApi = ({
  failed = false,
  groupsFailed = false,
  organizations = original,
  cachePricing,
}: Readonly<{
  failed?: boolean;
  groupsFailed?: boolean;
  organizations?: readonly CustomerOrganization[];
  cachePricing?: Readonly<Pick<SpendMetrics, "cache_cost" | "priced_cache_tokens">>;
}> = {}) => {
  const updateOrganization = vi.fn<(id: string, body: components["schemas"]["OrganizationPatch"]) => void>();
  const createOrganization = vi.fn((body: components["schemas"]["OrganizationCreate"]) => ({
    ...original[1],
    customer_org_id: body.customer_org_id,
    alias: body.alias,
    created_at: NOW.toISOString(),
  }));
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request = new Request(input, init);
    const url = new URL(request.url);
    if (url.pathname.endsWith("/.well-known/litellm-ui-config"))
      return Response.json({ server_root_path: "", proxy_base_url: "", admin_ui_disabled: false });
    if (url.pathname === "/management/v1/users/me")
      return Response.json({ user_id: "admin", user_role: "proxy_admin", user_email: null });
    if (url.pathname === "/management/v1/customer_organization_budget_groups") {
      if (groupsFailed) return Response.json({ detail: "Unavailable" }, { status: 503 });
      return Response.json({
        default_budgets: [{ budget_id: "group-growth", name: "Growth" }],
        default_budget_template_id: null,
        budget_groups_revision: "revision",
      });
    }
    if (url.pathname === "/management/v1/customer_organizations" && request.method === "GET") {
      const data = [
        ...organizations.map((organization) => ({
          ...organization,
          ...updateOrganization.mock.calls.findLast(([id]) => id === organization.customer_org_id)?.[1],
        })),
        ...createOrganization.mock.results.flatMap((result) => (result.type === "return" ? [result.value] : [])),
      ];
      return Response.json({
        data,
        meta: { total_count: data.length, page: 1, page_size: 100, total_pages: 1 },
        links: {},
      });
    }
    if (url.pathname === "/management/v1/customer_organizations/daily/activity/aggregated") {
      if (failed) return Response.json({ detail: "Unavailable" }, { status: 503 });
      const previous = url.searchParams.get("start_date") === "2024-02-01";
      const metrics = (spend: number) => ({
        spend,
        api_requests: 10,
        successful_requests: 9,
        failed_requests: 1,
        total_tokens: 100,
        prompt_tokens: 60,
        completion_tokens: 40,
        cache_read_input_tokens: previous ? 12 : 10,
        cache_creation_input_tokens: previous ? 6 : 4,
        cache_cost: spend / 100,
        priced_cache_tokens: previous ? 12 : 14,
        ...cachePricing,
        total_response_time_ms: 9000,
        timed_requests: 9,
      });
      return Response.json({
        results: [
          {
            date: previous ? "2024-02-15" : "2024-03-15",
            metrics: {
              ...metrics(previous ? 20 : 100),
              api_requests: 20,
              successful_requests: 18,
              failed_requests: 2,
              total_tokens: 200,
              prompt_tokens: 120,
              completion_tokens: 80,
            },
            breakdown: {
              entities: {
                "customer-00": { metrics: metrics(previous ? 5 : 75) },
                "customer-29": { metrics: metrics(previous ? 15 : 25) },
              },
            },
          },
        ],
        metadata: { has_more: false, total_pages: 1 },
      });
    }
    if (url.pathname === "/management/v1/customer_organizations" && request.method === "POST") {
      const body: components["schemas"]["OrganizationCreate"] = await request.json();
      const created = createOrganization(body);
      return Response.json({ customer_org_id: created.customer_org_id }, { status: 201 });
    }
    const organization = organizations.find(
      (entry) => url.pathname === `/management/v1/customer_organizations/${entry.customer_org_id}`,
    );
    if (organization && request.method === "GET") return Response.json(organization);
    if (organization && request.method === "PATCH") {
      const body: components["schemas"]["OrganizationPatch"] = await request.json();
      updateOrganization(organization.customer_org_id, body);
      return Response.json({ ...organization, ...body });
    }
    throw new Error(`Unexpected request ${request.method} ${url.pathname}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { createOrganization, updateOrganization };
};

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
  testQueryClient.clear();
  const claims = { key: "sk-test", user_id: "admin", user_role: "proxy_admin", exp: NOW.getTime() / 1000 + 3600 };
  document.cookie = `token=${btoa("{}")}.${btoa(JSON.stringify(claims))}.test; Path=/`;
  mockCustomerApi();
  vi.stubGlobal("requestIdleCallback", (callback: IdleRequestCallback) =>
    window.setTimeout(() => callback({ didTimeout: false, timeRemaining: () => 50 }), 0),
  );
});
afterEach(() => {
  testQueryClient.clear();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  document.cookie = "token=; Max-Age=0; Path=/";
});

const customerLinks = () => within(screen.getByRole("table", { name: "Customer organizations" })).getAllByRole("link");

describe("customer organization overview", () => {
  it("edits an existing organization budget and refreshes its row without changing filters or pagination", async () => {
    const { updateOrganization } = mockCustomerApi();
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomersPage />, {
      searchParams: "?search=customer&sort_by=customer&sort_order=desc&page=2&from=2024-02-01&to=2024-02-29",
      onUrlUpdate,
    });
    const row = await screen.findByRole("row", { name: /Customer 00/ });
    await user.click(within(row).getByRole("button", { name: "Edit", exact: true }));
    const limit = await screen.findByRole("spinbutton", { name: "Max Budget (USD)" });
    expect(limit).toHaveValue(10);
    fireEvent.change(limit, { target: { value: "25" } });
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(updateOrganization).toHaveBeenCalledExactlyOnceWith(
      "customer-00",
      expect.objectContaining({ budget: expect.objectContaining({ max_budget: 25, budget_duration: "1mo" }) }),
    );
    expect(await within(row).findByText("$25.00", { exact: true })).toBeInTheDocument();
    expect(screen.getByText("Page 2 of 2")).toBeInTheDocument();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  it("shows each organization's assigned tier or custom budget in the budget group column", async () => {
    mockCustomerApi({
      organizations: [
        {
          ...original[0],
          budget_id: "group-growth",
          budget_is_shared: true,
          default_budgets: [{ budget_id: "group-growth", name: "Member default" }],
        },
        { ...original[1], budget_id: "private-budget", budget_is_shared: false, budget: { max_budget: 25 } },
        original[2],
      ],
    });
    renderWithProviders(<CustomersPage />);
    await screen.findByRole("cell", { name: "Growth", exact: true });
    expect(screen.getByRole("columnheader", { name: "Tier" })).toBeInTheDocument();
    expect(
      within(screen.getByRole("row", { name: /Customer 00/ })).getByRole("cell", { name: "Growth", exact: true }),
    ).toBeInTheDocument();
    expect(
      within(screen.getByRole("row", { name: /Customer 01/ })).getByRole("cell", { name: "Custom", exact: true }),
    ).toBeInTheDocument();
    expect(within(screen.getByRole("row", { name: /Customer 02/ })).queryByText("Custom")).not.toBeInTheDocument();
    expect(screen.queryByText("Member default")).not.toBeInTheDocument();
  });

  it("keeps limits visible without calling a shared tier custom when its name cannot load", async () => {
    mockCustomerApi({
      groupsFailed: true,
      organizations: [{ ...original[0], budget_id: "group-growth", budget_is_shared: true }],
    });
    renderWithProviders(<CustomersPage />);
    const row = await screen.findByRole("row", { name: /Customer 00/ });
    expect(within(row).getByText("Budget group")).toBeInTheDocument();
    expect(within(row).getByText("$8.00 / $10.00")).toBeInTheDocument();
    expect(within(row).queryByText("Custom")).not.toBeInTheDocument();
  });

  it("should keep complete totals and shares while sorting, paginating, and searching", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPage />);
    await screen.findByRole("link", { name: "Customer 29" });
    const spendTable = within(screen.getByRole("table", { name: "Customer organizations" }));
    expect(spendTable.getByRole("columnheader", { name: "Tokens", exact: true })).toBeInTheDocument();
    expect(spendTable.queryByRole("columnheader", { name: "Recorded cache cost" })).not.toBeInTheDocument();
    expect(customerLinks()[0]).toHaveTextContent("Customer 29");
    expect(customerLinks()[0]).toHaveAttribute(
      "href",
      "/ui/customers/usage?org=customer-29&from=2024-03-01&to=2024-03-15",
    );
    const overview = screen.getByRole("region", { name: "Overall customer usage" });
    expect(within(overview).getByText("$100.00")).toBeInTheDocument();
    expect(within(overview).getByText("20")).toBeInTheDocument();
    expect(within(screen.getByRole("row", { name: /Customer 29/ })).getByRole("meter")).toHaveAttribute(
      "aria-valuenow",
      "25",
    );
    await user.click(screen.getByRole("button", { name: "Overall spend share" }));
    await waitFor(() => expect(customerLinks()[0]).toHaveTextContent("Customer 00"));
    expect(screen.getByRole("columnheader", { name: "Overall spend share" })).toHaveAttribute(
      "aria-sort",
      "descending",
    );
    const row = screen.getByRole("row", { name: /Customer 00/ });
    const totalTokens = within(row).getByRole("button", { name: "100 total tokens. View breakdown" });
    expect(totalTokens).toHaveTextContent("100");
    expect(within(row).getByText("17% cached")).toBeInTheDocument();
    await user.click(totalTokens);
    expect(await screen.findByRole("dialog", { name: "Token breakdown" })).toHaveTextContent(
      /Input tokens\s*60\s*Cache read tokens\s*10\s*Cache write tokens\s*4\s*Output tokens\s*40\s*Recorded cache cost\s*\$0\.75/,
    );
    await user.keyboard("{Escape}");
    expect(within(row).getByText("$8.00 / $10.00")).toBeInTheDocument();
    expect(within(row).getByText("Reached limit")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Go to next page" }));
    expect(await screen.findByText("Page 2 of 2")).toBeInTheDocument();
    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: /Search by name or ID/ }), {
        target: { value: "Customer 29" },
      });
    });
    await screen.findByText("Page 1 of 1");
    expect(customerLinks()).toHaveLength(1);
    expect(within(screen.getByRole("row", { name: /Customer 29/ })).getByText("25%")).toBeInTheDocument();
    expect(within(overview).getByText("$100.00")).toBeInTheDocument();
  });

  it.each(["range=2024-02-01..2024-02-29", "from=2024-02-01&to=2024-02-29"])(
    "should restore saved sorting and %s without changing monthly shares",
    async (dates) => {
      const user = userEvent.setup();
      renderWithProviders(<CustomersPage />, {
        searchParams: `?sort_by=customer&sort_order=asc&${dates}`,
      });
      await screen.findByRole("link", { name: "Customer 00" });
      expect(customerLinks()[0]).toHaveTextContent("Customer 00");
      expect(
        within(screen.getByRole("region", { name: "Overall customer usage" })).getByText("$20.00"),
      ).toBeInTheDocument();
      expect(within(screen.getByRole("row", { name: /Customer 00/ })).getByText("75%")).toBeInTheDocument();
      const row = within(screen.getByRole("row", { name: /Customer 00/ }));
      expect(row.getByText("20% cached")).toBeInTheDocument();
      await user.click(row.getByRole("button", { name: "100 total tokens. View breakdown" }));
      expect(await screen.findByRole("dialog", { name: "Token breakdown" })).toHaveTextContent(
        /Cache read tokens\s*12\s*Cache write tokens\s*6\s*Output tokens\s*40\s*Recorded cache cost\s*\$0\.05$/,
      );
      await user.keyboard("{Escape}");
      expect(screen.getByRole("link", { name: "Customer 00" })).toHaveAttribute(
        "href",
        "/ui/customers/usage?org=customer-00&from=2024-02-01&to=2024-02-29",
      );
      await user.click(screen.getByRole("button", { name: /1 Feb 2024/ }));
      await user.click(screen.getByRole("button", { name: /This month/ }));
      await user.click(screen.getByRole("button", { name: "Apply" }));
      expect(
        await within(screen.getByRole("region", { name: "Overall customer usage" })).findByText("$100.00"),
      ).toBeInTheDocument();
      expect(within(screen.getByRole("row", { name: /Customer 00/ })).getByText("75%")).toBeInTheDocument();
    },
  );

  it("should show a newly created customer first and resets the search, sort, and page", async () => {
    const { createOrganization } = mockCustomerApi();
    const user = userEvent.setup();
    renderWithProviders(<CustomersPage />, { searchParams: "?search=customer&sort_by=customer&sort_order=asc&page=2" });
    await screen.findByRole("link", { name: "Customer 25" });
    await user.click(screen.getByRole("button", { name: "Create organization" }));
    act(() => {
      fireEvent.change(screen.getByLabelText("Customer name"), { target: { value: "New organization" } });
    });
    act(() => {
      fireEvent.change(screen.getByLabelText("Organization ID (optional)"), { target: { value: "new-org" } });
    });
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Create customer" }));
    await screen.findByRole("link", { name: "New organization" });
    expect(customerLinks()[0]).toHaveTextContent("New organization");
    expect(screen.getByRole("textbox", { name: /Search by name or ID/ })).toHaveValue("");
    expect(screen.getByText("Page 1 of 2")).toBeInTheDocument();
    expect(createOrganization).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ customer_org_id: "new-org", alias: "New organization" }),
    );
  });

  it.each([
    { pricedTokens: 0, display: "—" },
    { pricedTokens: 14, display: "$0.00" },
  ])("shows $display for zero cache cost with $pricedTokens priced tokens", async ({ pricedTokens, display }) => {
    mockCustomerApi({ cachePricing: { cache_cost: 0, priced_cache_tokens: pricedTokens } });
    const user = userEvent.setup();
    renderWithProviders(<CustomersPage />);
    const row = await screen.findByRole("row", { name: /Customer 29/ });
    await user.click(within(row).getByRole("button", { name: "100 total tokens. View breakdown" }));
    const breakdown = await screen.findByRole("dialog", { name: "Token breakdown" });
    expect(breakdown).toHaveTextContent(`Recorded cache cost${display}`);
  });

  it("should report failed usage instead of displaying partial totals or zero shares", async () => {
    mockCustomerApi({ failed: true });
    renderWithProviders(<CustomersPage />);
    expect(await screen.findAllByText("Could not load customer organization usage.")).toHaveLength(2);
    expect(screen.queryByRole("link", { name: "Customer 29" })).not.toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
  });
});

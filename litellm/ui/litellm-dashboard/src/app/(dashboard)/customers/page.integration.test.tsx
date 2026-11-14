import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../../tests/test-utils";
import CustomersPage from "./page";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
const NOW = new Date("2024-03-15T12:00:00Z");
const requests: URL[] = [];
const creations: unknown[] = [];
let failed = false;
const original = Array.from({ length: 30 }, (_, index) => ({
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
  created_at: new Date(Date.UTC(2024, 1, index + 1)).toISOString(),
}));
let organizations = [...original];
const handleRequest = async (request: Request): Promise<Response> => {
  const url = new URL(request.url);
  if (url.pathname.endsWith("/.well-known/litellm-ui-config"))
    return Response.json({ server_root_path: "", proxy_base_url: "", admin_ui_disabled: false });
  if (url.pathname === "/management/v1/users/me")
    return Response.json({ user_id: "admin", user_role: "proxy_admin", user_email: null });
  if (url.pathname === "/management/v1/customer_organizations" && request.method === "GET")
    return Response.json({
      data: organizations,
      meta: { total_count: organizations.length, page: 1, page_size: 100, total_pages: 1 },
      links: {},
    });
  if (url.pathname === "/management/v1/customer_organizations/daily/activity/aggregated") {
    requests.push(url);
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
      cache_read_input_tokens: 10,
      cache_creation_input_tokens: 0,
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
    const body = await request.json();
    creations.push(body);
    organizations = [
      ...organizations,
      { ...original[1], customer_org_id: body.customer_org_id, alias: body.alias, created_at: NOW.toISOString() },
    ];
    return Response.json({ customer_org_id: body.customer_org_id }, { status: 201 });
  }
  throw new Error(`Unexpected request ${request.method} ${url.pathname}`);
};

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
  testQueryClient.clear();
  requests.length = 0;
  creations.length = 0;
  organizations = [...original];
  failed = false;
  const claims = { key: "sk-test", user_id: "admin", user_role: "proxy_admin", exp: NOW.getTime() / 1000 + 3600 };
  document.cookie = `token=${btoa("{}")}.${btoa(JSON.stringify(claims))}.test; Path=/`;
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => handleRequest(new Request(input, init))),
  );
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
  it("keeps complete totals and shares while sorting, paginating, and searching", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPage />);
    await screen.findByRole("link", { name: "Customer 29" });
    expect(customerLinks()[0]).toHaveTextContent("Customer 29");
    const overview = screen.getByRole("region", { name: "Overall customer usage" });
    expect(within(overview).getByText("$100.00")).toBeInTheDocument();
    expect(within(overview).getByText("20")).toBeInTheDocument();
    expect(within(screen.getByRole("row", { name: /Customer 29/ })).getByRole("meter")).toHaveAttribute(
      "aria-valuenow",
      "25",
    );
    await user.click(screen.getByRole("button", { name: "Share of overall spend" }));
    await waitFor(() => expect(customerLinks()[0]).toHaveTextContent("Customer 00"));
    expect(screen.getByRole("columnheader", { name: "Share of overall spend" })).toHaveAttribute(
      "aria-sort",
      "descending",
    );
    const row = screen.getByRole("row", { name: /Customer 00/ });
    expect(within(row).getByText("$8.00 / $10.00")).toBeInTheDocument();
    expect(within(row).getByText("Reached limit")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Go to next page" }));
    expect(await screen.findByText("Page 2 of 2")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: /Search by customer/ }), { target: { value: "Customer 29" } });
    await screen.findByText("Page 1 of 1");
    expect(customerLinks()).toHaveLength(1);
    expect(within(screen.getByRole("row", { name: /Customer 29/ })).getByText("25%")).toBeInTheDocument();
    expect(within(overview).getByText("$100.00")).toBeInTheDocument();
  });

  it("restores saved sorting and changes summary dates without changing monthly shares", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPage />, {
      searchParams: "?sort_by=customer&sort_order=asc&range=2024-02-01..2024-02-29",
    });
    await screen.findByRole("link", { name: "Customer 00" });
    expect(customerLinks()[0]).toHaveTextContent("Customer 00");
    expect(
      within(screen.getByRole("region", { name: "Overall customer usage" })).getByText("$20.00"),
    ).toBeInTheDocument();
    expect(within(screen.getByRole("row", { name: /Customer 00/ })).getByText("75%")).toBeInTheDocument();
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
  });

  it("shows a newly created customer first and resets the search, sort, and page", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPage />, { searchParams: "?search=customer&sort_by=customer&sort_order=asc&page=2" });
    await screen.findByRole("link", { name: "Customer 25" });
    await user.click(screen.getByRole("button", { name: "Create customer organization" }));
    fireEvent.change(screen.getByLabelText("Customer name"), { target: { value: "New organization" } });
    fireEvent.change(screen.getByLabelText("Customer organization id"), { target: { value: "new-org" } });
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Create customer" }));
    await screen.findByRole("link", { name: "New organization" });
    expect(customerLinks()[0]).toHaveTextContent("New organization");
    expect(screen.getByRole("textbox", { name: /Search by customer/ })).toHaveValue("");
    expect(screen.getByText("Page 1 of 2")).toBeInTheDocument();
    expect(creations).toEqual([expect.objectContaining({ customer_org_id: "new-org", alias: "New organization" })]);
  });

  it("reports failed usage instead of displaying partial totals or zero shares", async () => {
    failed = true;
    renderWithProviders(<CustomersPage />);
    expect(await screen.findAllByText("Could not load customer organization usage.")).toHaveLength(2);
    expect(screen.queryByRole("link", { name: "Customer 29" })).not.toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
  });
});

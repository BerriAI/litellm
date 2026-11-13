import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../../tests/test-utils";
import CustomersPage from "./page";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");

const NOW = new Date("2024-03-15T12:00:00Z");
const requests: URL[] = [];
const listRequests: URL[] = [];
const creations: unknown[] = [];
let failed = false;

const handleRequest = async (request: Request): Promise<Response> => {
  const url = new URL(request.url);
  if (url.pathname.endsWith("/.well-known/litellm-ui-config")) {
    return Response.json({ server_root_path: "", proxy_base_url: "", admin_ui_disabled: false });
  }
  if (url.pathname === "/management/v1/users/me") {
    return Response.json({ user_id: "admin", user_role: "proxy_admin", user_email: null });
  }
  if (url.pathname === "/management/v1/customer_organizations" && request.method === "GET") {
    listRequests.push(url);
    return Response.json({
      data: [
        { customer_org_id: "customer-a", alias: "Customer A", spend: 51, budget: { max_budget: 500 }, member_count: 3 },
        { customer_org_id: "over-budget", spend: 750, budget: { max_budget: 500 }, member_count: 2 },
        { customer_org_id: "zero-budget", spend: 1, budget: { max_budget: 0 }, member_count: 1 },
        { customer_org_id: "no-budget", spend: 0, budget: null, member_count: 0 },
        { customer_org_id: "unused-budget", spend: 0, budget: { max_budget: 500 }, member_count: 4 },
      ],
      meta: { total_count: 30, page: Number(url.searchParams.get("page")), page_size: 25, total_pages: 2 },
      links: {},
    });
  }
  if (url.pathname === "/management/v1/customer_organizations/daily/activity/aggregated") {
    requests.push(url);
    if (failed) return Response.json({ detail: "Unavailable" }, { status: 503 });
    const previous = url.searchParams.get("start_date") === "2024-02-01";
    return Response.json({
      results: [
        {
          date: previous ? "2024-02-01" : "2024-03-01",
          breakdown: {
            entities: {
              "customer-a": { metrics: { spend: previous ? 3 : 15, api_requests: 10, total_tokens: 300 } },
            },
          },
        },
        {
          date: previous ? "2024-02-15" : "2024-03-15",
          breakdown: {
            entities: {
              "customer-a": {
                metrics: {
                  spend: previous ? 4 : 36,
                  api_requests: previous ? 10 : 30,
                  total_tokens: previous ? 100 : 500,
                },
              },
            },
          },
        },
      ],
      metadata: { has_more: false, total_pages: 1 },
    });
  }
  if (url.pathname === "/management/v1/customer_organizations" && request.method === "POST") {
    const body: unknown = await request.json();
    creations.push(body);
    return Response.json({ customer_org_id: "new-org" }, { status: 201 });
  }
  throw new Error(`Unexpected request ${request.method} ${url.pathname}`);
};

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
  testQueryClient.clear();
  requests.length = 0;
  listRequests.length = 0;
  creations.length = 0;
  failed = false;
  const claims = { key: "sk-test", user_id: "admin", user_role: "proxy_admin", exp: NOW.getTime() / 1000 + 3600 };
  document.cookie = `token=${btoa("{}")}.${btoa(JSON.stringify(claims))}.test; Path=/`;
  vi.stubGlobal("fetch", vi.fn(handleRequest));
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

describe("customer organization usage", () => {
  it("changes range totals in one aggregate request while retaining current budget usage", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomersPage />, { searchParams: "?page=2", onUrlUpdate });
    const row = await screen.findByRole("row", { name: /Customer A/ });
    expect(within(row).getByRole("cell", { name: "$51.00 / $500.00 10.2% used" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Budget usage" })).not.toBeInTheDocument();
    expect(within(row).getByText("3")).toBeInTheDocument();
    expect(within(row).getByText("40")).toBeInTheDocument();
    expect(within(row).getByText("800")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /1 Mar 2024/ }));
    await user.click(screen.getByRole("button", { name: /Previous month/ }));
    await user.click(screen.getByRole("button", { name: "Apply" }));
    expect(await screen.findByText("$7.00")).toBeInTheDocument();
    const updatedRow = screen.getByRole("row", { name: /Customer A/ });
    expect(
      within(updatedRow).getByRole("cell", { name: "$7.00 Current budget $51.00 / $500.00 10.2% used" }),
    ).toBeInTheDocument();
    expect(within(updatedRow).getByText("3")).toBeInTheDocument();
    expect(within(updatedRow).getByRole("link", { name: "Customer A" })).toHaveAttribute(
      "href",
      "/ui/customers/usage?org=customer-a&from=2024-02-01&to=2024-02-29",
    );
    expect(requests.at(-1)?.searchParams.get("start_date")).toBe("2024-02-01");
    expect(requests.at(-1)?.searchParams.get("end_date")).toBe("2024-02-29");
    expect(listRequests.at(-1)?.searchParams.get("page")).toBe("1");
    expect(requests).toHaveLength(2);
    expect(requests.at(-1)?.searchParams.get("customer_org_ids")).toBe(
      "customer-a,over-budget,zero-budget,no-budget,unused-budget",
    );
    expect(requests.at(-1)?.searchParams.has("page")).toBe(false);
    expect(onUrlUpdate.mock.lastCall?.[0].queryString).toContain("range=2024-02-01..2024-02-29");
  });

  it("restores a custom range from the URL and sends edited calendar days", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPage />, { searchParams: "?range=2024-01-03..2024-01-10" });
    await screen.findByRole("row", { name: /Customer A/ });
    expect(requests.at(-1)?.searchParams.get("start_date")).toBe("2024-01-03");
    expect(requests.at(-1)?.searchParams.get("end_date")).toBe("2024-01-10");
    await user.click(screen.getByRole("button", { name: /3 Jan 2024/ }));
    fireEvent.change(screen.getByLabelText("Start date"), { target: { value: "2024-02-01" } });
    fireEvent.change(screen.getByLabelText("End date"), { target: { value: "2024-02-29" } });
    await user.click(screen.getByRole("button", { name: "Apply" }));
    expect(await screen.findByText("$7.00")).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.get("end_date")).toBe("2024-02-29");
  });

  it("shows over-budget percentages, zero and missing limits, and members without usage", async () => {
    renderWithProviders(<CustomersPage />);
    const over = await screen.findByRole("row", { name: /over-budget/ });
    expect(within(over).getByText("Current budget $750.00 / $500.00")).toBeInTheDocument();
    expect(within(over).getByText("150% used")).toBeInTheDocument();
    const zero = screen.getByRole("row", { name: /zero-budget/ });
    expect(within(zero).getByText("Current budget $1.00 / $0.00")).toBeInTheDocument();
    expect(within(zero).getByText("Percentage unavailable for a $0 budget")).toBeInTheDocument();
    const uncapped = screen.getByRole("row", { name: /no-budget/ });
    expect(within(uncapped).getByText("$0.00")).toBeInTheDocument();
    expect(within(uncapped).getByText("No max budget")).toBeInTheDocument();
    const unused = screen.getByRole("row", { name: /unused-budget/ });
    expect(within(unused).getByText("$0.00 / $500.00")).toBeInTheDocument();
    expect(within(unused).getByText("0% used")).toBeInTheDocument();
    expect(within(unused).getByText("4")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Members" })).toBeInTheDocument();
  });

  it("blocks ranges over 400 days before Apply and accepts the boundary", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPage />);
    await screen.findByRole("row", { name: /Customer A/ });
    await user.click(screen.getByRole("button", { name: /1 Mar 2024/ }));
    fireEvent.change(screen.getByLabelText("Start date"), { target: { value: "2024-01-01" } });
    fireEvent.change(screen.getByLabelText("End date"), { target: { value: "2025-02-05" } });
    expect(screen.getByText("Select a date range of 400 days or less.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply" })).toBeDisabled();
    expect(requests).toHaveLength(1);
    fireEvent.change(screen.getByLabelText("End date"), { target: { value: "2025-02-04" } });
    expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(requests.at(-1)?.searchParams.get("end_date")).toBe("2025-02-04");
  });

  it("uses the default month when a saved URL exceeds the range limit", async () => {
    renderWithProviders(<CustomersPage />, { searchParams: "?range=2020-01-01..2024-03-31" });
    await screen.findByRole("row", { name: /Customer A/ });
    expect(requests).toHaveLength(1);
    expect(requests[0].searchParams.get("start_date")).toBe("2024-03-01");
    expect(requests[0].searchParams.get("end_date")).toBe("2024-03-31");
  });

  it("shows a failure instead of presenting missing usage as zero", async () => {
    failed = true;
    renderWithProviders(<CustomersPage />);
    expect(await screen.findByText("Could not load customer organization usage.")).toBeInTheDocument();
    expect(screen.queryByRole("row", { name: /Customer A/ })).not.toBeInTheDocument();
  });

  it("creates an organization using the entered ID and alias", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CustomersPage />);
    await user.click(screen.getByRole("button", { name: "Create Customer" }));
    fireEvent.change(screen.getByLabelText("Customer organization id"), { target: { value: " new-org " } });
    fireEvent.change(screen.getByLabelText("Alias"), { target: { value: " New customer " } });
    const dialog = screen.getByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Create" }));
    await waitFor(() =>
      expect(creations).toEqual([{ customer_org_id: "new-org", alias: "New customer", default_budgets: [] }]),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});

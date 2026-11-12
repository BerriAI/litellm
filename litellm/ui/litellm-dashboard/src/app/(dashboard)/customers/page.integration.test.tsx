import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../../tests/test-utils";
import CustomersPage from "./page";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");

const NOW = new Date("2024-03-15T12:00:00Z");
const requests: URL[] = [];
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
  if (url.pathname === "/management/v1/customer_organization_usage") {
    requests.push(url);
    if (failed) return Response.json({ detail: "Unavailable" }, { status: 503 });
    const previous = url.searchParams.get("start_date") === "2024-02-01";
    return Response.json({
      data: [
        {
          customer_org_id: "customer-a",
          alias: "Customer A",
          spend: previous ? 7 : 12,
          requests: previous ? 20 : 40,
          tokens: previous ? 400 : 800,
        },
      ],
      meta: { total_count: 30, page: Number(url.searchParams.get("page")), page_size: 25, total_pages: 2 },
      links: {},
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
  it("applies the previous month to the API, updates totals and resets pagination", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<CustomersPage />, { searchParams: "?page=2", onUrlUpdate });
    const row = await screen.findByRole("row", { name: /Customer A/ });
    expect(within(row).getByText("$12.00")).toBeInTheDocument();
    expect(within(row).getByText("40")).toBeInTheDocument();
    expect(within(row).getByText("800")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /1 Mar 2024/ }));
    await user.click(screen.getByRole("button", { name: /Previous month/ }));
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }));
    expect(await screen.findByText("$7.00")).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.get("start_date")).toBe("2024-02-01");
    expect(requests.at(-1)?.searchParams.get("end_date")).toBe("2024-02-29");
    expect(requests.at(-1)?.searchParams.get("page")).toBe("1");
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
    await user.click(screen.getByRole("button", { name: "Apply", exact: true }));
    expect(await screen.findByText("$7.00")).toBeInTheDocument();
    expect(requests.at(-1)?.searchParams.get("end_date")).toBe("2024-02-29");
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
    await user.click(screen.getByRole("button", { name: "Create Customer", exact: true }));
    fireEvent.change(screen.getByLabelText("Customer organization id"), { target: { value: " new-org " } });
    fireEvent.change(screen.getByLabelText("Alias"), { target: { value: " New customer " } });
    const dialog = screen.getByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Create", exact: true }));
    await waitFor(() =>
      expect(creations).toEqual([{ customer_org_id: "new-org", alias: "New customer", default_budgets: [] }]),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});

import { NuqsAdapter } from "nuqs/adapters/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { EMPTY_DAILY_ACTIVITY_RESPONSE } from "@/components/UsagePage/dailyActivityApi";
import { act, fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "../../../../tests/test-utils";
import CustomersPage from "./page";

const fetchMock = vi.fn<typeof fetch>();
const customer = {
  user_id: "customer-1",
  alias: "Acme",
  blocked: false,
  spend: 0,
  fallbacks: null,
  litellm_budget_table: null,
};

const jsonResponse = (body: unknown) =>
  new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });

const renderCustomers = () =>
  renderWithProviders(
    <NuqsAdapter>
      <CustomersPage />
    </NuqsAdapter>,
  );

describe("customer navigation", () => {
  beforeEach(() => {
    testQueryClient.clear();
    window.history.replaceState(null, "", "/customers?source=usage#summary");
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    fetchMock.mockImplementation(async (input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      const path = new URL(request.url).pathname;
      if (request.method === "GET" && path === "/customer/list") return jsonResponse([customer]);
      if (request.method === "GET" && path === "/customer/info") return jsonResponse(customer);
      if (request.method === "GET" && path === "/customer/daily/activity/aggregated") {
        return jsonResponse(EMPTY_DAILY_ACTIVITY_RESPONSE);
      }
      throw new Error(`Unexpected request: ${request.method} ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    testQueryClient.clear();
    vi.unstubAllGlobals();
  });

  it("should render the customer list and restore its search when using browser Back from a customer", async () => {
    renderCustomers();
    expect(await screen.findByRole("heading", { name: "Customers", level: 1 })).toBeInTheDocument();
    expect(await screen.findByText("Acme")).toBeInTheDocument();
    act(() => {
      fireEvent.change(screen.getByPlaceholderText("Search by customer id or alias"), { target: { value: "Acme" } });
    });
    expect(await screen.findByText("Acme")).toBeInTheDocument();
    const historyLength = window.history.length;
    act(() => {
      fireEvent.click(screen.getByText("Acme"));
    });

    expect(await screen.findByRole("heading", { name: "Acme", level: 1 })).toBeInTheDocument();
    await waitFor(() => expect(window.location.search).toBe("?source=usage&id=customer-1"));
    expect(window.history).toHaveLength(historyLength + 1);
    expect(window.location.hash).toBe("#summary");
    expect(screen.getAllByRole("button")[0]).toHaveAccessibleName("Back to customers");
    expect(screen.getByRole("button", { name: "Back to customers" })).toBeVisible();

    act(() => window.history.back());

    expect(await screen.findByRole("heading", { name: "Customers", level: 1 })).toBeInTheDocument();
    expect(window.location.search).toBe("?source=usage");
    expect(window.location.hash).toBe("#summary");
    expect(screen.getByPlaceholderText("Search by customer id or alias")).toHaveValue("Acme");
  });

  it("should return directly linked customers to the list without losing other URL state", async () => {
    window.history.replaceState(null, "", "/customers?source=usage&id=customer-1#summary");
    const historyLength = window.history.length;
    renderCustomers();
    expect(await screen.findByRole("heading", { name: "Acme", level: 1 })).toBeInTheDocument();
    const back = screen.getByRole("button", { name: "Back to customers" });
    expect(back).toBeVisible();
    expect(screen.getAllByRole("button")[0]).toBe(back);

    act(() => {
      fireEvent.click(back);
    });

    expect(await screen.findByRole("heading", { name: "Customers", level: 1 })).toBeInTheDocument();
    await waitFor(() => expect(window.location.search).toBe("?source=usage"));
    expect(window.location.hash).toBe("#summary");
    expect(window.history).toHaveLength(historyLength);
    expect(screen.queryByRole("button", { name: "Back to customers" })).not.toBeInTheDocument();
  });
});

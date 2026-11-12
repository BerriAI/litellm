import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import EditCustomerDialog from "./EditCustomerDialog";

interface SavedRequest {
  path: string;
  body: unknown;
}

const customer: EndUser = {
  user_id: "customer-1",
  alias: "Customer one",
  blocked: false,
  spend: 0,
  budget_id: "budget-1",
  litellm_budget_table: {
    budget_id: "budget-1",
    created_at: "2026-01-01T00:00:00Z",
    max_budget: 25,
    tpm_limit: 1000,
    rpm_limit: 60,
    model_tpm_limit: { primary: 500 },
    model_rpm_limit: { primary: 10 },
  },
};

const fetchMock = vi.fn<typeof fetch>();
const onSaved = vi.fn();
const onOpenChange = vi.fn();
let savedRequests: SavedRequest[] = [];
let rejectBudget = false;

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const renderDialog = (value: EndUser = customer) =>
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } })
      }
    >
      <EditCustomerDialog open onOpenChange={onOpenChange} customer={value} onSaved={onSaved} />
    </QueryClientProvider>,
  );

const changeField = (label: string, value: string) => {
  act(() => {
    fireEvent.change(screen.getByRole("textbox", { name: label, exact: true }), { target: { value } });
  });
};

const save = () => {
  act(() => {
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }));
  });
};

const openRules = () => {
  act(() => {
    fireEvent.click(screen.getByRole("button", { name: "Rate limits", exact: true }));
  });
  act(() => {
    fireEvent.click(screen.getByRole("button", { name: "Fallbacks", exact: true }));
  });
};

describe("EditCustomerDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    savedRequests = [];
    rejectBudget = false;
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockImplementation(async (input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      const path = new URL(request.url).pathname;
      if (request.method === "GET" && path === "/models") {
        return jsonResponse({ data: [{ id: "primary" }, { id: "backup" }] });
      }
      if (request.method === "POST") {
        const body: unknown = await request.json();
        savedRequests.push({ path, body });
        if (path === "/budget/update" || path === "/budget/new") {
          return rejectBudget
            ? jsonResponse({ error: "Budget could not be saved" }, 400)
            : jsonResponse({ budget_id: path === "/budget/new" ? "new-budget" : "budget-1" });
        }
        if (path === "/customer/update") return jsonResponse(customer);
      }
      return jsonResponse({ error: "Unexpected request" }, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("should initially show compact optional sections", () => {
    renderDialog();
    expect(screen.getByRole("heading", { name: "Edit customer" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Rate limits", exact: true })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: "Fallbacks", exact: true })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("textbox", { name: "Tokens per minute", exact: true })).not.toBeInTheDocument();
  });

  it("should render the customer's overall and model group limits when expanded", async () => {
    renderDialog();
    openRules();

    expect(screen.getByRole("heading", { name: "Edit customer" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Tokens per minute", exact: true })).toHaveValue("1,000");
    expect(screen.getByRole("textbox", { name: "Requests per minute", exact: true })).toHaveValue("60");
    expect(screen.getByRole("textbox", { name: "Tokens per minute for primary" })).toHaveValue("500");
    expect(screen.getByRole("textbox", { name: "Requests per minute for primary" })).toHaveValue("10");
    await waitFor(() => expect(screen.getByRole("button", { name: "Add model group limit" })).toBeEnabled());
  });

  it("should save customer and model group limits through the budget endpoint before updating the customer", async () => {
    renderDialog();
    openRules();
    changeField("Tokens per minute", "2000");
    changeField("Requests per minute", "30");
    changeField("Tokens per minute for primary", "300");
    changeField("Requests per minute for primary", "0");
    save();

    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(savedRequests).toEqual([
      {
        path: "/budget/update",
        body: {
          budget_id: "budget-1",
          max_budget: 25,
          tpm_limit: 2000,
          rpm_limit: 30,
          model_tpm_limit: { primary: 300 },
          model_rpm_limit: { primary: 0 },
        },
      },
      {
        path: "/customer/update",
        body: { user_id: "customer-1", alias: "Customer one", blocked: false, budget_id: "budget-1", fallbacks: null },
      },
    ]);
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("should clear blank overall limits and removed model group limits", async () => {
    renderDialog();
    openRules();
    changeField("Tokens per minute", "");
    changeField("Requests per minute", "");
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Remove primary limit" }));
    });
    save();

    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(savedRequests[0]).toEqual({
      path: "/budget/update",
      body: {
        budget_id: "budget-1",
        max_budget: 25,
        tpm_limit: null,
        rpm_limit: null,
        model_tpm_limit: {},
        model_rpm_limit: {},
      },
    });
  });

  it("should create and attach a budget when a customer without one receives a rate limit", async () => {
    renderDialog({ ...customer, budget_id: null, litellm_budget_table: null });
    openRules();
    changeField("Tokens per minute", "2000");
    save();

    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(savedRequests).toEqual([
      {
        path: "/budget/new",
        body: {
          max_budget: null,
          tpm_limit: 2000,
          rpm_limit: null,
          model_tpm_limit: {},
          model_rpm_limit: {},
        },
      },
      {
        path: "/customer/update",
        body: {
          user_id: "customer-1",
          alias: "Customer one",
          blocked: false,
          budget_id: "new-budget",
          fallbacks: null,
        },
      },
    ]);
  });

  it("should keep the edit dialog open and skip the customer update when saving its budget fails", async () => {
    rejectBudget = true;
    renderDialog();
    openRules();
    changeField("Tokens per minute", "2000");
    save();

    await waitFor(() => expect(screen.getByRole("button", { name: "Save", exact: true })).toBeEnabled());
    expect(savedRequests.map((request) => request.path)).toEqual(["/budget/update"]);
    expect(screen.getByRole("heading", { name: "Edit customer" })).toBeInTheDocument();
    expect(onSaved).not.toHaveBeenCalled();
    expect(onOpenChange).not.toHaveBeenCalled();
  });

  it("should retain the customer's ordered fallback chain when changing its limits", async () => {
    renderDialog({ ...customer, fallbacks: [{ primary: ["reserve", "backup"] }] });
    openRules();
    expect(screen.getByRole("combobox", { name: "Fallback 1 for primary" })).toHaveValue("reserve");
    changeField("Tokens per minute", "2000");
    save();
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(savedRequests[1]).toMatchObject({
      path: "/customer/update",
      body: { fallbacks: [{ primary: ["reserve", "backup"] }] },
    });
  });

  it("should restore inherited fallbacks when gateway defaults are selected", async () => {
    renderDialog({ ...customer, fallbacks: [{ primary: ["backup"] }] });
    openRules();
    act(() => {
      fireEvent.click(screen.getByRole("switch", { name: "Use gateway defaults" }));
    });
    save();
    await waitFor(() => expect(onSaved).toHaveBeenCalledOnce());
    expect(savedRequests[1]).toMatchObject({ path: "/customer/update", body: { fallbacks: null } });
  });
});

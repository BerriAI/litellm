import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { chooseSelectOption } from "../../../../../tests/test-utils";
import CreateCustomerDialog from "./CreateCustomerDialog";

const fetchMock = vi.fn<typeof fetch>();
const onCreated = vi.fn();
const onOpenChange = vi.fn();
let createdBodies: unknown[] = [];
let modelsAvailable = true;
let modelGroups = ["primary", "backup"];

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const renderDialog = () =>
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } })
      }
    >
      <CreateCustomerDialog open onOpenChange={onOpenChange} onCreated={onCreated} />
    </QueryClientProvider>,
  );

const changeField = (name: string, value: string) => {
  act(() => {
    fireEvent.change(screen.getByRole("textbox", { name, exact: true }), { target: { value } });
  });
};

describe("CreateCustomerDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createdBodies = [];
    modelsAvailable = true;
    modelGroups = ["primary", "backup"];
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockImplementation(async (input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      const path = new URL(request.url).pathname;
      if (request.method === "GET" && path === "/models") {
        return modelsAvailable
          ? jsonResponse({ data: modelGroups.map((id) => ({ id })) })
          : jsonResponse({ error: "Models unavailable" }, 500);
      }
      if (request.method === "POST" && path === "/customer/new") {
        createdBodies.push(await request.json());
        return jsonResponse({ user_id: "customer-1", blocked: false, spend: 0 });
      }
      return jsonResponse({ error: "Unexpected request" }, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("should create a customer with customer-wide and model group limits", async () => {
    const user = userEvent.setup();
    renderDialog();

    expect(screen.getByRole("heading", { name: "Create customer" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Rate limits", exact: true })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: "Fallbacks", exact: true })).toHaveAttribute("aria-expanded", "false");
    changeField("Customer id", "customer-1");
    await user.click(screen.getByRole("button", { name: "Rate limits", exact: true }));
    changeField("Tokens per minute", "2000");
    changeField("Requests per minute", "20");
    await waitFor(() => expect(screen.getByRole("button", { name: "Add model group limit" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Add model group limit" }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Model group for backup" }), "primary");
    changeField("Tokens per minute for primary", "500");
    changeField("Requests per minute for primary", "0");
    await user.click(screen.getByRole("button", { name: "Add model group limit" }));
    changeField("Requests per minute for backup", "60");
    expect(screen.getByRole("button", { name: "Add model group limit" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Rate limits", exact: true }));
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("customer-1"));
    expect(createdBodies).toEqual([
      {
        user_id: "customer-1",
        alias: null,
        blocked: false,
        max_budget: null,
        budget_duration: null,
        tpm_limit: 2000,
        rpm_limit: 20,
        fallbacks: null,
        model_tpm_limit: { primary: 500 },
        model_rpm_limit: { primary: 0, backup: 60 },
      },
    ]);
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("should show a model loading error while allowing creation without optional model limits", async () => {
    modelsAvailable = false;
    const user = userEvent.setup();
    renderDialog();

    changeField("Customer id", "customer-1");
    await user.click(screen.getByRole("button", { name: "Rate limits", exact: true }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load model groups.");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add model group limit" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("customer-1"));
    expect(createdBodies).toEqual([
      {
        user_id: "customer-1",
        alias: null,
        blocked: false,
        max_budget: null,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        fallbacks: null,
      },
    ]);
  });

  it("should create customer-specific fallbacks in the selected order", async () => {
    modelGroups = ["primary", "backup", "reserve"];
    const user = userEvent.setup();
    renderDialog();

    changeField("Customer id", "customer-1");
    await user.click(screen.getByRole("button", { name: "Fallbacks", exact: true }));
    await user.click(screen.getByRole("switch", { name: "Use gateway defaults" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Add fallback chain" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Add fallback chain" }));
    expect(screen.getByRole("button", { name: "Create", exact: true })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Fallbacks", exact: true }));
    expect(screen.getByRole("status")).toHaveTextContent("Select at least one fallback model group.");
    await user.click(screen.getByRole("button", { name: "Fallbacks", exact: true }));
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Primary model group for backup" }), "primary");
    await user.click(screen.getByRole("button", { name: "Add fallback model" }));
    expect(screen.getByRole("combobox", { name: "Fallback 1 for primary" })).toHaveValue("backup");
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Fallback 1 for primary" }), "reserve");
    await user.click(screen.getByRole("button", { name: "Add fallback model" }));
    expect(screen.getByRole("combobox", { name: "Fallback 2 for primary" })).toHaveValue("backup");
    expect(screen.getByRole("button", { name: "Add fallback model" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Move backup up" }));
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("customer-1"));
    expect(createdBodies).toEqual([
      {
        user_id: "customer-1",
        alias: null,
        blocked: false,
        max_budget: null,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        fallbacks: [{ primary: ["backup", "reserve"] }],
      },
    ]);
  });

  it("should explicitly disable inherited fallbacks when custom rules are empty", async () => {
    const user = userEvent.setup();
    renderDialog();

    changeField("Customer id", "customer-1");
    await user.click(screen.getByRole("button", { name: "Fallbacks", exact: true }));
    await user.click(screen.getByRole("switch", { name: "Use gateway defaults" }));
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("customer-1"));
    expect(createdBodies).toEqual([
      {
        user_id: "customer-1",
        alias: null,
        blocked: false,
        max_budget: null,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        fallbacks: [],
      },
    ]);
  });

  it("should create numeric customer and model limits from shorthand and comma-separated values", async () => {
    const user = userEvent.setup();
    renderDialog();

    changeField("Customer id", "customer-1");
    await user.click(screen.getByRole("button", { name: "Rate limits", exact: true }));
    changeField("Tokens per minute", "100K");
    changeField("Requests per minute", "1.5M");
    await waitFor(() => expect(screen.getByRole("button", { name: "Add model group limit" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Add model group limit" }));
    changeField("Tokens per minute for backup", "2B");
    changeField("Requests per minute for backup", "12,500");
    await user.click(screen.getByRole("button", { name: "Create", exact: true }));

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("customer-1"));
    expect(createdBodies).toEqual([
      {
        user_id: "customer-1",
        alias: null,
        blocked: false,
        max_budget: null,
        budget_duration: null,
        tpm_limit: 100000,
        rpm_limit: 1500000,
        fallbacks: null,
        model_tpm_limit: { backup: 2000000000 },
        model_rpm_limit: { backup: 12500 },
      },
    ]);
  });

  it("should preserve the editing draft, format it on blur and explain invalid limits when collapsed", async () => {
    const user = userEvent.setup();
    renderDialog();

    changeField("Customer id", "customer-1");
    await user.click(screen.getByRole("button", { name: "Rate limits", exact: true }));
    const tokens = screen.getByRole("textbox", { name: "Tokens per minute", exact: true });
    await user.click(tokens);
    changeField("Tokens per minute", "100K");
    expect(tokens).toHaveFocus();
    expect(tokens).toHaveValue("100K");
    expect(screen.getByRole("status")).toHaveTextContent("= 100,000");
    changeField("Tokens per minute", "1.5M");
    expect(tokens).toHaveValue("1.5M");
    expect(screen.getByRole("status")).toHaveTextContent("= 1,500,000");
    await user.tab();
    expect(tokens).toHaveValue("1,500,000");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    await user.click(tokens);
    changeField("Tokens per minute", "10-ish");
    expect(tokens).toHaveValue("10-ish");
    expect(tokens).toHaveAttribute("aria-invalid", "true");
    await user.click(screen.getByRole("button", { name: "Rate limits", exact: true }));

    expect(screen.getByRole("status")).toHaveTextContent("Enter a whole number, or use K, M or B");
    expect(screen.getByRole("button", { name: "Create", exact: true })).toBeDisabled();
    expect(createdBodies).toEqual([]);
  });
});

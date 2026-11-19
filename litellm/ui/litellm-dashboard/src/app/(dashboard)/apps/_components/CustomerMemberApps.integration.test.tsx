import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  renderWithProviders as render,
  screen,
  fireEvent,
  act,
  waitFor,
  testQueryClient,
} from "../../../../../tests/test-utils";
import CustomerMemberApps from "./CustomerMemberApps";
import type { App } from "./appForms";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({ data: { data: [] }, isLoading: false, isError: false }),
}));
const app: App = {
  app_id: "app-1",
  resource_type: "team",
  resource_id: "team-assistant",
  alias: "Assistant",
  spend: 100,
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
};
const fetchMock = vi.fn<typeof fetch>();
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
beforeEach(() => {
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe("customer member apps", () => {
  it("should show a compact member budget row and save an app-specific override", async () => {
    const requests: Request[] = [];
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      requests.push(request.clone());
      return json(
        new URL(request.url).pathname.endsWith("/apps")
          ? [app]
          : {
              app_id: "app-1",
              customer_member_id: "member-1",
              spend: 2.5,
              budget_source: "customer_default",
              effective_budget: { rpm_limit: 1 },
            },
      );
    });
    const user = userEvent.setup();
    render(<CustomerMemberApps memberId="member-1" />);
    expect(await screen.findByText("$2.50")).toBeInTheDocument();
    expect(screen.getByText("Customer member default")).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader")).toHaveLength(3);
    expect(screen.queryByText("$100.00")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Edit member budget for Assistant" }));
    act(() => {
      fireEvent.click(screen.getByRole("checkbox", { name: "Set custom limits" }));
    });
    act(() => {
      fireEvent.change(screen.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "5" } });
    });
    await user.click(screen.getByRole("button", { name: "Save member app budget" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    const request = requests.find((row) => row.method === "PUT");
    expect(new URL(request!.url).pathname).toBe("/management/v1/apps/app-1/customer_members/member-1");
    expect(await request?.json()).toEqual({
      budget: {
        max_budget: 5,
        soft_budget: null,
        tpm_limit: null,
        rpm_limit: null,
        budget_duration: null,
        allowed_models: null,
        budget_limits: null,
      },
    });
  });

  it("should distinguish tiny positive member spend from zero", async () => {
    fetchMock.mockImplementation(async (input) =>
      json(
        new URL((input as Request).url).pathname.endsWith("/apps")
          ? [app]
          : {
              app_id: "app-1",
              customer_member_id: "member-1",
              spend: 0.00001234,
              budget_source: "none",
              effective_budget: null,
            },
      ),
    );
    render(<CustomerMemberApps memberId="member-1" />);
    expect(await screen.findByText("<$0.01")).toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
  });

  it("should identify the primary budget period alongside its inherited source", async () => {
    const member = {
      app_id: "app-1",
      customer_member_id: "member-1",
      spend: 99,
      budget_source: "app_default",
      effective_budget: { max_budget: 10, budget_duration: "1mo" },
      budget_usage: [{ source: "scalar", max_budget: 10, spend: 2, budget_duration: "1mo", reset_at: null }],
    };
    fetchMock.mockImplementation(async (input) =>
      json(new URL((input as Request).url).pathname.endsWith("/apps") ? [app] : member),
    );
    render(<CustomerMemberApps memberId="member-1" />);
    expect(await screen.findByText("App member default · Monthly")).toBeInTheDocument();
    expect(screen.getByText("$2.00 of $10.00 used")).toBeInTheDocument();
    expect(screen.queryByText("$99.00")).not.toBeInTheDocument();
  });

  it("should clear an existing override to restore inherited member defaults", async () => {
    const requests: Request[] = [];
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      requests.push(request.clone());
      return json(
        new URL(request.url).pathname.endsWith("/apps")
          ? [app]
          : {
              app_id: "app-1",
              customer_member_id: "member-1",
              spend: 0,
              budget_override_id: "override-1",
              budget_override: { max_budget: 5 },
              budget_source: "member_override",
              effective_budget: { max_budget: 5 },
            },
      );
    });
    const user = userEvent.setup();
    render(<CustomerMemberApps memberId="member-1" />);
    await user.click(await screen.findByRole("button", { name: "Edit member budget for Assistant" }));
    act(() => {
      fireEvent.click(screen.getByRole("checkbox", { name: "Set custom limits" }));
    });
    await user.click(screen.getByRole("button", { name: "Save member app budget" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    expect(await requests.find((row) => row.method === "PUT")?.json()).toEqual({ budget: null, budget_override: null });
  });

  it.each([false, true])(
    "should initialize and save a permanent budget while an increase is present %s",
    async (legacy) => {
      const requests: Request[] = [];
      fetchMock.mockImplementation(async (input) => {
        const request = input as Request;
        requests.push(request.clone());
        return json(
          new URL(request.url).pathname.endsWith("/apps")
            ? [app]
            : {
                app_id: "app-1",
                customer_member_id: "member-1",
                spend: 0,
                budget: legacy ? null : { max_budget: 5 },
                budget_override: legacy ? { max_budget: 5 } : null,
                effective_budget: { max_budget: 20 },
              },
        );
      });
      const user = userEvent.setup();
      render(<CustomerMemberApps memberId="member-1" />);
      await user.click(await screen.findByRole("button", { name: "Edit member budget for Assistant" }));
      expect(screen.getByRole("spinbutton", { name: "Spend limit (USD)" })).toHaveValue(5);
      fireEvent.change(screen.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "7" } });
      await user.click(screen.getByRole("button", { name: "Save member app budget" }));
      await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
      expect(await requests.find((request) => request.method === "PUT")?.json()).toEqual({
        budget: {
          max_budget: 7,
          soft_budget: null,
          tpm_limit: null,
          rpm_limit: null,
          budget_duration: null,
          allowed_models: null,
          budget_limits: null,
        },
        ...(legacy ? { budget_override: null } : {}),
      });
    },
  );

  it("should not enable overwriting a member budget after a read failure", async () => {
    fetchMock.mockImplementation(async (input) =>
      new URL((input as Request).url).pathname.endsWith("/apps")
        ? json([app])
        : json({ detail: "Member budgets unavailable" }, 503),
    );
    render(<CustomerMemberApps memberId="member-1" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Member budgets unavailable");
    expect(screen.queryByRole("button", { name: "Edit member budget for Assistant" })).not.toBeInTheDocument();
  });

  it("should show an empty state when there are no apps", async () => {
    fetchMock.mockResolvedValue(json([]));
    render(<CustomerMemberApps memberId="member-1" />);
    expect(await screen.findByText("No app budgets are configured yet.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});

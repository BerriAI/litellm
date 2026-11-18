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
  it("should show member spend rather than app totals and saves an app-specific override", async () => {
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
    expect(screen.queryByText("$100.00")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Edit member budget for Assistant" }));
    act(() => {
      fireEvent.click(screen.getByRole("checkbox", { name: "Set a budget and limits" }));
    });
    act(() => {
      fireEvent.change(screen.getByRole("spinbutton", { name: "Spend limit (USD)" }), { target: { value: "5" } });
    });
    await user.click(screen.getByRole("button", { name: "Save member app budget" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    const request = requests.find((row) => row.method === "PUT");
    expect(new URL(request!.url).pathname).toBe("/management/v1/apps/app-1/customer_members/member-1");
    expect(await request?.json()).toMatchObject({ budget_override: { max_budget: 5 } });
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
              budget_source: "member_override",
              effective_budget: { max_budget: 5 },
            },
      );
    });
    const user = userEvent.setup();
    render(<CustomerMemberApps memberId="member-1" />);
    await user.click(await screen.findByRole("button", { name: "Edit member budget for Assistant" }));
    act(() => {
      fireEvent.click(screen.getByRole("checkbox", { name: "Set a budget and limits" }));
    });
    await user.click(screen.getByRole("button", { name: "Save member app budget" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    expect(await requests.find((row) => row.method === "PUT")?.json()).toEqual({ budget_override: null });
  });

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

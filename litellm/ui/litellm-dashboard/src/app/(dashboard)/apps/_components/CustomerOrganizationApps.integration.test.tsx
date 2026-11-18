import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import {
  renderWithProviders as render,
  screen,
  fireEvent,
  waitFor,
  within,
  chooseSelectOption,
  testQueryClient,
} from "../../../../../tests/test-utils";
import CustomerOrganizationApps from "./CustomerOrganizationApps";
import type { App } from "./appForms";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({ data: { data: [] }, isLoading: false, isError: false }),
}));
const app: App = {
  app_id: "app-1",
  alias: "Assistant",
  spend: 0,
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
  tiers: [{ budget_id: "tier-1", name: "Standard", max_budget: 10 }],
  tier_budget_ids: ["tier-1"],
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

describe("customer organization apps", () => {
  it("assigns a first-use tier with a per-member RPM default after a missing association", async () => {
    const requests: Request[] = [];
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      requests.push(request.clone());
      if (new URL(request.url).pathname.endsWith("/apps")) return json([app]);
      if (request.method === "PUT") return json({ app_id: "app-1", customer_org_id: "org-1", spend: 0 });
      return json({ detail: "Not yet assigned" }, 404);
    });
    const user = userEvent.setup();
    render(<CustomerOrganizationApps organizationId="org-1" />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Select an app" }), "Assistant");
    await chooseSelectOption(user, await screen.findByRole("combobox", { name: "Tier" }), "Standard");
    const member = within(screen.getByRole("group", { name: "Per-member default for this customer" }));
    fireEvent.click(member.getByRole("checkbox", { name: "Set a budget and limits" }));
    fireEvent.change(member.getByRole("spinbutton", { name: "Requests per minute" }), { target: { value: "1" } });
    await user.click(screen.getByRole("button", { name: "Save app settings" }));
    await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
    const request = requests.find((row) => row.method === "PUT");
    expect(new URL(request!.url).pathname).toBe("/management/v1/apps/app-1/customer_organizations/org-1");
    expect(await request?.json()).toMatchObject({
      tier_budget_id: "tier-1",
      budget_override: null,
      customer_member_budget: { rpm_limit: 1 },
    });
  });

  it("does not treat server failures as a new uncapped association", async () => {
    fetchMock.mockImplementation(async (input) =>
      new URL((input as Request).url).pathname.endsWith("/apps")
        ? json([app])
        : json({ detail: "Association unavailable" }, 503),
    );
    const user = userEvent.setup();
    render(<CustomerOrganizationApps organizationId="org-1" />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Select an app" }), "Assistant");
    expect(await screen.findByRole("alert")).toHaveTextContent("Association unavailable");
    expect(screen.queryByRole("button", { name: "Save app settings" })).not.toBeInTheDocument();
  });
});

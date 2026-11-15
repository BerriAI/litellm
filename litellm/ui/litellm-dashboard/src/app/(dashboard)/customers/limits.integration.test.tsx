import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderWithProviders, screen, testQueryClient } from "../../../../tests/test-utils";
import CustomersLayout from "./layout";

beforeEach(() => testQueryClient.clear());
afterEach(() => {
  testQueryClient.clear();
  vi.unstubAllGlobals();
});

describe("Customer portal limits", () => {
  it.each([
    { limitsApply: true, reached: false },
    { limitsApply: true, reached: true },
    { limitsApply: false, reached: true },
    { limitsApply: false, reached: false },
  ])("shows the notices for limits_apply=$limitsApply and reached=$reached", async ({ limitsApply, reached }) => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (request: Request) => {
        expect(new URL(request.url).pathname).toBe("/management/v1/customer_portal/limits");
        const limits = {
          limits_apply: limitsApply,
          organization_limit: 5,
          teams_per_organization_limit: 5,
          members_per_team_limit: 5,
          organization_count: reached ? 5 : 1,
          any_limit_reached: reached,
        };
        return Response.json(limits);
      }),
    );
    renderWithProviders(
      <CustomersLayout>
        <h1>Customers</h1>
      </CustomersLayout>,
    );
    await act(async () => {
      await testQueryClient.ensureQueryData({ queryKey: ["get", "/management/v1/customer_portal/limits", {}] });
    });
    if (limitsApply) {
      expect(await screen.findByText(/Free tier: up to 5 customers/)).toHaveTextContent(
        "Free tier: up to 5 customers, 5 teams per customer, 5 members per team. LiteLLM Enterprise removes these limits.",
      );
      expect(screen.getByRole("link", { name: "Enterprise" })).toHaveAttribute(
        "href",
        "https://www.litellm.ai/enterprise",
      );
    } else {
      expect(screen.queryByText(/Free tier:/)).not.toBeInTheDocument();
    }
    if (limitsApply && reached) {
      expect(await screen.findByRole("alert")).toHaveTextContent("Free customer portal limit reached");
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Requests from new customers or members still go through but are not tracked",
      );
      expect(screen.getByRole("link", { name: "Upgrade to Enterprise" })).toHaveAttribute(
        "href",
        "https://www.litellm.ai/enterprise",
      );
    } else {
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    }
  });

  it.each([204, 503])("keeps the page available without notices when limits return %s", async (status) => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(null, { status })),
    );
    renderWithProviders(
      <CustomersLayout>
        <h1>Customers</h1>
      </CustomersLayout>,
    );
    await act(async () => {
      await testQueryClient
        .ensureQueryData({ queryKey: ["get", "/management/v1/customer_portal/limits", {}] })
        .catch(() => undefined);
    });
    expect(screen.getByRole("heading", { name: "Customers" })).toBeInTheDocument();
    expect(screen.queryByText(/Free tier:/)).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

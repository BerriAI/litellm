import type { UserEvent } from "@testing-library/user-event";
import { expect, vi } from "vitest";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { jsonResponse, screen, testQueryClient, waitFor } from "../../../../../tests/test-utils";
import type { App } from "./appForms";

export { chooseSelectOption as chooseAppOption } from "../../../../../tests/test-utils";

export const appFixture = (overrides: Partial<App> = {}): App => ({
  resource_type: "team",
  resource_id: "team-chat",
  app_id: "app-chat",
  alias: "chat",
  spend: 0,
  auto_create_customers: true,
  created_at: "",
  updated_at: "",
  created_by: "",
  updated_by: "",
  ...overrides,
});

export function mockAppRequests(routes: Record<string, (request: Request) => unknown>) {
  const requests: Request[] = [];
  const writes: Request[] = [];
  testQueryClient.clear();
  registerBaseUrlGetter(() => "http://localhost");
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async (input, init) => {
      const request = new Request(input, init);
      requests.push(request.clone());
      if (request.method !== "GET") writes.push(request.clone());
      const route = `${request.method} ${new URL(request.url).pathname}`;
      const handler = routes[route];
      if (!handler) throw new Error(`Unexpected request ${route}`);
      const response = await handler(request);
      return response instanceof Response ? response : jsonResponse(response);
    }),
  );
  return { requests, writes, routes };
}

export const appBudgetPolicyCases = ["custom", "legacy", "temporary"].map((source) => {
  const legacy = source === "legacy";
  const temporary = source === "temporary";
  return {
    source,
    legacy,
    temporary,
    policy: {
      budget: legacy ? null : { max_budget: 5, budget_duration: "1mo" },
      budget_override: legacy || temporary ? { max_budget: legacy ? 5 : 20, budget_duration: "1mo" } : null,
      budget_override_expires_at: temporary ? "2027-01-01T00:00:00Z" : null,
    },
    expectedPatch: {
      budget: {
        max_budget: 7,
        budget_duration: "1mo",
        ...(legacy
          ? { soft_budget: null, tpm_limit: null, rpm_limit: null, allowed_models: null, budget_limits: null }
          : {}),
      },
      ...(legacy ? { budget_override: null } : {}),
    },
  };
});

export async function saveAppBudget(user: UserEvent, requests: readonly Request[], name: string): Promise<unknown> {
  await user.click(screen.getByRole("button", { name }));
  await waitFor(() => expect(requests.some((request) => request.method === "PUT")).toBe(true));
  return requests
    .find((request) => request.method === "PUT")!
    .clone()
    .json();
}

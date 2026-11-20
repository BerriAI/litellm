import type { UserEvent } from "@testing-library/user-event";
import { expect } from "vitest";
import { screen, waitFor } from "../../../../../tests/test-utils";

export { chooseSelectOption as chooseAppOption } from "../../../../../tests/test-utils";

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

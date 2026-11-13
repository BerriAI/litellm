import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { budgetTestNetwork, jsonResponse, type BudgetRequest } from "@/../tests/budgetTestNetwork";
import { chooseSelectOption } from "@/../tests/test-utils";
import type { BudgetAssignment } from "../../hooks/budgets/budgetProfileTypes";
import { localDateTime } from "./budgetAssignmentPayload";
import BudgetAssignmentModal from "./BudgetAssignmentModal";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");
afterEach(() => vi.unstubAllGlobals());

const future = new Date(Date.now() + 30 * 86400_000).toISOString();
const past = new Date(Date.now() - 30 * 86400_000).toISOString();
const assignment: BudgetAssignment = {
  subject_type: "user",
  subject_id: "person-a",
  budget_id: "standard",
  revision: "original",
  budget_override: { budget_limits: [{ budget_duration: "1mo", max_budget: 20 }] },
  budget_override_expires_at: null,
  effective_budget: {
    budget_limits: [
      { budget_duration: "1mo", max_budget: 20 },
      { budget_duration: "12mo", max_budget: 200 },
    ],
  },
  windows: [
    { budget_duration: "1mo", max_budget: 20, reset_at: future, cap_policy: "pause_until_reset" },
    { budget_duration: "12mo", max_budget: 200, reset_at: future, cap_policy: "pause_until_reset" },
  ],
};

function setup(handler: (request: BudgetRequest) => Response | Promise<Response> = () => jsonResponse(assignment)) {
  const network = budgetTestNetwork((request) =>
    request.path === "/budget/list"
      ? jsonResponse([
          { budget_id: "standard", is_profile: true },
          { budget_id: "premium", is_profile: true },
          { budget_id: "shared", is_profile: false },
        ])
      : handler(request),
  );
  render(
    <QueryClientProvider client={network.client}>
      <BudgetAssignmentModal open onOpenChange={() => {}} />
    </QueryClientProvider>,
  );
  return network;
}

async function loadPerson(id = "person-a") {
  fireEvent.change(screen.getByLabelText("Person ID"), { target: { value: id } });
  await userEvent.click(screen.getByRole("button", { name: "Load person" }));
  return screen.findByLabelText("Personal monthly cap (USD)");
}
const writes = (requests: BudgetRequest[]) => requests.filter((request) => request.method === "PATCH");

describe("BudgetAssignmentModal", () => {
  it("saves a temporary annual increase against an existing annual cap without resending personal caps", async () => {
    const saved = {
      ...assignment,
      revision: "saved",
      budget_override: {
        ...assignment.budget_override,
        temp_budget_increase: 15,
        temp_budget_duration: "12mo",
        temp_budget_expiry: future,
      },
      windows: assignment.windows.map((window) =>
        window.budget_duration === "12mo" ? { ...window, max_budget: 215 } : window,
      ),
    };
    const { requests } = setup((request) => jsonResponse(request.method === "PATCH" ? saved : assignment));
    const user = userEvent.setup();
    await loadPerson();
    fireEvent.change(screen.getByLabelText("Increase amount (USD)"), { target: { value: "15" } });
    fireEvent.change(screen.getByLabelText("Increase expires"), { target: { value: localDateTime(future) } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Applies to" }), "Annual");
    await user.click(screen.getByRole("button", { name: "Save assignment" }));
    await waitFor(() => expect(writes(requests)).toHaveLength(1));
    expect(writes(requests)[0]).toEqual({
      path: "/management/v1/budget_assignments/user/person-a",
      method: "PATCH",
      body: {
        revision: "original",
        budget_override: {
          temp_budget_increase: 15,
          temp_budget_expiry: new Date(localDateTime(future)).toISOString(),
          temp_budget_duration: "12mo",
        },
      },
    });
    const effective = screen.getByRole("region", { name: "Saved effective limits" });
    expect(await within(effective).findByText(/\$215/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Save assignment" })).toBeDisabled();
  });

  it("assigns a selected profile to a customer member and leaves shared budgets out of profile selection", async () => {
    const { requests } = setup(() =>
      jsonResponse({ ...assignment, subject_type: "customer_member", subject_id: "member-a" }),
    );
    const user = userEvent.setup();
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Person type" }), "Customer member");
    await loadPerson("member-a");
    const profile = screen.getByRole("combobox", { name: "Assigned profile" });
    await waitFor(() => expect(profile).toBeEnabled());
    await user.click(profile);
    expect(screen.queryByRole("option", { name: "shared" })).not.toBeInTheDocument();
    await user.clear(profile);
    await user.type(profile, "premium");
    await user.click(await screen.findByRole("option", { name: "premium" }));
    await user.click(screen.getByRole("button", { name: "Save assignment" }));
    await waitFor(() => expect(writes(requests)[0]?.body).toEqual({ revision: "original", budget_id: "premium" }));
    expect(writes(requests)[0].path).toBe("/management/v1/budget_assignments/customer_member/member-a");
  });

  it("keeps a dirty draft and its revision on refresh, then requires explicit reload after a conflict", async () => {
    const newer = {
      ...assignment,
      revision: "newer",
      budget_override: { budget_limits: [{ budget_duration: "1mo" as const, max_budget: 40 }] },
    };
    const replies = [assignment, newer, newer];
    const { client, requests } = setup((request) =>
      request.method === "PATCH"
        ? jsonResponse({ detail: "Revision conflict" }, 409)
        : jsonResponse(replies.shift() ?? newer),
    );
    const monthly = await loadPerson();
    fireEvent.change(monthly, { target: { value: "0" } });
    await act(async () => {
      await client.invalidateQueries({ queryKey: ["budget-assignment"] });
    });
    expect(screen.getByLabelText("Personal monthly cap (USD)")).toHaveValue(0);
    await userEvent.click(screen.getByRole("button", { name: "Save assignment" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This budget changed since you loaded it.");
    expect(writes(requests)[0].body).toEqual({
      revision: "original",
      budget_override: { budget_limits: [{ budget_duration: "1mo", max_budget: 0 }] },
    });
    await userEvent.click(screen.getByRole("button", { name: "Discard edits and reload" }));
    await waitFor(() => expect(screen.getByLabelText("Personal monthly cap (USD)")).toHaveValue(40));
    expect(screen.getByRole("button", { name: "Save assignment" })).toBeDisabled();
  });

  it("does not apply a saved response to another person selected while the request is pending", async () => {
    const response = Promise.withResolvers<Response>();
    const other = {
      ...assignment,
      subject_id: "person-b",
      revision: "other",
      budget_override: { budget_limits: [{ budget_duration: "1mo" as const, max_budget: 80 }] },
    };
    const { requests } = setup((request) =>
      request.method === "PATCH"
        ? response.promise
        : jsonResponse(request.path.endsWith("person-b") ? other : assignment),
    );
    fireEvent.change(await loadPerson(), { target: { value: "30" } });
    await userEvent.click(screen.getByRole("button", { name: "Save assignment" }));
    await waitFor(() => expect(writes(requests)).toHaveLength(1));
    await loadPerson("person-b");
    await act(async () =>
      response.resolve(
        jsonResponse({
          ...assignment,
          revision: "saved",
          budget_override: { budget_limits: [{ budget_duration: "1mo", max_budget: 30 }] },
        }),
      ),
    );
    expect(screen.getByLabelText("Personal monthly cap (USD)")).toHaveValue(80);
    expect(screen.getByRole("button", { name: "Save assignment" })).toBeDisabled();
  });

  it("ignores a late read after the selected person changes", async () => {
    const response = Promise.withResolvers<Response>();
    const other = {
      ...assignment,
      subject_id: "person-b",
      revision: "other",
      budget_override: {
        budget_limits: [{ budget_duration: "1mo" as const, max_budget: 80 }],
      },
    };
    const { requests } = setup((request) =>
      request.path.endsWith("person-a") ? response.promise : jsonResponse(other),
    );
    fireEvent.change(screen.getByLabelText("Person ID"), { target: { value: "person-a" } });
    await userEvent.click(screen.getByRole("button", { name: "Load person" }));
    await waitFor(() => expect(requests.some((request) => request.path.endsWith("person-a"))).toBe(true));
    await loadPerson("person-b");
    await act(async () => response.resolve(jsonResponse(assignment)));
    expect(screen.getByLabelText("Personal monthly cap (USD)")).toHaveValue(80);
    expect(screen.getByLabelText("Person ID")).toHaveValue("person-b");
  });

  it("displays expired overrides separately from expired increases and reactivates only the whole override expiry", async () => {
    const expired = {
      ...assignment,
      budget_override_expires_at: past,
      budget_override: {
        ...assignment.budget_override,
        temp_budget_increase: 5,
        temp_budget_duration: "1mo",
        temp_budget_expiry: past,
      },
    };
    const { requests } = setup(() => jsonResponse(expired));
    await loadPerson();
    expect(screen.getByText("Personal overrides have expired. The assigned profile is active.")).toBeVisible();
    expect(screen.getByText("This temporary increase has expired. It is stored but adds no capacity.")).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Reactivate indefinitely" }));
    await userEvent.click(screen.getByRole("button", { name: "Save assignment" }));
    await waitFor(() =>
      expect(writes(requests)[0]?.body).toEqual({ revision: "original", budget_override_expires_at: null }),
    );
  });

  it("clears personal overrides and their expiry explicitly", async () => {
    const { requests } = setup();
    await loadPerson();
    await userEvent.click(screen.getByRole("button", { name: "Clear all personal overrides" }));
    expect(screen.getByText("All personal overrides will be removed when you save.")).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Save assignment" }));
    await waitFor(() =>
      expect(writes(requests)[0]?.body).toEqual({
        revision: "original",
        budget_override: null,
        budget_override_expires_at: null,
      }),
    );
  });
});

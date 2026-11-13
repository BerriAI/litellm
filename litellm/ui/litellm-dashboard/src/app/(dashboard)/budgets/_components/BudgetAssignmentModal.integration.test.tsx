import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { chooseSelectOption } from "../../../../../tests/test-utils";
import BudgetAssignmentModal from "./BudgetAssignmentModal";

const { get, patch } = vi.hoisted(() => ({ get: vi.fn(), patch: vi.fn() }));
vi.mock("@/components/networking", () => ({ apiClient: { get, patch } }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token", userRole: "Admin" }),
}));

const assignment = {
  subject_type: "user",
  subject_id: "person",
  budget_template_id: "standard",
  budget_override: null,
  budget_override_expires_at: null,
  effective_budget: { budget_limits: [{ budget_duration: "1mo", max_budget: 100 }] },
  windows: [{ budget_duration: "1mo", max_budget: 100 }],
};

beforeEach(() => {
  vi.clearAllMocks();
  get.mockResolvedValue(assignment);
  patch.mockResolvedValue(assignment);
});

function renderEditor() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <BudgetAssignmentModal open onOpenChange={vi.fn()} />
    </QueryClientProvider>,
  );
  return userEvent.setup();
}

it("saving inherited limits does not create personal overrides", async () => {
  const user = renderEditor();
  fireEvent.change(screen.getByLabelText("Person ID"), { target: { value: "person" } });
  await user.click(screen.getByRole("button", { name: "Load budget" }));
  await user.click(await screen.findByRole("button", { name: "Save person budget" }));
  await waitFor(() =>
    expect(patch).toHaveBeenCalledWith("/management/v1/budget_assignments/user/person", {
      accessToken: "test-token",
      body: {},
    }),
  );
});

it("sends a customer member's direct increase with its target window and UTC expiry", async () => {
  const user = renderEditor();
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "Person type" }), "Customer member");
  fireEvent.change(screen.getByLabelText("Person ID"), { target: { value: "member" } });
  await user.click(screen.getByRole("button", { name: "Load budget" }));
  fireEvent.change(await screen.findByLabelText("Personal monthly limit (USD)"), { target: { value: "50" } });
  fireEvent.change(screen.getByLabelText("Profile ID"), { target: { value: "power-profile" } });
  fireEvent.change(screen.getByLabelText("Temporary increase (USD)"), { target: { value: "25" } });
  fireEvent.change(screen.getByLabelText("Increase expires (UTC)"), { target: { value: "2028-03-01T12:00" } });
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "Increase applies to" }), "Annual limit");
  await user.click(screen.getByRole("button", { name: "Save person budget" }));
  await waitFor(() =>
    expect(patch).toHaveBeenCalledWith("/management/v1/budget_assignments/customer_member/member", {
      accessToken: "test-token",
      body: {
        budget_id: "power-profile",
        budget_override: {
          budget_limits: [{ budget_duration: "1mo", max_budget: 50 }],
          temp_budget_increase: 25,
          temp_budget_duration: "12mo",
          temp_budget_expiry: "2028-03-01T12:00:00.000Z",
        },
      },
    }),
  );
});

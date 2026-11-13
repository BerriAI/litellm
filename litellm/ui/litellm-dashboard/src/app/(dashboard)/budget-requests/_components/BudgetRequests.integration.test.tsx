import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient, chooseSelectOption } from "../../../../../tests/test-utils";
import BudgetRequests from "./BudgetRequests";

const { get, post, put } = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), put: vi.fn() }));
vi.mock("@/components/networking", () => ({ apiClient: { get, post, put } }));
const path = "/management/v1/budget_increase_requests";
const row = {
  id: "request-1",
  submitted_by: "requester",
  amount: 50,
  duration: "1mo",
  expires_at: "2028-03-01T12:00:00Z",
  reason: "Finish report",
  status: "pending_review",
  created_at: "2028-02-01T12:00:00Z",
};
const assignment = {
  effective_budget: { cap_policy: "pause_allow_increase" },
  windows: [
    { budget_duration: "1mo", max_budget: 100 },
    { budget_duration: "12mo", max_budget: 500 },
  ],
};
beforeEach(() => {
  testQueryClient.clear();
  vi.clearAllMocks();
  get.mockImplementation(async (url: string) => {
    if (url === path) return { requests: [row], has_more: false };
    if (url === path + "/request-1") return row;
    return assignment;
  });
  post.mockResolvedValue(row);
  put.mockResolvedValue({ ...row, status: "approved" });
});

it("submits own annual increase with an absolute UTC expiry and reason", async () => {
  const user = userEvent.setup();
  renderWithProviders(
    <BudgetRequests accessToken="test" userRole="internal_user" userId="requester" isViewOnly={false} />,
  );
  await user.click(await screen.findByRole("button", { name: "Request increase" }));
  fireEvent.change(await screen.findByLabelText("Additional budget (USD)"), { target: { value: "75" } });
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "Applies to" }), "Annual limit");
  fireEvent.change(screen.getByLabelText("Expires at (UTC)"), { target: { value: "2028-03-01T12:00" } });
  fireEvent.change(screen.getByLabelText("Reason for request"), { target: { value: "Finish the migration" } });
  await user.click(screen.getByRole("button", { name: "Submit request" }));
  await waitFor(() =>
    expect(post).toHaveBeenCalledWith(path, {
      accessToken: "test",
      body: { amount: 75, duration: "12mo", expires_at: "2028-03-01T12:00:00.000Z", reason: "Finish the migration" },
    }),
  );
});

it("defaults to the configured annual limit when no monthly limit exists", async () => {
  get.mockImplementation(async (url: string) =>
    url === path ? { requests: [], has_more: false } : { ...assignment, windows: [assignment.windows[1]] },
  );
  const user = userEvent.setup();
  renderWithProviders(
    <BudgetRequests accessToken="test" userRole="internal_user" userId="requester" isViewOnly={false} />,
  );
  await user.click(await screen.findByRole("button", { name: "Request increase" }));
  expect(await screen.findByRole("combobox", { name: "Applies to" })).toHaveTextContent("Annual limit");
  fireEvent.change(screen.getByLabelText("Expires at (UTC)"), { target: { value: "2028-03-01T12:00" } });
  fireEvent.change(screen.getByLabelText("Reason for request"), { target: { value: "Finish annual work" } });
  await user.click(screen.getByRole("button", { name: "Submit request" }));
  await waitFor(() =>
    expect(post).toHaveBeenCalledWith(path, {
      accessToken: "test",
      body: { amount: 50, duration: "12mo", expires_at: "2028-03-01T12:00:00.000Z", reason: "Finish annual work" },
    }),
  );
});

it("lets an admin review the requested amount and sends the rejection explanation", async () => {
  const user = userEvent.setup();
  renderWithProviders(
    <BudgetRequests accessToken="test" userRole="proxy_admin" userId="reviewer" isViewOnly={false} />,
  );
  await user.click(await screen.findByRole("button", { name: "Review request" }));
  expect(await screen.findByText("Current personal limits")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Reason for rejection (required to reject)"), {
    target: { value: "Use the remaining annual capacity" },
  });
  await user.click(screen.getByRole("button", { name: "Reject", exact: true }));
  await waitFor(() =>
    expect(put).toHaveBeenCalledWith(path + "/request-1/reject", {
      accessToken: "test",
      body: { reason: "Use the remaining annual capacity" },
    }),
  );
});

it("keeps an approval conflict visible without claiming success", async () => {
  put.mockRejectedValue(new Error("A temporary increase is already active."));
  const user = userEvent.setup();
  renderWithProviders(
    <BudgetRequests accessToken="test" userRole="proxy_admin" userId="reviewer" isViewOnly={false} />,
  );
  await user.click(await screen.findByRole("button", { name: "Review request" }));
  await screen.findByText("Current personal limits");
  await user.click(screen.getByRole("button", { name: "Approve increase" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("A temporary increase is already active.");
  expect(screen.getByRole("dialog")).toBeInTheDocument();
});

it.each(["proxy_admin_viewer", "Admin"])("does not expose decision controls to read-only %s sessions", async (role) => {
  const user = userEvent.setup();
  renderWithProviders(<BudgetRequests accessToken="test" userRole={role} userId="viewer" isViewOnly />);
  await user.click(await screen.findByRole("button", { name: "Request details" }));
  await screen.findByText("Requested by");
  expect(screen.queryByRole("button", { name: "Approve increase" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Request increase" })).not.toBeInTheDocument();
});

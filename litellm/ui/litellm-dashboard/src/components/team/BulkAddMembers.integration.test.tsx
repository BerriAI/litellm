import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders as render } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "@/components/networking";
import BulkAddMembers from "./BulkAddMembers";

vi.mock("@/components/networking", () => ({ apiClient: { post: vi.fn() } }));

const onAdded = vi.fn();
const openImport = () => {
  render(
    <BulkAddMembers
      accessToken="team-admin-token"
      teamId="course"
      members={[{ user_email: "old@example.com", user_id: "old", role: "admin" }]}
      onAdded={onAdded}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Bulk add members" }));
};

beforeEach(() => vi.clearAllMocks());

describe("BulkAddMembers", () => {
  it("sends new members and their shared role and zero budget, then shows individual outcomes", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({
      results: [
        { user_email: "new@example.com", success: true },
        { user_email: "failed@example.com", success: false, error: "Membership limit reached" },
      ],
    });
    openImport();
    fireEvent.change(screen.getByLabelText(/Paste emails/), {
      target: { value: "new@example.com, old@example.com, NEW@example.com, invalid, failed@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Role for all new members"), { target: { value: "admin" } });
    fireEvent.change(screen.getByLabelText(/Member budget/), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "Confirm adding 2 members" }));
    expect(await screen.findByText("new@example.com Added")).toBeInTheDocument();
    expect(screen.getByText("failed@example.com Failed Membership limit reached")).toBeInTheDocument();
    expect(apiClient.post).toHaveBeenCalledExactlyOnceWith("/team/bulk_member_add", {
      accessToken: "team-admin-token",
      body: {
        team_id: "course",
        members: [
          { user_email: "new@example.com", role: "admin" },
          { user_email: "failed@example.com", role: "admin" },
        ],
        max_budget_in_team: 0,
      },
    });
    expect(onAdded).toHaveBeenCalledOnce();
    expect(screen.getByRole("button", { name: "Confirm adding 2 members" })).toBeDisabled();
  });

  it("keeps input and shows a failed request without claiming success", async () => {
    vi.mocked(apiClient.post).mockRejectedValue(new Error("Team admin permission required"));
    openImport();
    fireEvent.change(screen.getByLabelText(/Paste emails/), { target: { value: "new@example.com" } });
    fireEvent.click(screen.getByRole("button", { name: "Confirm adding 1 members" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Team admin permission required");
    expect(screen.getByLabelText(/Paste emails/)).toHaveValue("new@example.com");
    expect(onAdded).not.toHaveBeenCalled();
  });

  it("imports CSV files and omits an unspecified budget from the request", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ results: [{ user_email: "csv@example.com", success: true }] });
    openImport();
    const file = new File(['email\n"csv@example.com"'], "students.csv", { type: "text/csv" });
    Object.defineProperty(file, "text", { value: async () => 'email\n"csv@example.com"' });
    fireEvent.change(screen.getByLabelText(/upload a CSV/), { target: { files: [file] } });
    fireEvent.click(await screen.findByRole("button", { name: "Confirm adding 1 members" }));
    expect(await screen.findByText("csv@example.com Added")).toBeInTheDocument();
    expect(apiClient.post).toHaveBeenCalledExactlyOnceWith("/team/bulk_member_add", {
      accessToken: "team-admin-token",
      body: { team_id: "course", members: [{ user_email: "csv@example.com", role: "user" }] },
    });
  });
  it("preserves success results when the refreshed team includes the new member", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ results: [{ user_email: "new@example.com", success: true }] });
    const props = { accessToken: "team-admin-token", teamId: "course", onAdded };
    const { rerender } = render(<BulkAddMembers {...props} members={[]} />);
    fireEvent.click(screen.getByRole("button", { name: "Bulk add members" }));
    fireEvent.change(screen.getByLabelText(/Paste emails/), { target: { value: "new@example.com" } });
    fireEvent.click(screen.getByRole("button", { name: "Confirm adding 1 members" }));
    expect(await screen.findByText("new@example.com Added")).toBeInTheDocument();
    rerender(<BulkAddMembers {...props} members={[{ user_email: "new@example.com", user_id: "new", role: "user" }]} />);
    expect(screen.getByText("new@example.com Added")).toBeInTheDocument();
  });

  it("blocks an oversized import and a negative budget", () => {
    openImport();
    fireEvent.change(screen.getByLabelText(/Paste emails/), {
      target: { value: Array.from({ length: 501 }, (_, index) => `user${index}@example.com`).join("\n") },
    });
    expect(screen.getByRole("button", { name: "Confirm adding 501 members" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/Paste emails/), { target: { value: "new@example.com" } });
    fireEvent.change(screen.getByLabelText(/Member budget/), { target: { value: "-1" } });
    expect(screen.getByRole("button", { name: "Confirm adding 1 members" })).toBeDisabled();
    expect(apiClient.post).not.toHaveBeenCalled();
  });
});

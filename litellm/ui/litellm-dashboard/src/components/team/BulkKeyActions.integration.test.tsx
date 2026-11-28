import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders as render } from "../../../tests/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
import { apiClient } from "@/components/networking";
import BulkKeyActions from "./BulkKeyActions";

vi.mock("@/components/networking", () => ({ apiClient: { post: vi.fn() } }));
const onUpdated = vi.fn();
const openActions = () => {
  render(
    <BulkKeyActions
      accessToken="team-admin-token"
      teamId="course"
      selectedKeys={["hash-a", "hash-b"]}
      onUpdated={onUpdated}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Bulk actions (2 keys)" }));
};
beforeEach(() => vi.clearAllMocks());

it("broadcasts a budget to only the selected team keys and exposes partial failures", async () => {
  vi.mocked(apiClient.post).mockResolvedValue({
    successful_updates: [{ key: "hash-a", key_info: {} }],
    failed_updates: [{ key: "hash-b", failed_reason: "Key no longer exists" }],
    total_requested: 2,
  });
  openActions();
  fireEvent.change(screen.getByLabelText("Value"), { target: { value: "0" } });
  fireEvent.click(screen.getByRole("button", { name: "Confirm for 2 keys" }));
  expect(await screen.findByText("1 of 2 keys updated.")).toBeInTheDocument();
  expect(screen.getByText(/Key no longer exists/)).toBeInTheDocument();
  expect(apiClient.post).toHaveBeenCalledExactlyOnceWith("/team/key/bulk_update", {
    accessToken: "team-admin-token",
    body: { team_id: "course", key_ids: ["hash-a", "hash-b"], update_fields: { max_budget: 0 } },
  });
  expect(onUpdated).toHaveBeenCalledOnce();
});

it("requires confirmation before deleting and sends one list of selected keys", async () => {
  vi.mocked(apiClient.post).mockResolvedValue({ deleted_keys: ["hash-a", "hash-b"] });
  openActions();
  fireEvent.change(screen.getByLabelText("Action"), { target: { value: "delete" } });
  expect(apiClient.post).not.toHaveBeenCalled();
  expect(screen.getByText("Deleted keys cannot be recovered.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Confirm for 2 keys" }));
  expect(await screen.findByText("2 of 2 keys updated.")).toBeInTheDocument();
  expect(apiClient.post).toHaveBeenCalledExactlyOnceWith("/key/delete", {
    accessToken: "team-admin-token",
    body: { keys: ["hash-a", "hash-b"] },
  });
});

it.each(["block", "unblock"])("continues after an individual %s failure and reports both outcomes", async (action) => {
  vi.mocked(apiClient.post).mockRejectedValueOnce(new Error("Permission denied")).mockResolvedValueOnce({});
  openActions();
  fireEvent.change(screen.getByLabelText("Action"), { target: { value: action } });
  fireEvent.click(screen.getByRole("button", { name: "Confirm for 2 keys" }));
  expect(await screen.findByText("1 of 2 keys updated.")).toBeInTheDocument();
  expect(screen.getByText(/Permission denied/)).toBeInTheDocument();
  expect(apiClient.post).toHaveBeenNthCalledWith(1, `/key/${action}`, {
    accessToken: "team-admin-token",
    body: { key: "hash-a" },
  });
  expect(apiClient.post).toHaveBeenNthCalledWith(2, `/key/${action}`, {
    accessToken: "team-admin-token",
    body: { key: "hash-b" },
  });
});

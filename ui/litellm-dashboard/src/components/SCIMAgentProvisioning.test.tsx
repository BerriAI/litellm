import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../tests/test-utils";
import { SCIMAgentProvisioning } from "./SCIMAgentProvisioning";
import { apiClient } from "./networking";
import { toast } from "@/lib/toast";

vi.mock("./networking", () => ({ apiClient: { get: vi.fn(), post: vi.fn(), put: vi.fn() } }));
vi.mock("@/lib/toast", () => ({ toast: { success: vi.fn(), fromError: vi.fn() } }));
vi.mock("@/components/common_components/AccessGroupSelector", () => ({ default: () => null }));

const source = {
  source_id: "source-one",
  display_name: "Engineering",
  tenant_id: "11111111-1111-4111-8111-111111111111",
  enabled: true,
  group_mappings: [],
};

describe("SCIM agent source configuration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    vi.mocked(apiClient.get).mockResolvedValue([]);
  });

  it("changes a newly created source through update and clears the submitted secret", async () => {
    vi.mocked(apiClient.post).mockResolvedValue(source);
    vi.mocked(apiClient.put).mockResolvedValue({ ...source, enabled: false });
    const user = userEvent.setup();
    renderWithProviders(<SCIMAgentProvisioning accessToken="admin-token" />);
    fireEvent.change(screen.getByLabelText("Source name"), { target: { value: source.display_name } });
    fireEvent.change(screen.getByLabelText("Entra tenant ID"), { target: { value: source.tenant_id } });
    fireEvent.change(screen.getByLabelText(/Dedicated SCIM token/), { target: { value: "test-scim-token" } });
    await user.click(screen.getByRole("button", { name: "Save provisioning source" }));
    await waitFor(() => expect(toast.success).toHaveBeenCalledOnce());
    expect(screen.getByLabelText("Entra tenant ID")).toBeDisabled();
    expect(screen.queryByLabelText(/Dedicated SCIM token/)).not.toBeInTheDocument();
    await user.click(screen.getByLabelText("Enable this provisioning source"));
    await user.click(screen.getByRole("button", { name: "Save provisioning source" }));
    await waitFor(() =>
      expect(apiClient.put).toHaveBeenCalledWith("/scim/v2/sources/source-one", {
        accessToken: "admin-token",
        body: { display_name: source.display_name, tenant_id: source.tenant_id, enabled: false, group_mappings: [] },
      }),
    );
    expect(apiClient.post).toHaveBeenCalledOnce();
    await user.click(screen.getByRole("button", { name: "New source" }));
    expect(screen.getByLabelText(/Dedicated SCIM token/)).toHaveValue("");
  });

  it("preserves an existing source when a save fails and shows the error", async () => {
    vi.mocked(apiClient.get).mockResolvedValue([source]);
    const failure = new Error("A mapped access group does not exist");
    vi.mocked(apiClient.put).mockRejectedValue(failure);
    const user = userEvent.setup();
    renderWithProviders(<SCIMAgentProvisioning accessToken="admin-token" />);
    await user.click(await screen.findByRole("button", { name: "Engineering" }));
    await user.click(screen.getByRole("button", { name: "Save provisioning source" }));
    await waitFor(() => expect(toast.fromError).toHaveBeenCalledWith(failure));
    expect(toast.success).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Entra tenant ID")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Save provisioning source" })).toBeEnabled();
  });
});

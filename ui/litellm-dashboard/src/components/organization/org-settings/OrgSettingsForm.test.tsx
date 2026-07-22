import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));
vi.mock("@/components/ModelSelect/ModelSelect", () => ({
  ModelSelect: ({ onChange }: { onChange: (values: string[]) => void }) => (
    <button type="button" onClick={() => onChange([])}>
      clear-models
    </button>
  ),
}));
vi.mock("@/components/vector_store_management/VectorStoreSelector", () => ({
  __esModule: true,
  default: ({ onChange }: { onChange: (values: string[]) => void }) => (
    <button type="button" onClick={() => onChange(["vs-2"])}>
      set-vector-stores
    </button>
  ),
}));
vi.mock("@/components/mcp_server_management/MCPServerSelector", () => ({
  __esModule: true,
  default: ({
    onChange,
  }: {
    onChange: (values: { servers: string[]; accessGroups: string[]; toolsets: string[] }) => void;
  }) => (
    <button type="button" onClick={() => onChange({ servers: ["srv-2"], accessGroups: [], toolsets: [] })}>
      set-mcp
    </button>
  ),
}));

import type { Organization } from "@/components/networking";

import { OrgSettingsForm } from "./OrgSettingsForm";

const org: Organization = {
  organization_id: "org-1",
  organization_alias: "acme",
  budget_id: "budget-1",
  metadata: {},
  models: ["gpt-5.2"],
  spend: 0,
  model_spend: {},
  created_at: "2026-01-01T00:00:00Z",
  created_by: "admin",
  updated_at: "2026-01-01T00:00:00Z",
  updated_by: "admin",
  litellm_budget_table: { max_budget: 100, budget_duration: "30d", tpm_limit: 1000, rpm_limit: 50 },
  teams: null,
  users: null,
  members: null,
  object_permission: {
    object_permission_id: "op-1",
    mcp_servers: ["srv-1"],
    mcp_access_groups: [],
    vector_stores: ["vs-1"],
  },
};

const renderForm = (overrides?: { patchOrganization?: ReturnType<typeof vi.fn>; onSaved?: () => void }) => {
  const patchOrganization = overrides?.patchOrganization ?? vi.fn().mockResolvedValue({});
  const onSaved = overrides?.onSaved ?? vi.fn();
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <OrgSettingsForm
        organizationId="org-1"
        org={org}
        accessToken="token"
        onCancel={vi.fn()}
        onSaved={onSaved}
        patchOrganization={patchOrganization}
      />
    </QueryClientProvider>,
  );
  return { patchOrganization, onSaved };
};

describe("OrgSettingsForm", () => {
  it("disables Save while the form is pristine", () => {
    renderForm();

    expect(screen.getByRole("button", { name: "Save Changes" })).toBeDisabled();
  });

  it("sends only the edited field", async () => {
    const user = userEvent.setup();
    const { patchOrganization } = renderForm();

    await user.clear(screen.getByLabelText("Organization Name"));
    await user.type(screen.getByLabelText("Organization Name"), "acme-2");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(patchOrganization).toHaveBeenCalledTimes(1));
    expect(patchOrganization).toHaveBeenCalledWith("org-1", { organization_alias: "acme-2" });
  });

  it("sends null when a limit is cleared", async () => {
    const user = userEvent.setup();
    const { patchOrganization } = renderForm();

    await user.clear(screen.getByLabelText("Tokens per minute Limit (TPM)"));
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(patchOrganization).toHaveBeenCalledTimes(1));
    expect(patchOrganization).toHaveBeenCalledWith("org-1", { tpm_limit: null });
  });

  it("sends models as [] when the selector is cleared", async () => {
    const user = userEvent.setup();
    const { patchOrganization } = renderForm();

    await user.click(screen.getByRole("button", { name: "clear-models" }));
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(patchOrganization).toHaveBeenCalledTimes(1));
    expect(patchOrganization).toHaveBeenCalledWith("org-1", { models: [] });
  });

  it("wraps a vector store change in object_permission without mcp keys", async () => {
    const user = userEvent.setup();
    const { patchOrganization } = renderForm();

    await user.click(screen.getByRole("button", { name: "set-vector-stores" }));
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(patchOrganization).toHaveBeenCalledTimes(1));
    expect(patchOrganization).toHaveBeenCalledWith("org-1", {
      object_permission: { vector_stores: ["vs-2"] },
    });
  });

  it("wraps an mcp change in object_permission with all three mcp keys", async () => {
    const user = userEvent.setup();
    const { patchOrganization } = renderForm();

    await user.click(screen.getByRole("button", { name: "set-mcp" }));
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(patchOrganization).toHaveBeenCalledTimes(1));
    expect(patchOrganization).toHaveBeenCalledWith("org-1", {
      object_permission: { mcp_servers: ["srv-2"], mcp_access_groups: [], mcp_toolsets: [] },
    });
  });

  it("does not send a patch when an edit is reverted to the original value", async () => {
    const user = userEvent.setup();
    renderForm();

    const alias = screen.getByLabelText("Organization Name");
    await user.clear(alias);
    await user.type(alias, "acme");

    await waitFor(() => expect(screen.getByRole("button", { name: "Save Changes" })).toBeDisabled());
  });

  it("blocks submit and shows an error for invalid metadata JSON", async () => {
    const user = userEvent.setup();
    const { patchOrganization } = renderForm();

    await user.type(screen.getByLabelText("Metadata"), "not json");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Metadata must be a valid JSON object");
    expect(patchOrganization).not.toHaveBeenCalled();
  });

  it("keeps the view open when the patch fails", async () => {
    const user = userEvent.setup();
    const onSaved = vi.fn();
    const { patchOrganization } = renderForm({
      patchOrganization: vi.fn().mockRejectedValue(new Error("boom")),
      onSaved,
    });

    await user.clear(screen.getByLabelText("Organization Name"));
    await user.type(screen.getByLabelText("Organization Name"), "acme-2");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(patchOrganization).toHaveBeenCalledTimes(1));
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("calls onSaved after a successful patch", async () => {
    const user = userEvent.setup();
    const onSaved = vi.fn();
    renderForm({ onSaved });

    await user.clear(screen.getByLabelText("Requests per minute Limit (RPM)"));
    await user.type(screen.getByLabelText("Requests per minute Limit (RPM)"), "75");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1));
  });
});
